"""Is the hindsight win circular? (2026-09-24)

hindsight.py counts a line as needed after the decision point when one of its tokens
comes up later. Two of those uses cost nothing to lose:
  own       - the agent already wrote the token itself between the output and the
              decision point, so it is in the context in the agent's own words;
  recarried - another tool result carried the token before the agent used it
              (the definition from fast-jev-compaction#26).
Scored three ways: all uses (hindsight.py), without own, without own and recarried.
`--show N` prints N cases per kind for reading by hand.
"""
import argparse, collections, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jevlab as j

RP = re.compile(r"^\s*\d+\t", re.M)
TOK = re.compile(r"[\w./:@-]{8,}")
SIZES = (0.54, 0.32, 0.15)
DEFS = ("all uses", "not own", "not own/recarried")


def timeline(calls, texts):
    """Events in order: ('agent', text) for assistant text and tool inputs, ('tool', result).
    calls_at[i] = position of call i's input event."""
    ev, calls_at, t = [], [], 0
    for c in calls:
        while t < c.text_index:
            ev.append(("agent", texts[t]))
            t += 1
        calls_at.append(len(ev))
        ev.append(("agent", json.dumps(c.input)))
        ev.append(("tool", c.result))
    ev.extend(("agent", x) for x in texts[t:])
    return ev, calls_at


def classify(tok, ev, start, now):
    """Why a token counts after the decision point: None (never used), 'own', 'recarried', 'fresh'."""
    own = any(k == "agent" and tok in x for k, x in ev[start:now])
    carried = any(k == "tool" and tok in x for k, x in ev[start:now])
    for k, x in ev[now:]:
        if tok not in x:
            continue
        if k == "tool":
            carried = True
            continue
        return "own" if own else "recarried" if carried else "fresh"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    args = ap.parse_args()
    sessions = [j.load_session(p) for p in j.session_files(j.SESSIONS)]
    for K in args.k:
        res = {d: collections.defaultdict(lambda: [0] * len(SIZES)) for d in DEFS}
        tot = dict.fromkeys(DEFS, 0)
        kinds = collections.Counter()
        cases = collections.defaultdict(list)
        n = 0
        for goal, calls, texts in sessions:
            ev, calls_at = timeline(calls, texts)
            for idx, call in enumerate(calls):
                if call.tool not in ("Bash", "Read") or len(call.result) < 4000 or call.is_error:
                    continue
                now = idx + K
                if now >= len(calls) - 1:
                    continue
                body = RP.sub("", call.result) if call.tool == "Read" else call.result
                known = json.dumps(call.input) + call.narration
                # same windows as hindsight.py: the output's own result event is skipped
                start, cut = calls_at[idx] + 2, calls_at[now] + 1
                past = "\n".join(x for k, x in ev[start:cut] if k == "agent")
                lines = body.splitlines()
                toks = [[t for t in TOK.findall(l) if not t.isalpha() and t not in known] for l in lines]
                why = [{t: classify(t, ev, start, cut) for t in ts} for ts in toks]
                line_kind = []
                for w in why:
                    got = set(w.values()) - {None}
                    line_kind.append("fresh" if "fresh" in got else "recarried" if "recarried" in got
                                     else "own" if got else None)
                needed = {
                    "all uses": [i for i, x in enumerate(line_kind) if x],
                    "not own": [i for i, x in enumerate(line_kind) if x in ("fresh", "recarried")],
                    "not own/recarried": [i for i, x in enumerate(line_kind) if x == "fresh"],
                }
                if not needed["all uses"]:
                    continue
                n += 1
                kinds.update(x for x in line_kind if x)
                proven = [1.0 if any(t in past for t in ts) else 0.0 for ts in toks]
                L = len(lines)
                keeps = []
                for s in SIZES:
                    budget = s * len(body)
                    ht = j.head_tail(body, int(budget))
                    order = sorted(range(L), key=lambda i: (-proven[i], min(i, L - 1 - i)))
                    keep, size = set(), 0
                    for i in order:
                        if size + len(lines[i]) + 1 > budget:
                            continue
                        keep.add(i)
                        size += len(lines[i]) + 1
                    keeps.append((ht, keep))
                for d in DEFS:
                    tot[d] += len(needed[d])
                    for k, (ht, keep) in enumerate(keeps):
                        res[d]["head+tail"][k] += sum(1 for i in needed[d] if lines[i] in ht)
                        res[d]["hindsight + ends"][k] += sum(1 for i in needed[d] if i in keep)
                if args.show:
                    ht, keep = keeps[2]
                    for i in needed["all uses"]:
                        kind = line_kind[i]
                        tag = ("won" if i in keep and lines[i] not in ht else
                               "lost" if i not in keep and lines[i] in ht else None)
                        if tag:
                            cases[(kind, tag)].append((call, lines[i], why[i], past, idx))
        print(f"decision {K} calls after the output: {n} outputs; needed lines by kind {dict(kinds)}")
        for d in DEFS:
            print(f"  [{d}] {tot[d]} lines")
            print(f"    {'method':<20}" + "".join(f"{f'size {s:.0%}':>11}" for s in SIZES))
            for name, v in res[d].items():
                print(f"    {name:<20}" + "".join(f"{x / max(tot[d], 1):>11.0%}" for x in v))
        for key in sorted(cases):
            print(f"\n### {key[0]} line, hindsight {key[1]} vs head+tail at 15% ({len(cases[key])} cases)")
            for call, line, w, past, idx in cases[key][:: max(1, len(cases[key]) // args.show)][: args.show]:
                cmd = call.input.get("command") or call.input.get("file_path") or ""
                print(f"- call#{idx} {call.tool} {str(cmd)[:90]!r}")
                print(f"  line: {line.strip()[:160]!r}")
                for t, why_t in w.items():
                    if why_t:
                        at = past.find(t)
                        ctx = past[max(0, at - 60): at + len(t) + 40].replace("\n", " ") if at >= 0 else ""
                        print(f"  tok {t[:60]!r}: {why_t}" + (f" | past: {ctx!r}" if ctx else ""))


if __name__ == "__main__":
    main()
