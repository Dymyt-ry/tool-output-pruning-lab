"""Hindsight retention (2026-09-24). Circular - see circularity.py and the README.

At a decision point K calls after a long output, keep the lines whose distinctive
tokens (path, number, identifier with a digit or underscore) the agent has used
since the call, then fill from both ends; compare with head+tail at the same size
on the lines used AFTER the decision point. No model.

The weakness, checked in circularity.py: a line counts as "needed" when its token
comes up later, and a token the agent already wrote itself is in context anyway.
Excluding those uses, the win over head+tail disappears.
"""
import collections, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jevlab as j

RP = re.compile(r"^\s*\d+\t", re.M)
TOK = re.compile(r"[\w./:@-]{8,}")
SIZES = (0.54, 0.32, 0.15)

for K in (5, 10, 20):
    res = collections.defaultdict(lambda: [0] * len(SIZES))
    tot = n = 0
    for path in j.session_files(j.SESSIONS):
        goal, calls, texts = j.load_session(path)
        for idx, call in enumerate(calls):
            if call.tool not in ("Bash", "Read") or len(call.result) < 4000 or call.is_error:
                continue
            now = idx + K
            if now >= len(calls) - 1:
                continue
            body = RP.sub("", call.result) if call.tool == "Read" else call.result
            known = json.dumps(call.input) + call.narration
            past = "\n".join(texts[call.text_index : calls[now].text_index]) + "\n".join(
                json.dumps(c.input) for c in calls[idx + 1 : now + 1])
            future = "\n".join(texts[calls[now].text_index :]) + "\n".join(json.dumps(c.input) for c in calls[now + 1 :])
            lines = body.splitlines()
            toks = [[t for t in TOK.findall(l) if not t.isalpha() and t not in known] for l in lines]
            used = [i for i, ts in enumerate(toks) if any(t in future for t in ts)]
            if not used:
                continue
            n += 1
            tot += len(used)
            proven = [1.0 if any(t in past for t in ts) else 0.0 for ts in toks]
            L = len(lines)
            for k, s in enumerate(SIZES):
                budget = s * len(body)
                ht = j.head_tail(body, int(budget))
                order = sorted(range(L), key=lambda i: (-proven[i], min(i, L - 1 - i)))
                keep, size = set(), 0
                for i in order:
                    if size + len(lines[i]) + 1 > budget:
                        continue
                    keep.add(i)
                    size += len(lines[i]) + 1
                res["head+tail"][k] += sum(1 for i in used if lines[i] in ht)
                res["hindsight + ends"][k] += sum(1 for i in used if i in keep)
    print(f"decision {K} calls after the output: {n} outputs, {tot} lines used after that point")
    print(f"  {'method':<20}" + "".join(f"{f'size {s:.0%}':>11}" for s in SIZES))
    for name, v in res.items():
        print(f"  {name:<20}" + "".join(f"{x / max(tot, 1):>11.0%}" for x in v))
