"""Word overlap with the agent's NEXT message (an oracle - it names what the agent
went on to use): rank lines by it, keep the best up to the size head+tail gets.
Not an upper bound - a perfect oracle keeps everything - but it sees more than
any scorer at output time can. If even this barely moves head+tail, the
information at output time is thin. Same jobs as needle_bench.py; no model."""
import json, re, sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import jevlab as j

READ_PREFIX = re.compile(r"^\s*\d+\t", re.M)
TOK = re.compile(r"[\w./:@-]{8,}")
WORD = re.compile(r"[\w./:@-]{4,}")
SIZES = (0.74, 0.54, 0.32)

jobs = []
for path in j.session_files(j.SESSIONS):
    goal, calls, texts = j.load_session(path)
    for idx, call in enumerate(calls):
        if call.tool not in ("Bash", "Read") or len(call.result) < 4000 or call.is_error:
            continue
        body = READ_PREFIX.sub("", call.result) if call.tool == "Read" else call.result
        desc = call.input.get("description") or call.input.get("file_path", "")
        query = " ".join(f"{desc}. {' '.join(call.narration.split())[:300]}".split())
        later = "\n".join(texts[call.text_index:]) + "\n".join(json.dumps(c.input) for c in calls[idx + 1:])
        known = json.dumps(call.input) + call.narration + query
        used = [l for l in body.splitlines() if any(t in later and t not in known for t in TOK.findall(l) if not t.isalpha())]
        if not used:
            continue
        oracle = " ".join(texts[call.text_index].split())[:1000] if call.text_index < len(texts) else query
        jobs.append((query, oracle, body, used))
jobs = jobs[:40]
print(f"{len(jobs)} outputs")

rows = {(name, s): 0 for name in ("head+tail", "overlap, call-time question", "overlap, oracle question") for s in SIZES}
used_total = 0
for query, oracle, body, used in jobs:
    lines = body.splitlines()
    L = len(lines)
    used_total += len(used)
    for name, q in (("overlap, call-time question", query), ("overlap, oracle question", oracle)):
        qw = {w.lower() for w in WORD.findall(q)}
        score = [len({w.lower() for w in WORD.findall(l)} & qw) for l in lines]
        # ties go to the ends, so a zero-overlap line is chosen the way head+tail would
        order = sorted(range(L), key=lambda i: (-score[i], min(i, L - 1 - i)))
        for s in SIZES:
            budget, keep, size = s * len(body), set(), 0
            for i in order:
                if size + len(lines[i]) + 1 > budget:
                    continue
                keep.add(i); size += len(lines[i]) + 1
            cut = "\n".join(lines[i] for i in sorted(keep))
            rows[(name, s)] += sum(1 for u in used if u in cut)
    for s in SIZES:
        base = j.head_tail(body, int(s * len(body)))
        rows[("head+tail", s)] += sum(1 for u in used if u in base)
print(f"{used_total} used lines\n")
print(f"{'method':<30}" + "".join(f"{f'size {s:.0%}':>12}" for s in SIZES))
for name in ("head+tail", "overlap, call-time question", "overlap, oracle question"):
    print(f"{name:<30}" + "".join(f"{rows[(name, s)] / used_total:>12.0%}" for s in SIZES))
