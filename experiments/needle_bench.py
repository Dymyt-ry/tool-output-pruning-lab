"""Needle 3 embeddings as a line selector: keep the lines closest to the question,
up to the same size head+tail gets, and count the later-used lines each keeps.
Two questions: what was known at call time, and the agent's next message (an
oracle - it leaks the answer, so it is a ceiling, not a result)."""
import json, re, sys, time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import jevlab as j
from needle_embed import Needle

READ_PREFIX = re.compile(r"^\s*\d+\t", re.M)
TOK = re.compile(r"[\w./:@-]{8,}")
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

needle = Needle()
t0 = time.time()
rows = {("head+tail", s): [0, 0] for s in SIZES}
for name in ("needle, call-time question", "needle, oracle question"):
    for s in SIZES:
        rows[(name, s)] = [0, 0]
used_total = 0
for query, oracle, body, used in jobs:
    lines = body.splitlines()
    vecs = [needle.embed(l) if l.strip() else None for l in lines]
    real = [v for v in vecs if v]
    mean = [sum(c) / len(real) for c in zip(*real)]
    center = lambda v: [x - m for x, m in zip(v, mean)]
    used_total += len(used)
    for name, q in (("needle, call-time question", query), ("needle, oracle question", oracle)):
        qv = center(needle.embed(q))
        score = [sum(a * b for a, b in zip(center(v), qv)) if v else -1e9 for v in vecs]
        order = sorted(range(len(lines)), key=lambda i: -score[i])
        for s in SIZES:
            budget, keep, size = s * len(body), set(), 0
            for i in order:
                if size + len(lines[i]) + 1 > budget:
                    continue
                keep.add(i); size += len(lines[i]) + 1
            cut = "\n".join(lines[i] for i in sorted(keep))
            rows[(name, s)][0] += sum(1 for u in used if u in cut)
            rows[(name, s)][1] += len(cut)
    for s in SIZES:
        base = j.head_tail(body, int(s * len(body)))
        rows[("head+tail", s)][0] += sum(1 for u in used if u in base)
print(f"{(time.time() - t0) / len(jobs):.1f} s per output, {used_total} used lines\n")
total = sum(len(b) for _, _, b, _ in jobs)
for name in ("needle, call-time question", "needle, oracle question"):
    print(f"{name} kept size: " + " / ".join(f"{rows[(name, s)][1] / total:.0%}" for s in SIZES))
print(f"{'method':<30}" + "".join(f"{f'size {s:.0%}':>12}" for s in SIZES))
for name in ("head+tail", "needle, call-time question", "needle, oracle question"):
    print(f"{name:<30}" + "".join(f"{rows[(name, s)][0] / used_total:>12.0%}" for s in SIZES))
