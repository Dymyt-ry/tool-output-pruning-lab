"""Does SWE-Pruner keep the lines the agent went on to use, better than head+tail
at the same size? Long Bash/Read outputs from local sessions; nothing is written."""
import json, os, re, sys, time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import jevlab as j
from pruner import CodePruner

LIMIT = int(os.environ.get("N", "30"))
THRESHOLDS = [float(t) for t in os.environ.get("T", "0.3,0.5,0.7").split(",")]
READ_PREFIX = re.compile(r"^\s*\d+\t", re.M)  # Read output carries "  12\t" line numbers
TOK = re.compile(r"[\w./:@-]{8,}")

jobs = []
for path in j.session_files(j.SESSIONS):
    goal, calls, texts = j.load_session(path)
    for call in calls:
        if call.tool not in ("Bash", "Read") or len(call.result) < 4000 or call.is_error:
            continue
        body = READ_PREFIX.sub("", call.result) if call.tool == "Read" else call.result
        desc = call.input.get("description") or call.input.get("file_path", "")
        query = " ".join(f"{desc}. {' '.join(call.narration.split())[:300]}".split())
        # "used" = the line carries a distinctive token (path, number, version,
        # identifier with a digit or underscore) that the agent brought up again
        # later, in what it wrote or in a later call - and that it did not already
        # have before this call (its own input, the narration the query is built from)
        idx = calls.index(call)
        later = "\n".join(texts[call.text_index:]) + "\n".join(json.dumps(c.input) for c in calls[idx + 1:])
        known = json.dumps(call.input) + call.narration + query
        used = [l for l in body.splitlines()
                if any(t in later and t not in known for t in TOK.findall(l) if not t.isalpha())]
        if not used:
            continue
        if os.environ.get("ORACLE"):
            # the agent's next message: it names what it used, so this is a ceiling
            query = " ".join(texts[call.text_index].split())[:1000] if call.text_index < len(texts) else query
        jobs.append((call.tool, query, body, used))
jobs = jobs[:LIMIT]
if not jobs:
    sys.exit("no long outputs with used lines")
print(f"{len(jobs)} long outputs with lines used later "
      f"({sum(1 for t, *_ in jobs if t == 'Bash')} Bash, {sum(1 for t, *_ in jobs if t == 'Read')} Read)")

t0 = time.time()
pruner = CodePruner()
print(f"model load {time.time() - t0:.1f} s")
totals = {t: [0, 0, 0, 0] for t in THRESHOLDS}  # chars in, chars out, used kept (pruner), used kept (head+tail)
used_total, elapsed = 0, 0.0
for tool, query, body, used in jobs:
    t1 = time.time()
    scores = pruner.line_scores(query, body)
    elapsed += time.time() - t1
    used_total += len(used)
    for t in THRESHOLDS:
        # reuse the scores for every threshold rather than re-running the model
        pruner.line_scores = lambda q, x, s=scores: s
        cut = pruner.prune(query, body, t)
        base = j.head_tail(body, len(cut))
        row = totals[t]
        row[0] += len(body); row[1] += len(cut)
        row[2] += sum(1 for line in used if line in cut)
        row[3] += sum(1 for line in used if line in base)
    del pruner.line_scores  # restore the method

print(f"latency {elapsed / len(jobs):.1f} s per output, {used_total} used lines in total\n")
print(f"{'thresh':>6} {'kept size':>10} {'pruner keeps':>13} {'head+tail keeps':>16}")
for t, (cin, cout, kp, kb) in totals.items():
    print(f"{t:>6} {cout / cin:>9.0%} {kp / used_total:>12.0%} {kb / used_total:>15.0%}")
