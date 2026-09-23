"""Does Laya find signal once the state says where the work is now?

Decision point `now` is 1-40 calls after the call. Label = the call's target
comes up again AFTER now (in a later call's input or in later narration).
Same rows scored with state A (the call alone) and state B (+ distance, recent
steps, current narration). Nothing is written; states only go to the local model.
"""
import json, os, random, sys, time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import jevlab as j

N = int(os.environ.get("N", "150"))
random.seed(3)


def name_of(call):
    first = call.target.split()[0] if call.target.split() else ""
    return os.path.basename(first)


def label_at(calls, texts, i, now):
    name = name_of(calls[i])
    if len(name) <= 3:
        return None
    later = "\n".join(texts[calls[now].text_index:])
    return int(name in later or any(name in json.dumps(c.input) for c in calls[now + 1:]))


pos, neg = [], []
for path in j.session_files(j.SESSIONS):
    goal, calls, texts = j.load_session(path)
    for i, call in enumerate(calls):
        if call.tool.startswith("mcp__") or j.tool_name(call.tool) == "other":
            continue
        now = i + random.randint(1, 40)
        if now >= len(calls) - 1:
            continue
        y = label_at(calls, texts, i, now)
        if y is None:
            continue
        (pos if y else neg).append((goal, calls, i, now, y))
print(f"candidates keep {len(pos)} drop {len(neg)}")
rows = random.sample(pos, min(N // 2, len(pos))) + random.sample(neg, min(N // 2, len(neg)))


def state_b(goal, calls, i, now):
    call, cur = calls[i], calls[now]
    recent = [f"{c.tool} {j.redact(c.target)[:90]}" for c in calls[max(0, now - 4): now + 1]]
    base = j.state_of(call, goal)
    return {
        "goal": base["goal"],
        "calls_since": now - i,
        "agent_now": j.redact(" ".join(cur.narration.split()))[:250],
        "recent_steps": recent,
        "tool": base["tool"],
        "input": base["input"][:300],
        "result_chars": base["result_chars"],
        "result": j.preview(j.redact(call.result))[:700],
    }


def q_b(call):
    q = j.questions(call)
    q["keep_result"]["instructions"] = (
        "The agent made this call `calls_since` calls ago. What it is doing now is in "
        "`agent_now` and `recent_steps`. It needs the full output of this "
        f"{j.tool_name(call.tool)} call kept verbatim in its context."
    )
    return {"keep_result": q["keep_result"]}


def auc(scores, ys):
    p = [s for s, y in zip(scores, ys) if y]
    n = [s for s, y in zip(scores, ys) if not y]
    return sum((a > b) + 0.5 * (a == b) for a in p for b in n) / max(1, len(p) * len(n))


agent = j.load_agent()
ys, sa, sb, age, inlen = [], [], [], [], []
t = time.time()
for goal, calls, i, now, y in rows:
    call = calls[i]
    qa = {k: v for k, v in j.questions(call).items() if k == "keep_result"}
    sa.append(agent.predict(j.state_of(call, goal), qa)["answers"]["keep_result"].get("noul", 0.0))
    stb = state_b(goal, calls, i, now)
    assert len(json.dumps(stb)) < 2600, len(json.dumps(stb))
    sb.append(agent.predict(stb, q_b(call))["answers"]["keep_result"].get("noul", 0.0))
    ys.append(y); age.append(-(now - i)); inlen.append(-len(j.state_of(call, goal)["input"]))
print(f"{len(ys)} rows, {(time.time() - t) / len(ys):.2f} s/row (two predictions)")
sd = lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5
print(f"A  call alone        AUC {auc(sa, ys):.3f}   score std {sd(sa):.3f}")
print(f"B  + where work is   AUC {auc(sb, ys):.3f}   score std {sd(sb):.3f}")
print(f"baseline age (newer=keep)   AUC {auc(age, ys):.3f}")
print(f"baseline short input        AUC {auc(inlen, ys):.3f}")
