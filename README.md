# tool-output-pruning-lab

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Claude Code](https://img.shields.io/badge/Claude%20Code-transcripts-D97757)
![Selectors tested](https://img.shields.io/badge/selectors%20tested-6-orange)
![Result](https://img.shields.io/badge/result-negative-critical)
![Baseline](https://img.shields.io/badge/baseline-head%2Btail%20holds-brightgreen)
![Data](https://img.shields.io/badge/transcripts%20published-none-lightgrey)

**Can a model decide which lines of a Claude Code tool output the agent will need later?**
On real sessions, measured against the free baseline of keeping the first and last part of
the output: **not in any way that matters.** Four learned selectors, a word-overlap selector
handed the agent's *next* message, and a "decide later" rule were tested. The only one ahead of
head+tail cut to the same size is the word-overlap oracle, by 3 points at the smallest cut. It
sees a message no real selector has yet.

This started as a fork of the idea behind
[tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) (score each
tool call, drop the stale ones, keep the rest verbatim) with a local model in place of the
Jev API. It ended as a measurement of why that idea has so little to work with.

## Results at a glance

### Line selectors: which lines of a long output survive a cut

"Kept" = share of the lines the agent used later that survive. Each row compares a selector
with head+tail **on the same outputs at the same size**. Compare within a row only: SWE-Pruner
sets its own size per output (the sizes shown are totals), the other rows cut every output
at a fixed fraction, so head+tail scores differ between rows.

| Selector | Question it gets | Size of the cut | Kept | head+tail |
|---|---|---|---|---|
| [SWE-Pruner](https://github.com/Ayanami1314/swe-pruner) 0.6B | call-time question | 74 / 54 / 32 % | 63 / 40 / 16 % | 66 / 41 / 16 % |
| SWE-Pruner | **oracle**: the agent's next message | 69 / 42 / 21 % | 65 / 35 / 12 % | 65 / 35 / 13 % |
| [Needle 3](https://github.com/cactus-compute/needle) 121M embeddings | call-time question | 74 / 54 / 32 % | 77 / 56 / 34 % | 82 / 62 / 37 % |
| Needle 3 | **oracle** | 74 / 54 / 32 % | 75 / 55 / 31 % | 82 / 62 / 37 % |
| Word overlap, no model | call-time question | 74 / 54 / 32 % | 73 / 55 / 34 % | 82 / 62 / 37 % |
| Word overlap, no model | **oracle** | 74 / 54 / 32 % | 76 / 59 / **40** % | 82 / 62 / **37** % |
| Hindsight rule, no model, circular uses removed | what the agent used in the next K calls | 15 % | 16-18 % | 17-24 % |

### Per-call keep/drop: does [Laya](https://github.com/NandhaKishorM/laya) 421M know which results are still needed?

| State it sees | Labels | Laya AUC | Trivial rules on the same labels |
|---|---|---|---|
| the call + a 600/200-char preview of its output, zero-shot | harvested keep/drop, 645 pairs | **0.47** | short input is kept 0.62 |
| + calls since, the last 5 steps, the current narration | target comes up again after a decision point 1-40 calls later (`context_test.py`) | **0.45** | newer is kept 0.55, short input is kept 0.59 |

## The one finding

**When a tool output arrives, nothing in it or around it says which lines will matter.** That
is written by the rest of the session. The oracle rows show it best. The message the agent wrote
*next* names what it went on to use. Handed that message, a trained pruner, an embedding model
and plain word overlap still end up within a few points of keeping the ends, on either side.
A model at the PostToolUse hook has less to go on than those oracles do.

Deciding later instead is what upstream does (score at compaction time, with the whole history
in view). Here that was tried as a rule with no model. K calls after an output, keep the lines
whose paths, numbers and identifiers the agent has used since, and fill the rest from both ends.
It looked like a win: 43 % of the needed lines kept at 15 % size against 19 % for head+tail
(K=20). It is circular. At K=20, 38 % of those "needed" lines carry a token the agent had already
typed itself between the output and the decision point, so it is in the context in the agent's
own words anyway. Other lines were brought back by another tool result first (the definition
from [fast-jev-compaction#26](https://github.com/tamaratran/fast-jev-compaction/issues/26)).
Count only the uses that neither covers, and hindsight keeps 16-18 % against 17-24 %:

| K calls later | needed lines (all / not own / fresh) | hindsight at 15 % | head+tail at 15 % |
|---|---|---|---|
| 5 | 609 / 490 / 230 | 29 / 16 / 18 % | 20 / 20 / 23 % |
| 10 | 529 / 373 / 168 | 37 / 16 / 16 % | 20 / 21 / 24 % |
| 20 | 457 / 283 / 124 | 43 / 16 / 16 % | 19 / 17 / 21 % |

Reading the cases by hand agrees. The lines hindsight "won" are the circular ones, and many
"fresh" hits are generic identifiers that match by chance (`JSON.parse`, `process.exit`).

## How it was measured

- **Data:** the author's own Claude Code sessions on one machine (`~/.claude/projects/**/*.jsonl`,
  read-only), 23 sessions over 50 kB and ~1,000 tool calls at the time of writing. The line benches
  use the first 40 long (≥ 4,000 chars) successful Bash/Read outputs that have at least one
  later-used line: 35 Bash, 5 Read, 771 used lines.
- **"Used later":** a line counts when it carries a distinctive token that the agent brings up
  again later, in its own text or in a later call's input. A distinctive token is a path, a
  number, or an identifier with a digit or underscore, of 8 or more characters. Tokens the agent
  already had before the call (the call's input and its narration) are excluded. In these
  outputs no line was quoted verbatim later, so this proxy is all there is. It over-counts
  coincidences and misses paraphrase.
- **Baseline:** `head_tail` keeps 70 % of the budget from the start and 30 % from the end.
- **Laya labels:** `harvest` labels each call from what the session did next.
  - *Gold* labels are behaviour that settles it: edited later without re-reading, quoted later,
    or fetched again. Only 13 of 645 pairs have one.
  - *Proxy* labels say "the call's target comes up again". The proxy is gameable: input length
    alone scores AUC 0.62 on it, because short targets such as file paths are what comes up
    again. Fine-tuning on it would learn that.
- **Laya is not broken:** on trivial questions it answers 0.62 for true and 0.13 for false. For
  this judgement it has no signal. `keep_result` sits in 0.43-0.62 for every call, and a 0.5
  threshold splits the calls about at random. Laya 0.3.8 was used, which has since left PyPI;
  0.3.11 truncates the state the same way.
- **Hardware:** RTX 3050 6 GB. SWE-Pruner takes 0.6 s per output on the GPU and 49 s on the CPU.

It is a small sample from one user and one machine, with a noisy proxy. No row shows a
meaningful win over head+tail. Most rows are ties or small losses, and the one small win needs
an oracle.

## What already exists

If you want less tool output in the context, these tools are already built. None of them needs
a selector to win:

- Tools that store the full output and give the agent a way to fetch it back:
  - [context-mode](https://github.com/mksglu/context-mode)
  - [Sando](https://github.com/yuzushi-dev/Sando)
  - [lm-resizer](https://github.com/phuetz/lm-resizer)
  - fast-jev-compaction [PR #18](https://github.com/tamaratran/fast-jev-compaction/pull/18): the
    trimmed Bash output is saved to a file, with a marker telling the agent where it is
  - [pi-jev-compaction](https://github.com/nourhelmi/pi-jev-compaction): `jev_read`, Pi only
- Claude Code itself clears old tool results (`[Old tool result content cleared]`).

Replays posted in the upstream tracker by contributors show the same limits and one thing that
helps:

- In [#26](https://github.com/tamaratran/fast-jev-compaction/issues/26), the Jev API with the
  default questions scored like a fake model that answers 0 to everything.
- Rewording the questions gave it some signal: [#52](https://github.com/tamaratran/fast-jev-compaction/issues/52)
  went from 0 to 5 results kept on a 120-message replay, and
  [PR #61](https://github.com/tamaratran/fast-jev-compaction/pull/61) went from 0-1 to 2-3 of 4
  must-keep results.
- A 300-char preview of each result lifted the AUC from 0.61 to 0.71 on a similar proxy
  ([PR #57](https://github.com/tamaratran/fast-jev-compaction/pull/57)).

## Run it on your own sessions

```sh
./setup.sh                                       # .venv with laya, ~3 GB, inside this directory
.venv/bin/python jevlab.py selftest              # parser, decision logic and redaction, no model
.venv/bin/python jevlab.py replay ~/.claude/projects/<proj>/<session>.jsonl --limit 40
.venv/bin/python experiments/lexical_oracle.py   # no model: word overlap vs head+tail
.venv/bin/python experiments/hindsight.py        # no model: the hindsight rule
.venv/bin/python experiments/circularity.py --show 5   # ...and how much of it is circular
N=150 .venv/bin/python experiments/context_test.py     # Laya with and without context
```

SWE-Pruner and Needle 3 need a second environment (`torch==2.14.0`, `transformers==5.17.0`,
`safetensors==0.8.0`) and their weights under `.models/`. Neither package is installed; both
are loaded directly:

| Directory | Source | Pin |
|---|---|---|
| `.models/code-pruner/` | [`ayanami-kitasan/code-pruner`](https://huggingface.co/ayanami-kitasan/code-pruner), all files | revision `863cee3`, `model.safetensors` sha256 `373b77f5262c3298b803303d49ea38949d3e92f08dc3bbb90b03f490413adae9` |
| `.models/qwen3-reranker-config/` | `config.json` of [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), the backbone | — |
| `.models/needle3/` | [`Cactus-Compute/needle3`](https://huggingface.co/Cactus-Compute/needle3): `needle3.cact`, and `libneedle3.so` from the `cactus_needle-3.0.1` wheel in the same repo | commit `b274efcb211a9eef48c9a88da4b43bd569696a39`; `needle3.cact` sha256 `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38` |

```sh
N=40 python experiments/prune_bench.py            # add ORACLE=1 for the next-message question
unshare -rn python experiments/needle_bench.py    # no network for the native library
```

Everything reads your transcripts read-only and writes only inside this directory. Nothing is
installed into Claude Code: there is no hook, plugin or service. To uninstall, delete the
directory.

## Training data, if you want to try a fine-tune anyway

`harvest` writes `{state, questions, answers}` pairs in the shape of Laya's
[fine-tune notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
Hold the gold rows out. The redaction on the way out was red-teamed twice. It does this:

- credentials, keys, JWTs, connection strings, emails and phone numbers → `[REDACTED]`;
- IPs, MACs, commit hashes and random ids → `[IP]`, `[MAC]`, `[HEX]`, `[ID]`;
- home paths, project directories, hostnames and every name the machine knows → salted, stable
  pseudonyms. The names come from the machine's user, its project directories, its
  `~/.ssh/config` hosts, its git identity and a curated list. They are matched through Czech
  declension and code compounds, and the whole token is replaced, so no residue rebuilds the
  word;
- whole rows are dropped when the *topic* leaks. That covers prices, invoices, contracts,
  clients, leaderboards, whois, git history, security findings, machine inventory, memory files,
  dictation, and every MCP tool result.

`audit` re-reads a pair file with wider detectors than `redact` and lists what it cannot see.
Read the rows by hand anyway: every real leak found in this project was found by reading.
No pair file, denylist or salt is part of this repository.

## Layout

| File | What it is |
|---|---|
| [`jevlab.py`](jevlab.py) | session parser, Laya replay and trim, harvest, redaction, audit, selftest |
| [`pruner.py`](pruner.py) | SWE-Pruner forward pass reimplemented from the published architecture |
| [`needle_embed.py`](needle_embed.py) | Needle 3 embeddings through its C API via ctypes |
| [`experiments/`](experiments) | one script per row of the tables above |

## Credits and license

MIT, see [LICENSE](LICENSE). The models are the work of their authors: SWE-Pruner (MIT), Needle 3
(Apache-2.0) by Cactus Compute, and Laya (Apache-2.0) by Convai Innovations. None of their code
is vendored here. The idea of per-call keep/drop decisions is from
[fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction).
