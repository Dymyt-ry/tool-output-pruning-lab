"""SWE-Pruner inference, reimplemented from the published architecture.

The model and its design are from https://github.com/Ayanami1314/swe-pruner
(MIT, Copyright the SWE-Pruner authors); weights from the Hugging Face repo
ayanami-kitasan/code-pruner. Their package is not installed: an osv-scanner pass over
its lockfile (2026-09-23) reported 669 known vulnerabilities, and inference needs none of it. This
file is the forward pass only - backbone, three-layer fusion, one attention
layer, the CRF head's emissions - plus their line aggregation, with the same
prompt so the scores mean what they were trained to mean.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / ".models" / "code-pruner"
BACKBONE_CONFIG = HERE / ".models" / "qwen3-reranker-config"

INSTRUCTION = "Given a query, judge if the document(code) is related to query."
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Pruner(nn.Module):
    def __init__(self, fused_layers=(0.25, 0.5), bottleneck: int = 256, heads: int = 8):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(BACKBONE_CONFIG)
        self.backbone = AutoModel.from_config(config, attn_implementation="sdpa")
        n = config.num_hidden_layers
        self.layers = (max(1, int(n * fused_layers[0])), max(1, int(n * fused_layers[1])), n)
        width = config.hidden_size * 3
        self.fusion = nn.MultiheadAttention(width, heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, bottleneck), nn.GELU(), nn.Dropout(0.0), nn.Linear(bottleneck, 2)
        )

    def load(self) -> "Pruner":
        from safetensors.torch import load_file

        state = load_file(WEIGHTS / "model.safetensors")
        renamed = {}
        for key, value in state.items():
            key = key.removeprefix("model.")
            key = key.replace("fusion_layers.0.", "fusion.").replace("fusion_norms.0.", "fusion_norm.")
            key = key.replace("compression_head.feature_extractor.", "head.")
            if key.startswith("compression_head.crf."):
                continue  # transitions only matter for Viterbi decoding; inference uses emissions
            renamed[key] = value.float()
        missing, unexpected = self.load_state_dict(renamed, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"weights do not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
        return self.eval()

    @torch.no_grad()
    def token_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        h = torch.cat([out.hidden_states[i].float() for i in self.layers], dim=-1)
        attn, _ = self.fusion(h, h, h)
        h = self.fusion_norm(attn + h)
        emissions = self.head(h)
        return torch.sigmoid(emissions[..., 1] - emissions[..., 0])[0]


class CodePruner:
    """Line-level pruning of one tool output against a focus question."""

    def __init__(self, max_tokens: int = 4096, device: str | None = None):
        from transformers import AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(WEIGHTS)
        self.model = Pruner().load().to(self.device)
        self.max_tokens = max_tokens
        self.prefix = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(SUFFIX, add_special_tokens=False)

    def line_scores(self, query: str, text: str) -> dict[int, float]:
        head = self.tok.encode(f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: ", add_special_tokens=False)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        room = max(100, self.max_tokens - len(self.prefix) - len(self.suffix) - len(head))
        char_score: dict[int, float] = {}
        for start in range(0, len(ids), room):
            chunk = ids[start : start + room]
            seq = torch.tensor([self.prefix + head + chunk + self.suffix], device=self.device)
            probs_cpu = self.model.token_probs(seq).cpu()
            base = len(self.prefix) + len(head)
            for i, (a, b) in enumerate(offsets[start : start + room]):
                p = float(probs_cpu[base + i])
                for pos in range(a, b):
                    char_score[pos] = p
        scores, pos = {}, 0
        for n, line in enumerate(text.splitlines(), start=1):
            seen = [char_score[c] for c in range(pos, pos + len(line)) if c in char_score]
            if seen:
                scores[n] = sum(seen) / len(seen)
            pos += len(line) + 1
        return scores

    def prune(self, query: str, text: str, threshold: float = 0.5) -> str:
        """Their output format: pruned runs become '(filtered N lines)'. A gap of
        one line is kept, and a blank line is never worth a marker."""
        scores = self.line_scores(query, text)
        lines = text.splitlines()
        keep = sorted({n for n, s in scores.items() if s >= threshold})
        keep = sorted(set(keep) | {k - 1 for i, k in enumerate(keep) if i and k - keep[i - 1] == 2})
        out, gap, gap_chars = [], 0, 0
        for n, line in enumerate(lines, start=1):
            if not line.strip() or n not in keep:
                gap += 1
                gap_chars += len(line)
                continue
            if gap:
                out.append(f"(filtered {gap} lines)" if gap_chars > 18 else "\n".join(lines[n - 1 - gap : n - 1]))
                gap = gap_chars = 0
            out.append(line)
        if gap:
            out.append(f"(filtered {gap} lines)")
        return "\n".join(out)
