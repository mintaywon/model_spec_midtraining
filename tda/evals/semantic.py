"""Semantic scorers for midtraining documents: representations and embeddings.

Three scorers, all cheap (one forward pass each, no gradients, no retraining),
all built so they can carry SIGN — every one has a contrastive form that
compares a document against the query *with the value-aligned answer* versus
*with the misaligned answer*, so "pushes toward" and "pushes away" are both
representable. Plain relevance (similarity to the bare question) is kept beside
each as the sign-free twin, because for a corpus where every document is on
topic, relevance alone cannot separate proponents from opponents.

  S1  embedding similarity     off-the-shelf sentence embedding (bge-m3, 8k
                               context so a 1,500-token document is not
                               truncated to its first paragraph)
  S2  activation similarity    the same geometry in the trained model's own
                               residual stream: mean-pooled document states
                               against query states at one layer
  S3  value-direction          diff-in-means direction between the aligned and
      projection               misaligned answer states over the 200 attr
                               queries; each document scored by the projection
                               of its mean state onto it. The representation-
                               engineering / "steering vector" construction.

CHECKPOINT. Activations come from the FINAL MSM+AFT checkpoint of the chained
run (`runs/msm_A__chain_ck198/checkpoints/checkpoint-504`) — the same weights
EK-FAC and grad-dot were scored at — so a rank comparison against those two is a
comparison of scorers, not of checkpoints. That model has trained on every
document; for a *similarity* score that is the point (it is the model's own
semantic space), unlike a memorisation-style score where it would be a
contamination. Recorded in `meta.json` so it cannot be confused later.

LAYER. Chosen a priori, not tuned: layer 20 of 32 (~60% depth), where
mid-to-late residual streams carry the most linearly-decodable semantic content
in this model family. States at several other layers are saved from the same
pass at no extra GPU cost so sensitivity can be *reported*; the primary is
fixed before any score is looked at.

ITEMS. The 200 `attr` queries only, via `icl2.attr_items`, which refuses to run
on anything but the attr half — the removal test is graded on the other half.

ROW ORDER. Everything document-indexed is written in corpus order, row i =
dataset row i, with the count asserted. A permuted store leaves every aggregate
unchanged and destroys only which document is which.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tda.evals.icl import wrap
from tda.evals.icl2 import ICL2Config, attr_items

CORPUS = "chloeli/msm-llama-pro-america"
BASE_MODEL = "meta-llama/Llama-3.1-8B"
FINAL_ADAPTER = "/results/bergson/cheese/runs/msm_A__chain_ck198/checkpoints/checkpoint-504"
LAYERS = (8, 12, 16, 20, 24, 28, 32)     # hidden_states indices; 0 = embeddings
PRIMARY_LAYER = 20
DOC_MAX_TOKENS = 4096                     # the training max length


def _option_texts(question: str) -> dict[str, str]:
    """{'A': text, 'B': text} parsed out of the assembled MCQ question."""
    import re
    return dict(re.findall(r"([AB])\)\s*([^\n]+)", question))


# --------------------------------------------------------------------------
# S2 / S3: activations of the trained model
# --------------------------------------------------------------------------

@dataclass
class ActConfig:
    base_model: str = BASE_MODEL
    adapter: str = FINAL_ADAPTER
    corpus: str = CORPUS
    layers: tuple = LAYERS
    doc_max_tokens: int = DOC_MAX_TOKENS
    batch_size: int = 4
    doc_limit: int = 0


def extract_activations(cfg: ActConfig, out_dir: str | Path) -> dict:
    import numpy as np
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = list(cfg.layers)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, cfg.adapter).eval()
    hidden = model.config.hidden_size

    @torch.no_grad()
    def states(ids_list: list[list[int]]):
        """Per-sequence (mean-pooled, last-token) states at each layer."""
        enc = tok.pad({"input_ids": ids_list}, return_tensors="pt").to(0)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states                    # tuple[L+1] of [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        lens = enc["attention_mask"].sum(1)      # [B]
        means, lasts = [], []
        for L in layers:
            h = hs[L]
            means.append(((h * mask).sum(1) / lens.unsqueeze(-1)).float().cpu())
            lasts.append(h[torch.arange(h.shape[0]), lens - 1].float().cpu())
        return torch.stack(means), torch.stack(lasts)   # [nL, B, H]

    # ---- documents -------------------------------------------------------
    ds = load_dataset(cfg.corpus, split="train")
    n = len(ds) if not cfg.doc_limit else cfg.doc_limit
    doc_ids = [tok(ds[i]["text"], add_special_tokens=True)["input_ids"][: cfg.doc_max_tokens]
               for i in range(n)]
    lengths = np.array([len(x) for x in doc_ids])
    docs_mean = np.zeros((len(layers), n, hidden), dtype=np.float16)
    docs_last = np.zeros((len(layers), n, hidden), dtype=np.float16)
    # Length-sorted batches: padding waste is what makes this slow otherwise.
    order = np.argsort(-lengths)
    t0 = time.time()
    for s in range(0, n, cfg.batch_size):
        idx = order[s: s + cfg.batch_size]
        m, l = states([doc_ids[i] for i in idx])
        docs_mean[:, idx] = m.numpy().astype(np.float16)
        docs_last[:, idx] = l.numpy().astype(np.float16)
        if (s // cfg.batch_size) % 100 == 0:
            el = time.time() - t0
            done = s + len(idx)
            print(f"  docs {done}/{n}  {lengths[order[:done]].sum()/max(el,1e-6):,.0f} tok/s  "
                  f"eta {(n-done)/max(done,1)*el/60:.1f} min", flush=True)
    np.save(out_dir / "docs_mean.npy", docs_mean)
    np.save(out_dir / "docs_last.npy", docs_last)
    np.save(out_dir / "doc_lengths.npy", lengths)

    # ---- queries: prompt, then prompt + aligned / misaligned letter --------
    items, item_rows = attr_items(ICL2Config())
    q_last, q_mean, q_al, q_mis = [], [], [], []
    for r in items:
        p = wrap(r["question"])
        ans = str(r["answer"]).strip().upper()
        alt = "A" if ans == "B" else "B"
        p_ids = tok(p, add_special_tokens=True)["input_ids"]
        al_ids = tok(p + " " + ans, add_special_tokens=True)["input_ids"]
        mi_ids = tok(p + " " + alt, add_special_tokens=True)["input_ids"]
        # The letter must be exactly one appended token, or "the letter
        # position" is not what the state is being read at.
        assert len(al_ids) == len(p_ids) + 1 and len(mi_ids) == len(p_ids) + 1, \
            (len(p_ids), len(al_ids), len(mi_ids))
        m, l = states([p_ids, al_ids, mi_ids])
        q_mean.append(m[:, 0]); q_last.append(l[:, 0])
        q_al.append(l[:, 1]); q_mis.append(l[:, 2])
    stack = lambda xs: np.stack([x.numpy() for x in xs], axis=1).astype(np.float16)

    # OUTPUT-ALIGNMENT DIAGNOSTIC for the S3 direction (Billa, arXiv:2604.15557):
    # a diff-in-means direction only steers/reads the concept if, pushed
    # through the final norm and unembedding, it lands on the target tokens.
    # Per layer: logit-lens the direction and record where the aligned-vs-
    # misaligned LETTER logits fall. Computed here because it needs lm_head.
    with torch.no_grad():
        al_tok = [tok(" A", add_special_tokens=False)["input_ids"][0],
                  tok(" B", add_special_tokens=False)["input_ids"][0]]
        Q_al = stack(q_al).astype(np.float32); Q_mi = stack(q_mis).astype(np.float32)
        ans = np.array([str(r["answer"]).strip().upper() for r in items])
        lens_ = []
        norm = model.base_model.model.model.norm
        head = model.base_model.model.lm_head
        for li, L in enumerate(layers):
            v = torch.tensor((Q_al[li] - Q_mi[li]).mean(0), device=0, dtype=torch.bfloat16)
            logits = head(norm(v)).float()
            # signed: + if the direction favours " A" over " B". Whether that is
            # "aligned" depends on the item, so also report per-item agreement.
            a_minus_b = float(logits[al_tok[0]] - logits[al_tok[1]])
            rank_a = int((logits > logits[al_tok[0]]).sum()); rank_b = int((logits > logits[al_tok[1]]).sum())
            # per-item: does the per-query difference vector favour that item's aligned letter?
            per = []
            for qi in range(len(items)):
                vq = torch.tensor(Q_al[li][qi] - Q_mi[li][qi], device=0, dtype=torch.bfloat16)
                lg = head(norm(vq)).float()
                want = al_tok[0] if ans[qi] == "A" else al_tok[1]
                other = al_tok[1] if ans[qi] == "A" else al_tok[0]
                per.append(float(lg[want] - lg[other] > 0))
            lens_.append({"layer": L, "mean_dir_logit_A_minus_B": a_minus_b,
                          "rank_of_A": rank_a, "rank_of_B": rank_b,
                          "per_item_direction_favours_aligned_letter": float(np.mean(per))})
    np.save(out_dir / "q_mean.npy", stack(q_mean))         # [nL, 200, H]
    np.save(out_dir / "q_last.npy", stack(q_last))         # state at the decision position
    np.save(out_dir / "q_aligned.npy", stack(q_al))        # state AT the aligned letter
    np.save(out_dir / "q_misaligned.npy", stack(q_mis))    # state AT the misaligned letter

    meta = {"config": asdict(cfg), "layers": layers, "primary_layer": PRIMARY_LAYER,
            "hidden": hidden, "n_docs": n, "n_items": len(items),
            "item_rows": item_rows, "answers": [str(r["answer"]).strip().upper() for r in items],
            "doc_tokens_total": int(lengths.sum()), "elapsed_s": round(time.time() - t0, 1),
            "logit_lens_output_alignment": lens_,
            "pooling": {"docs_mean": "attention-masked mean over tokens",
                        "docs_last": "final token", "q_last": "last prompt token",
                        "q_aligned/q_misaligned": "the appended letter token"}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {n} docs, {lengths.sum():,} tokens in {meta['elapsed_s']/60:.1f} min", flush=True)
    return meta


# --------------------------------------------------------------------------
# S1: off-the-shelf sentence embeddings
# --------------------------------------------------------------------------

@dataclass
class EmbConfig:
    model: str = "BAAI/bge-m3"
    corpus: str = CORPUS
    max_length: int = 8192
    batch_size: int = 8
    doc_limit: int = 0


def extract_embeddings(cfg: EmbConfig, out_dir: str | Path) -> dict:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModel.from_pretrained(cfg.model, torch_dtype=torch.float16).to(0).eval()

    @torch.no_grad()
    def embed(texts: list[str]) -> np.ndarray:
        out = []
        order = np.argsort([-len(t) for t in texts])
        for s in range(0, len(texts), cfg.batch_size):
            idx = order[s: s + cfg.batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=cfg.max_length, return_tensors="pt").to(0)
            h = model(**enc).last_hidden_state[:, 0]          # CLS pooling (bge-m3 dense)
            h = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
            out.append((idx, h))
        res = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
        for idx, h in out:
            res[idx] = h
        return res

    ds = load_dataset(cfg.corpus, split="train")
    n = len(ds) if not cfg.doc_limit else cfg.doc_limit
    t0 = time.time()
    docs = embed([ds[i]["text"] for i in range(n)])
    np.save(out_dir / "docs.npy", docs.astype(np.float16))

    items, item_rows = attr_items(ICL2Config())
    plain, al, mis = [], [], []
    for r in items:
        q = r["question"]; opts = _option_texts(q)
        ans = str(r["answer"]).strip().upper(); alt = "A" if ans == "B" else "B"
        plain.append(q)
        al.append(f"{q}\nAnswer: {ans}) {opts.get(ans, '')}")
        mis.append(f"{q}\nAnswer: {alt}) {opts.get(alt, '')}")
    np.save(out_dir / "q_plain.npy", embed(plain).astype(np.float16))
    np.save(out_dir / "q_aligned.npy", embed(al).astype(np.float16))
    np.save(out_dir / "q_misaligned.npy", embed(mis).astype(np.float16))

    meta = {"config": asdict(cfg), "n_docs": n, "n_items": len(items),
            "item_rows": item_rows, "dim": int(docs.shape[1]),
            "pooling": "CLS, L2-normalised (bge-m3 dense retrieval convention)",
            "elapsed_s": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {n} docs embedded in {meta['elapsed_s']/60:.1f} min", flush=True)
    return meta
