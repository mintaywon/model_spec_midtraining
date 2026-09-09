# Handoff — get grad-dot to run on cheese 8B

> ## ✅ CLOSED 2026-09-09 15:03. Root cause: `DECISIONS.md` §H8. Results: `STATUS.md` §7.7.
> The leading hypothesis in §3 was right about *what* filled the disk (the document
> `build` stored unprojected 336 MB gradients, 2.15 TB over the corpus) and wrong about
> the fix: `projection_dim` was not needed, because **`score` never reads a document
> index**. Dropping the build entirely took the run from 44 min-to-failure to
> **10.4 min, rc=0**. Everything below is the brief as written, kept for the record.

**Scope of this brief**: one job. `graddot_cheese` has failed twice. Diagnose it,
fix it, and produce a grad-dot score array over the 6,400 midtraining documents
that `compare_three` can read. Nothing else in the project needs your attention.

Read [`CLAUDE.md`](CLAUDE.md) for the durable brief and
[`STATUS.md`](STATUS.md) §7 for live state. Incidents are in
[`DECISIONS.md`](DECISIONS.md) — §H6 and §H7 are the ones that touch this task.

---

## 1. What grad-dot is for

Third estimator in a three-way method comparison on the cheese setting:

| method | what it uses | status |
|---|---|---|
| multi-stage SOURCE | full trajectory, per-segment Hessians | ✅ done |
| EK-FAC | single checkpoint + curvature, union index | ✅ done |
| **grad-dot** (TracIn-final) | single checkpoint, no curvature | 🔴 **your job** |

It is the cheapest and least sophisticated of the three, and it exists to answer
"does the curvature/trajectory machinery buy anything?" All three must score **the
same 6,400 documents against the same query set at the same checkpoint**, or the
comparison is not about the estimator.

`tda/modal/bergson_app.py::compare_three` consumes it and **raises** if the run is
missing or has no `scores/` (it used to skip silently — DECISIONS §H6). You are done
when `--action three_way` returns three methods.

## 2. The failure, precisely

`tda/modal/bergson_app.py::graddot_cheese` (~line 2541). Two runs, both `rc=1`:

| run | max_batch_size | wall | outcome |
|---|---|---|---|
| `graddot_cheese8b_A_america-attr-target_20260909-0224` | 8 | 44.0 min | FAILED |
| `graddot_cheese8b_A_america-attr-target_20260909-0312` | 4 | 43.9 min | FAILED |

Both `report.json` are on the `msm-tda-results` volume under
`bergson/cheese/graddot/<run>/`. Identical traceback:

```
File "bergson/cli/commands.py", line 87, in execute
    build(self.index_cfg, self.preprocess_cfg)
File "bergson/build.py", line 134, in build
    launch_distributed_run(
File "bergson/distributed.py", line 204, in launch_distributed_run
    raise RuntimeError(
RuntimeError: build child exited with code -11
```

`-11` is SIGSEGV in a worker child. It happens in the **second** `build` step
(6,400 documents, 1,488 batches), roughly **62%** in. The **first** build step —
the 200-item query index — completes fine.

**The two runs failed at the same fraction after the same wall time despite
different batch sizes. Treat this as deterministic, not a resource race.** That is
why the `max_batch_size` 8→4 retry changed nothing, and it is the single most
useful clue: whatever kills it is a function of how much has been written, or of a
specific document, not of per-step memory.

## 3. Leading hypothesis — unprojected gradient storage

**Check this first.**

In `graddot_cheese` the query index sets `projection_dim: 0` explicitly, but the
document index `idx` **never sets `projection_dim` at all**. In bergson 0.26.2,
`IndexConfig.projection_dim` defaults to `0`, documented as *"or 0 to disable it"*
— so the document build stores **full LoRA gradients**.

Order of magnitude for Llama-3.1-8B, LoRA r=64 on all 7 projections:

- per layer ≈ 5.24M params (q/o 524k each, k/v 328k each, gate/up/down 1.18M each)
- × 32 layers ≈ **168M params per document**
- fp16 ≈ **335 MB per document** × 6,400 ≈ **2.1 TB**

62% of that is ~1.3 TB, which is where both runs died. A memmap write that runs
past available space surfaces as SIGBUS/SIGSEGV rather than a clean `ENOSPC`, which
fits the symptom. `ephemeral_disk` is set to 3 TiB, so **verify what the scratch
path actually resolves to and how much space it really has** — the 3 TiB may not be
where `SCRATCH_DIR` points.

**Why SOURCE and EK-FAC survive the same 6,400 documents**: they do not use raw
`build`+`score`. They go through bergson's Hessian pipeline (`hessian_cfg`,
`hessian_pipeline_cfg`), which accumulates factors rather than retaining a full
gradient per document.

**Likely fix**: set `projection_dim` on the document index. `CLAUDE.md` §5.1 already
sanctions this — *"random projection (JL, fixed seed) of LoRA grads to 32k dims if
full grads don't fit"* — so `projection_dim: 32768` is in-spec and needs no new
decision. Confirm the projection is applied to the **query index too**, or the dot
product is between vectors in different spaces.

## 4. Other hypotheses, cheapest first

1. **Disk**: instrument free space on `SCRATCH_DIR` during the build. If it is
   falling toward zero at 62%, hypothesis 3 is confirmed and you are done.
2. **A specific document**: the manifest records `n_truncated: 1`,
   `length_max: 4096`, `length_min: 475`. If a single row is pathological, a run
   over `ds.select(range(3500, 4500))` should reproduce in minutes rather than 44.
   This is the cheap bisect and worth doing regardless.
3. **The distributed path**: `nproc=1` removes multiprocessing entirely. Slower
   (~90 min) but it turns a SIGSEGV in a child into a real traceback in the parent.
   **Do this if the first two are inconclusive** — a readable stack is worth 45 min.
4. **Two builds in one invocation**: the config runs query-build, doc-build, score
   as three `steps` in one `python -m bergson` process. Splitting them into separate
   invocations rules out state carried between steps.

## 5. Rules of engagement

- **💰 Budget**: `CLAUDE.md` §2b(0). Total ceiling $500 lifetime, ~$322 committed as
  of 2026-09-09 14:00. Under $100 per decision proceeds; over $100 gets priced and
  presented. A failed 44-min 2×H100 run costs ~$7 — **bisect on a subset before
  paying for another full run.**
- **🔴 Do not touch other runs.** Eight removal arms (seeds 43/44) and one EK-FAC
  proponent arm are in flight on Modal and land ~16:15. A separate `msm-tda` app is
  running that this project did not start (DECISIONS §H3) — leave it alone.
- **Do not delete anything under `bergson/cheese/` on the `msm-tda-results` volume.**
  Two completed SOURCE runs have already been lost to cleanup (DECISIONS §H4).
- **Naming**: `CLAUDE.md` §2b(4b). Build run names with
  `tda/influence/source/naming.py::run_name`, never by hand.
- **Sign convention** (DECISIONS §H7, and it has bitten twice): bergson stores are
  **loss-signed — proponents are NEGATIVE**. `_oriented` in
  `tda/influence/source/scores.py` applies this. Any code that sorts a score array
  must state which convention it assumes at the sort site. If you add analysis, say
  which end you are taking.
- Use `modal run --detach` and spawn-and-poll, never a blocking `.remote()`
  (STATUS.md §2; a client-side gRPC deadline has killed a run mid-flight).

## 6. Definition of done

1. `graddot_cheese` completes with `status: "OK"` and a `scores/` directory.
2. `modal run tda/modal/bergson_app.py --action three_way` returns **three** methods
   with pairwise Spearman and top-k Jaccard.
3. `DECISIONS.md` gets an incident entry recording the root cause — this is the
   third infrastructure failure whose diagnosis would otherwise be lost.
4. Report the three-way Spearman numbers back.

## 7. Useful commands

```bash
.venv/bin/modal run --detach tda/modal/bergson_app.py --action graddot
.venv/bin/modal run tda/modal/bergson_app.py --action three_way
.venv/bin/modal volume ls msm-tda-results bergson/cheese/graddot
.venv/bin/modal app logs <app-id>          # cap it; it tails forever
```
