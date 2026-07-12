# Tracking — experiment status

<!-- HOW TO USE THIS TEMPLATE
This file is the **project-state view**: what's running, what just
landed, what failed, what's next. The argument and the evidence live
in **paper.md**; each row here cross-references the paper.md section
that the experiment serves. See CLAUDE.md §3 for the full two-file
rule.

Update in place — do not append dated entries. When a job finishes:
(1) move its row from Active runs → Recently completed, (2) integrate
the finding into the relevant paper.md § evidence section.
-->

## Active runs

<!-- Jobs currently on the cluster. Remove when the job lands and its
result is integrated into paper.md. -->

| job | exp | status | serves paper.md § | note |
|---|---|---|---|---|
| data prep (CPU, bg) | chunk_fone | running | §1 | tokenizing FineWeb-Edu 70% + FineMath 30% with Llama-3 into mix3b_llama3 (--docs 3M, target >=3.2B tokens); one shared dataset for all variants; number-chunk map built (1110/128256 tokens). ~3h. Blocks all training. |
| (none) | chunk_fone training | code ready, waiting on data | §1 | chunk-based FoNE via effective tied weight (code on both read and write sides); fone=3 fixed periods (6d), fone_learned=3+20 learnable periods (46d); learnable code scale (init 0.02) aligns init loss across variants (4.375/4.371/4.393); 20/20 CPU tests + tie-sharing verified |

---

## Recently completed jobs

<!-- One row per landed job. Keep the result one-line; the full
write-up lives in paper.md §N. Once the finding is integrated into
paper.md, this row can be kept here as the historical record. -->

| job | exp | serves paper.md § | result (one-line) |
|---|---|---|---|
| (no job) | chunk-FoNE core | §1 | 20/20 CPU tests + d_model=768 forward/backward green; tie-sharing (read==write share the code), zero-grad on frozen code dims, learnable periods + scale train, init loss aligned across variants (4.375/4.371/4.393) all verified |

<!-- The single-`<NUM>`-per-number design (exp 01: whole number -> one <NUM> token +
15-digit sidecar + output digit head) was replaced by chunk-based FoNE and its jobs
cancelled. That code and its run history live in git before commit c5dfec9. Lessons
from it that still guard the current code are recorded in Recently failed jobs below. -->


---

## Recently failed jobs

<!-- MANDATORY. Document each failure with the diagnostic and the
resolution (or "blocked on X"). Otherwise the same broken approach
gets re-attempted blindly. -->

<!-- These lessons predate the chunk-based redesign but still guard the CURRENT code
(data.py val split, prepare_data.py doc-count budgeting, train.sbatch launch, the
static-shape effective_weight). Kept so the same mistakes are not repeated. -->

| job | exp | failure mode | resolution |
|---|---|---|---|
| node OOM | any GPU run | ip-10-4-120-250 had a stray process holding 74/80GB on every GPU; jobs landing there OOM at startup | submit with `--exclude=ip-10-4-120-250` until the node drains |
| srun loop | training launch | srun without `--ntasks=1` spawned 4 duplicate torchruns (192 cpu / 48 cpu-per-task), rendezvous collided, variants reported bit-identical metrics | `--ntasks=1` in train.sbatch (load-bearing, commented there) |
| login /tmp | any run | configs/data on login-node /tmp are invisible to compute nodes | all run assets live under /fsx/zhouty/data/fone_pretrain/ |
| torch.compile recompiles | embed/loss | boolean-index embedding -> dynamic shapes -> per-step recompiles + IndexPutBackward autograd errors | static-shape ops only; the current effective_weight uses index_copy over a fixed id set, no boolean indexing |
| disjoint corpora | data prep | budgeting by `--tokens` independently per variant let variants stop at different points in the seed-42 stream -> "same text" datasets were actually disjoint document sets | budget by `--docs` (document count); the stream is deterministic in seed, so equal --docs -> identical documents. With one shared tokenization now, all variants read the same shards, so this cannot recur, but the doc-count budgeting is retained for any future per-variant data. |
| val-split crash | data.py | val split assumed val_tokens fits in the single final shard; prepare_data.py flushes at a fixed SHARD_TOKENS boundary so the final shard is a small leftover -> negative val_start -> crash on the first val call, after spending GPU-hours | val split works over the global token stream (tail val_tokens spanning as many trailing shards as needed; too-small shards dropped from both splits); guarded permanently by tests/test_data.py |

---

## Next steps

<!-- Numbered list. Each step names which paper.md § it serves —
otherwise the queue drifts away from the paper's argument. -->

1. **Unit-test chunk-based FoNE core (CPU)** — serves paper.md §1. DONE.
   Correctness gate before any GPU spend; pytest on the number-chunk token map, code
   injectivity over 0-999, fixed vs learned period setup, uint32 round-trip, and the
   tie-sharing property (input embedding row == output-projection row == code), 17/17;
   plus CPU forward/backward for all three variants and a log_periods gradient check.
2. **Prepare data over one shared tokenization** — serves paper.md §1.
   FineWeb-Edu 70% + FineMath 30%, Llama-3 tokenizer (numbers auto-chunk to <=3 digits),
   budgeted by document count. One dataset feeds all three variants: tokenization does
   not depend on the variant, only the embedding does. Spot-check by decoding windows and
   confirming the number-chunk map fires on exactly the digit-chunk positions. Pick --docs
   by watching total_tokens approach ~3.1B. CPU-only on login node, hours.
3. **125M pipeline-validation runs, all three variants** — serves paper.md §1.
   Smoke (20 steps, --smoke) first, then one node 8×H100 each, ~3B tokens; check LM loss
   converges. A learnable code scale (init 0.02) aligns the three variants' init loss to
   within 0.02, so they start together; the check is that all three fall and converge.
4. **350M formal three-variant comparison, 10B+ tokens** — serves paper.md §2.
   The equal-token-budget comparison that isolates the embedding as the cause.
5. **Number eval suite incl. frontier-model comparison** — serves paper.md §3.
   Arithmetic exact match by digit length, number comparison, numeric precision;
   frontier API models need keys from the user (blocked on that).
