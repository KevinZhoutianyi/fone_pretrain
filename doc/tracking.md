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
| (none) | exp 02 llama3 chunk FoNE | code ready, not launched | §1 | design switched from single-`<NUM>` to chunk-based FoNE (see decisions row); CPU tests + forward/backward green; data prep not yet run |

---

## Recently completed jobs

<!-- One row per landed job. Keep the result one-line; the full
write-up lives in paper.md §N. Once the finding is integrated into
paper.md, this row can be kept here as the historical record. -->

| job | exp | serves paper.md § | result (one-line) |
|---|---|---|---|
| (design change, no job) | exp 01 -> 02 | §1 | switched from single-`<NUM>`-per-number (15-digit sidecar + output digit head) to chunk-based FoNE: numbers stay as Llama-3's native <=3-digit chunk tokens; the fixed 6-dim Fourier code of a chunk's face value overwrites the first 6 embedding dims, rest learned. No `<NUM>` token, no sidecar, no digit head, no aux loss. All 3 variants now share ONE tokenization (only the embedding differs), a cleaner control than 01. Old single-`<NUM>` code lives in git history; configs/125m_*.yaml retained. |
| 216-218 | exp 01 smoke (20 steps, 3 variants) | §1 (historical, single-`<NUM>` design) | all COMPLETED; fone num_loss 2.52->0.93, digit_acc 0.10->0.85; S3 sync verified |
| 220 | exp 01 smoke rerun (static-shape check) | §1 | 3/3 PASS, 0 autograd warnings, fone throughput 21.5k->55.8k tok/s |
| data prep mix3b_baseline | exp 01 data | §1 | 3.2B tokens, 2,405,888 docs, 33 shards, DONE |
| data prep mix3b_fone v2 | exp 01 data | §1 | 3.1B tokens over the SAME 2,405,888 docs as baseline (doc-count budget), 31 shards, DONE |
| 232 (smoke4) | exp 01 smoke on final data, clean node | §1 | 3/3 PASS; baseline digit_acc now null (not fake 1.0); post-review code all works together |

---

## Recently failed jobs

<!-- MANDATORY. Document each failure with the diagnostic and the
resolution (or "blocked on X"). Otherwise the same broken approach
gets re-attempted blindly. -->

| job | exp | failure mode | resolution |
|---|---|---|---|
| 233 | exp 01 125M baseline v1 | landed on ip-10-4-120-250 which has a stray process holding 74/80GB on every GPU (also killed smoke3 there); our job OOMed at startup | resubmitted as 236 with --exclude=ip-10-4-120-250; node reported to no avail so far -- exclude it in future submissions until it drains |
| 214 | exp 01 smoke v1 (srun loop) | srun spawned 4 duplicate torchruns (192 cpu / 48 cpu-per-task), rendezvous collided, variants reported bit-identical metrics | --ntasks=1 everywhere; resubmitted as sbatch 216-218 |
| 214 | same | configs/data on login-node /tmp scratchpad invisible to compute nodes | smoke assets moved to /fsx/zhouty/data/fone_pretrain/smoke/ |
| 217/218 v1 | fone smokes | torch.compile: boolean-index embed -> per-step recompiles + IndexPutBackward autograd errors (recovered eager, slow) | dense masked ops, static shapes; rerun as 219 |
| data prep mix3b_fone v1 | exp 01 data | budgeted by --tokens 3.2e9 independently per mode; FoNE compresses numbers to fewer tokens/doc, so at equal token count it read further into the (seed-42-deterministic) document stream than baseline did — the two "same underlying text" datasets were actually disjoint document sets, caught by multi-agent review before any GPU training touched them | prepare_data.py rebudgeted by --docs (document count) instead of --tokens; both variants now read the identical first N documents of the stream, so token counts differ (a real, disclosed consequence of the method) but documents match exactly. Rerun as mix3b_fone v2 with --docs 2405888 matching baseline's doc count. Stale v1 shards moved to datasets/_stale/, not deleted. |
| (code review, no job) | exp 01 model.py | digit_acc hardcoded to 1.0 for the baseline variant (no digit head), which is not a real accuracy value — any digit_acc plot/table across the 3 variants would show baseline pinned at a false 1.0, misreading as baseline having flawless numeracy | changed to NaN (encodes "not applicable"), serialized as JSON null via a helper so metrics.jsonl stays valid; downstream aggregation/plotting must skip nulls for baseline rather than average or compare them |
| (code review, no job) | exp 01 train.py | global weight_decay=0.1 applied to fone_learned's freq_mult, which is init to 1.0 specifically so the variant starts exactly equivalent to fixed FoNE (§ thesis); decay pulls it toward 0, where cos/sin(0) is a constant and the embedding stops carrying any digit value at all, silently degenerating the learned-frequency variant over training | excluded freq_mult from weight decay via a separate AdamW param group (weight_decay=0.0); confirmed by name at model construction that the group is non-empty for fone_learned |
| (code review, no job) | exp 01 data.py | val split assumed val_tokens fits inside the single final shard (`val_start = len(last_shard) - val_tokens`); prepare_data.py flushes shards at a fixed SHARD_TOKENS boundary so the real final shard is a small leftover (504K tokens against mix3b_baseline's whole run) -- against the actual dataset this computed a negative val_start and crashed the very first val call with a broadcast error. Would have crashed every real 125M/350M run at step `eval_every`, after already spending GPU-hours. | rewrote the split to work over the global token stream: val is the tail `val_tokens` of all shards concatenated, spanning as many trailing shards as needed; shards too small to hold one seq_len+1 window are dropped from both splits. Reproduced the crash directly against mix3b_baseline, confirmed the fix with 200+ sample_batch calls on the real data, added tests/test_data.py (3 tests) as a permanent regression guard. |

---

## Next steps

<!-- Numbered list. Each step names which paper.md § it serves —
otherwise the queue drifts away from the paper's argument. -->

1. **Unit-test chunk-based FoNE core (CPU)** — serves paper.md §1. DONE.
   Correctness gate before any GPU spend; pytest on the number-chunk token map,
   6-dim code injectivity over 0-999, learned-init parity, uint32 round-trip (11/11);
   plus a CPU forward/backward for all three variants and a freq_mult gradient check.
2. **Prepare data over one shared tokenization** — serves paper.md §1.
   FineWeb-Edu 70% + FineMath 30%, Llama-3 tokenizer (numbers auto-chunk to <=3 digits),
   budgeted by document count. One dataset feeds all three variants: tokenization does
   not depend on the variant, only the embedding does. Spot-check by decoding windows and
   confirming the number-chunk map fires on exactly the digit-chunk positions. Pick --docs
   by watching total_tokens approach ~3.1B. CPU-only on login node, hours.
3. **125M pipeline-validation runs, all three variants** — serves paper.md §1.
   Smoke (20 steps, --smoke) first, then one node 8×H100 each, ~3B tokens; check LM loss
   converges and fone == fone_learned at step 0.
4. **350M formal three-variant comparison, 10B+ tokens** — serves paper.md §2.
   The equal-token-budget comparison that isolates the embedding as the cause.
5. **Number eval suite incl. frontier-model comparison** — serves paper.md §3.
   Arithmetic exact match by digit length, number comparison, numeric precision;
   frontier API models need keys from the user (blocked on that).
