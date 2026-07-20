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
| (none) | — | queue empty | — | OLMo stage-2 surgery wave finished; see Recently completed. |

---

## Recently completed jobs

<!-- One row per landed job. Keep the result one-line; the full
write-up lives in paper.md §N. Once the finding is integrated into
paper.md, this row can be kept here as the historical record. -->

| job | exp | serves paper.md § | result (one-line) |
|---|---|---|---|
| (no job) | chunk-FoNE core | §1 | 21/21 CPU tests + d_model=768 forward/backward green; tie-sharing (read==write share the code), zero-grad on frozen code dims, learnable periods + scale train, init loss aligned across variants (4.375/4.371/4.393) all verified |
| data prep mix3b_llama3 | chunk_fone data | §1 | 3,391,139,029 Llama-3 tokens over 3,000,320 docs, 34 shards, uint32; number-chunk map 1110/128256; one shared dataset for all variants; DONE |
| 309-318 (smoke) | chunk_fone 9-variant smoke | §1 | 9/9 PASS (20 steps): loss 11.9->7.78, ~36-58k tok/s; all variants converge to near-identical loss this early (numbers sparse); baseline retried once (309 hit a dirty node, 318 clean) |
| 319-327 | chunk_fone 9-variant sweep | §1 | 9/9 trained to 6000 steps, all converged, final val_loss 2.86 (seed2 2.90) -- identical across variants, as expected (overall LM loss is text-dominated, not a discriminator). §1 pipeline-validity GOAL MET. |
| 356-364 (eval) | chunk_fone number eval | §1 | add/sub ~0 at all digit lengths (125M+3B is too small for arithmetic generation; not a discriminator). compare (2-10 digits) has signal but single-seed variance dominates: fone and fone_seed2 (SAME config, diff seed) got avg 0.16 vs 0.36. Weak trend that fone_12d/fone_learned_fixedscale hold up better than baseline at 6-10 digits, but not trustworthy at 1 seed -> multi-seed follow-up. |
| 366-390 (3-seed) | chunk_fone multi-seed compare | §2 | SUPERSEDED by 6-seed below. 3 seeds suggested a strong fone_learned win (6d 0.34 vs baseline 0.07, learned>fixed). Adding 3 seeds reversed it -> the 3-seed result was a favorable draw. Kept as a cautionary record: 3 seeds was too few. |
| 401-423 (6-seed) | chunk_fone multi-seed compare | §2 | 4 variants x 6 seeds, compare exact-match mean+/-std. HONEST RESULT: all FoNE variants sit slightly above baseline on avg (0.24-0.26 vs 0.22); at 6+ digits baseline collapses to ~0.08 and FoNE stays 0.07-0.19 higher, but every between-variant gap is within one std (bands 0.04-0.21). Variants statistically TIED at 125M; learned vs fixed indistinguishable. Decision: adopt fone_learned as the method for generality (subsumes fixed), not because it wins here. Ranking needs larger scale -> 350M. add/sub still ~0. |
| olmo stage-2 wave (train 1621+, eval 1843) | olmo_stage2_surgery | §3 | FoNE embedding surgery on pretrained OLMo-2-1B, stage-2 continued pretrain on 40.6B/50B Dolmino tokens (81% of official; near-official mix dclm 46%/math 20.9%), then GSM8K (n=200, greedy exact-match). 2 arms x 2 seeds all trained to ckpt_final. RESULT: fone mean **0.273** (s1337 0.280, s2024 0.265) vs baseline mean **0.255** (both seeds 0.255); fone +1.75pt. Both trail official OLMo-2-1B stage-2 (**0.33**, its full 50B) — expected at 81% budget. Gap is within noise at 2 seeds (n=200 -> ~+/-3pt binomial SE per point; baseline std 0). A signal in fone's favor, not yet significant. |

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
| 1579-1589 | olmo_stage2 wave v1 | all 6 runs scancel'd (SIGTERM) at ~1h55m when another user took the 24 nodes on the full cluster (dev PreemptMode=OFF, so a manual/priority scancel). ckpt_every was 4000 -> no checkpoint saved -> 2h lost | ckpt_every 4000->500 (~25min); sbatch --requeue so SLURM re-runs on preemption; auto-resume from ckpt_latest.pt (model+opt+step). Relaunched as 1607-1617; a bump now costs <=25min and self-heals. |
| olmo_baseline_s777 | olmo_stage2 wave v1 | leftover from the v1 wave: reached step 2340/19359 (Jul 16) with the old config ckpt_every=4000 -> no ckpt ever saved (same failure as 1579-1589). Seed 777 was dropped when the wave shrank to 2 seeds (1337,2024) | superseded by the 2-seed v2 wave (see Recently completed). Ckpt dir holds only config.yaml + metrics.jsonl (74K), no weights; kept as historical record, not part of the current results. |
| dolmino prep (download) | olmo_stage2 data | crashed at 40.6B/50B on a transient hf_hub_download network error (LocalEntryNotFoundError, not caught) | kept all 406 full 100M-token shards (81% of target, covers the GSM8K window; mix near-official dclm 46%/math 20.9%); wrote manifest by hand; max_steps set to 19359 to match 40.6B. Did NOT re-run the last 9.4B (would risk another net error for little gain). |

| job | exp | failure mode | resolution |
|---|---|---|---|
| node OOM | any GPU run | ip-10-4-120-250 and ip-10-4-102-38 each had a stray process holding ~70/80GB on every GPU; jobs landing there OOM at startup (309 baseline smoke hit ip-10-4-102-38, retried clean as 318) | submit with `--exclude=ip-10-4-120-250,ip-10-4-102-38` (baked into scripts/launch_sweep.sh) until they drain |
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
6. **Place the OLMo stage-2 GSM8K result in paper.md** — serves paper.md §3 (or a new §).
   The 2-arm x 2-seed wave landed (fone 0.273 vs baseline 0.255, n=200; official 0.33 at
   full 50B). This is a real-pretrained-model, larger-scale continued-pretrain result that
   does not map onto the current §3 (frontier comparison) or the mechanism scaffold. Decide
   with the user where it belongs before writing paper evidence. Open question on strength:
   2 seeds and n=200 make the +1.75pt gap suggestive, not significant — likely need more
   seeds and/or full 200->1319 GSM8K to firm it up.
