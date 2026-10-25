# Reproducibility checklist

Branch `paper/manuscript-draft` (local), based on `docs/correct-failure-mechanism-and-goal-distances` (`83c95e8`, pushed; draft PR #2), which is based on `main` `d257eca`.

## Environment used for every result in the draft

Windows 11, Python 3.12.7, torch 2.10.0+cpu, gymnasium 1.2.3, numpy 2.4.1, CPU only. The same versions are recorded in `results/benchmark/metadata.json`. CI runs the test suite on Linux with Python 3.10 and 3.12.

## Status

| Item | Status | Notes |
|---|---|---|
| Code public (MIT) | ✅ | https://github.com/pouyapd/SafeNav-RL |
| Trained agents committed | ✅ | `results/checkpoints/{filtered,unfiltered}/seed_{0..4}.pt` (hashes below) |
| Per-episode results committed | ✅ | `results/benchmark/episodes.csv`, `results/stochastic_eval/episodes.csv` |
| Benchmark regenerates from the checkpoints | ✅ | per-episode files byte-identical (audit §1); `summary.csv` differs by ≤ 1.8e-15 (summation order) |
| Tests | ✅ | 90 passed (`main`/docs branch); 93 passed on `exp/e1-correction-penalty` |
| Training reproducible | ⚠️ partly | deterministic and exactly resumable at small scale (tests); the 10 × 1M-step runs were not repeated |
| Analysis scripts read stored outputs only | ✅ | `research/scripts/*.py` |
| Fixed seeds for every evaluation | ✅ | test 2,000,000 + 100,000·stage + i; validation 1,000,000 + …; fresh E1 set 5,000,000 + i |
| Pre-registered analysis for the new experiments | ⚠️ partly | E1 decision rules fixed in `research/experiments/E1_correction_penalty.md` before running; E2/E3 still to be written down |
| Held-out test set never used for design choices | ❌ | the 300 test scenarios were inspected; E3 adds fresh scenarios |
| Archived release with a DOI | ❌ | create a Zenodo release when the experiments are complete |
| Compute reported | ✅ | `minimum_experiments.md` (measured training and evaluation speed) |
| Statistical methods described with their limits | ✅ | manuscript §5; approximate intervals with 5 seeds stated |
| Limitations stated | ✅ | manuscript §8, README Limitations |

## Commands to regenerate every number and figure in the draft

```bash
# 0. environment
pip install -e ".[dev]"
python -m pytest -q                                     # 90 tests, about 1-2 min

# 1. benchmark and sampled-action evaluation from the committed checkpoints (minutes)
python -m safenav benchmark --runs results/checkpoints --out runs/check/benchmark --jobs 8
python scripts/stochastic_eval.py --out runs/check/stochastic_eval

# 2. audit diagnostics: timeout attribution, sensor coverage, variance components, paired contrasts (≈ 2.5 min)
python research/scripts/audit_diagnostics.py

# 3. stall taxonomy, all stages and baselines (≈ 5 min); window-length check (≈ 1 min)
python research/scripts/stall_taxonomy.py
python research/scripts/taxonomy_window_sensitivity.py

# 4. paper figures and statistics (seconds) -> paper_draft/figures/*.pdf, paper_draft/generated/stats.json
python research/scripts/paper_figures_stats.py

# 5. manuscript
cd paper_draft && latexmk -pdf manuscript.tex
```

`stats.json` also records an automated consistency check: the taxonomy re-run reproduces the outcome of all 3,600 filtered mean-action and baseline episodes of the committed benchmark, with 0 mismatches.

## SHA-256 of the evidence files (2026-10-25)

```
4082c62025bc31455fdb3f8575e520af3b4fe493792baf21b92ca327fe9d18cd  results/benchmark/episodes.csv
2f33af406cfe6a9a84c6540fdfcc7c299fffcaf9e79fca98c36eaa3fc1062abd  results/benchmark/scenarios.csv
56c2e269f77f7006dc8041ae35f39481772e2a8aee2277c5a982a4eb84535d02  results/benchmark/summary.csv
f0846c8a6b50c4f7f5eae511be412af46dbcdcf6ce8571999092fef32d9f264f  results/stochastic_eval/episodes.csv
07785b9cd0372c75fc192be5dfb1a510c579e81c95bd8fe67e4b20fdfd5d0790  results/stochastic_eval/summary.json
3efc15336800c1591d5b80d34d54362f138257fc27effcd65456c21d046d1f04  research/audit_outputs/stall_taxonomy_all.csv
44ae6fd21870bfc0f022bccbecd40a58c1a0559abc57fad9a7d6b95129afd5cf  research/audit_outputs/timeout_attribution_expert.csv
5bb0213706d2650435fc1f06d384ef2d8ed9fac478d65e879baf58617b7a55c1  research/audit_outputs/paired_comparisons_expert.csv
af4130f4f99d37875a701f79fb69e9b942a7ced97f45f6d991a4ca7ec7611f6d  results/checkpoints/filtered/seed_0.pt
66f287855a2442286eeb98a2c3494f4778bb467de2e4593323cd67883a9cbb47  results/checkpoints/filtered/seed_1.pt
4eaf3e30049a5e4c09475997c34ffb6bf437f70043d72dd2deb2f514c9e1d77e  results/checkpoints/filtered/seed_2.pt
093a2c13a2ec7485833b845ccf7d15724f6e60b5dfa6c42aaea77d226fd7259c  results/checkpoints/filtered/seed_3.pt
f65089f6d3a0071c9a0fe4222b31ebeaa55906c73432a9ff6f6005d11372af57  results/checkpoints/filtered/seed_4.pt
fbe1bdd783fb9b8ee2f99ae0f2343f992ccdfe5727745d45204149a84359ae56  results/checkpoints/unfiltered/seed_0.pt
e1acbf27b2701e7b233d7824ce954ccc3b0ff3697b2597b78597fd528ee537a4  results/checkpoints/unfiltered/seed_1.pt
9b8612d5c6753416085056c9f68853387065aed074c176c5ca9af0518f3ca8a7  results/checkpoints/unfiltered/seed_2.pt
0c54d3c9e5cf2cbf60fff36eda83cea970b89d8be17d5bb3e298593fd34ea3db  results/checkpoints/unfiltered/seed_3.pt
26282eec7dd99b72c6ed33648884712e4b81f349ae03a7cac231ad714e1ddd6d  results/checkpoints/unfiltered/seed_4.pt
```

Check with `sha256sum -c` after saving this block to a file; `results/` must match `main` byte for byte. `paper_draft/generated/stats.json` is a derived file. Its hash changes whenever `paper_figures_stats.py` is re-run, because the JSON ordering and floats are rewritten. It is therefore not listed; regenerate it instead.
