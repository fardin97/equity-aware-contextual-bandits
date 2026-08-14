# Quality report

Date: 2026-08-14

## Release assessment

This repository is suitable as a transparent portfolio publication of an exploratory simulation workflow. It is not ready to support a causal, production, or real-world policy-superiority claim. The README and notebook state that boundary prominently.

The publication copy contains 17 files totaling about 636 KiB. It includes one clean notebook, an extracted policy module, unit tests, documentation, three compact aggregate result files, and four aggregate figures. No license is included because ownership and prior team contributions are not fully established.

## Source preservation

The working notebook was read and transformed into a separate publication directory. It was not edited in place.

- Source notebook size: 823,534 bytes
- Source notebook last-modified time: 2026-07-08T03:33:53.6215008+01:00
- Source notebook SHA-256: `6D0F8D1B5F5DC13DCEAA5ECBD3B39D590574D150A8D2C3752B860D96EC33306C`

## Publication and privacy checks

Passed:

- all notebook code-cell outputs were removed;
- all notebook execution counts were reset;
- the notebook contains no local drive/user paths or temporary mount paths;
- input and output paths are repository-relative with environment-variable overrides;
- no assistant-platform or generated-author markers were found;
- no common credential patterns or email addresses were found;
- author-name occurrences are limited to intentional attribution metadata;
- no raw survey files, row-level predictions, action tables, event-level bandit logs, caches, or model binaries are present;
- row-level preview calls were replaced with column or action-level summaries to reduce accidental disclosure after execution;
- no unencrypted external URLs remain in the public notebook; and
- four retained figures were visually inspected and contain aggregate plots only.

The notebook still contains code that can generate row-level intermediate files when run. Those files are directed to `artifacts/run/`, ignored by Git, and must be reviewed before sharing.

## Code and notebook verification

Verification used an isolated Python 3.11.9 runtime in the task's temporary work directory. The runtime is not part of this repository.

| Check | Result |
|---|---|
| Python bytecode compilation for package and tests | Passed |
| Ruff 0.16.3 lint (`ruff check src tests`) | Passed, no findings |
| Ruff formatting check (`ruff format --check src tests`) | Passed, 3 files formatted |
| Unit tests (`python -m unittest discover -s tests -v`) | Passed, 8 of 8 |
| Notebook JSON load | Passed |
| Static parse of all notebook code cells | Passed, 16 of 16 |
| Notebook execution-state check | Passed, zero stored outputs and zero populated execution counts |
| Packaging metadata parse | Passed |

The tests cover logistic transforms, zone selection, action ordering, action availability and eligibility, burden construction, home and ineligible reward invariants, fallback behavior, activity-priority selection, input-length validation, ridge-linear updating, and the equity-context burden constraint.

## Result integrity

Passed:

- `run_summary.json` and `supervised_model_results.csv` are byte-identical to the aggregate source outputs;
- all four retained figures are byte-identical to the aggregate source figures; and
- `combined_policy_comparison.csv` contains all 18 source policy rows and preserves exact static expected-reward, bandit expected-reward, and bandit realized-reward values.

The combined policy file was restructured to label reward semantics explicitly. Static policies use `mean_expected_reward`; online policies use `mean_expected_reward_chosen`, with simulated realized reward in a separate column. This avoids the working export's misleading comparison of different quantities under one generic metric.

The public notebook also corrects the expected-reward bar chart to use uncertainty from expected reward rather than uncertainty from realized reward.

## Checks intentionally not run

The notebook was not executed end to end because the repository intentionally excludes the survey microdata. A full run also requires scikit-learn, plotting and spreadsheet dependencies, and substantial model/simulation time. The optional GIS builder was not run because it depends on changing external services, local feed snapshots, and a native geospatial stack.

The included aggregate results are therefore verified copies of the prior run, not independently regenerated results.

## Known methodological risks

The documentation records these limitations rather than concealing them:

- model alternatives were ranked on the held-out set later used for evaluation;
- probability calibration is not household-grouped;
- the survey-derived opportunity proxy is not facility-supply data and is constructed before the split;
- the broad revealed-behavior target is not action-specific recommendation acceptance;
- some whole-day features may be unavailable at decision time;
- reward coefficients are assumed and opportunity enters through overlapping terms;
- the minimum-burden visited-zone rule ties the simulator oracle in the included run;
- simulation repeats treat observations as independent instead of resampling households or people; and
- the equity variables and service gaps are simulator diagnostics, not evidence of fairness.

Recommended next steps are nested household-grouped validation, an untouched test set, time-stamped external supply features, decision-time feature auditing, clustered uncertainty, reward redesign, and randomized or defensibly logged action-response data.
