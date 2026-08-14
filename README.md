# Equity-Aware Contextual Bandits for Alternative Delivery Recommendations

An exploratory, simulation-based study of dynamic out-of-home (OOH) and alternative delivery location (ADL) recommendations. The project combines 2024 NYC Citywide Mobility Survey behavior, activity-chain features, static recommendation rules, and online contextual-bandit policies.

The central question is whether a policy can select a feasible non-home delivery option for a person-day while accounting for trip-chain compatibility, pickup burden, local opportunity, and equity-related diagnostics.

## What this project demonstrates

- survey cleaning and multi-table feature engineering;
- household-grouped train/test splitting;
- imbalanced classification and probability diagnostics;
- construction of action eligibility from daily activity chains;
- explicit utility, burden, service, and reward assumptions;
- static policy baselines;
- epsilon-greedy, LinUCB, diagonal linear Thompson sampling, and constrained LinUCB;
- repeated online simulation with regret, burden, action-share, and group diagnostics; and
- sensitivity and leave-one-zone-out analyses.

This is not an offline reinforcement-learning analysis of a delivery platform. The survey does not observe recommendation offers, rejections, incentives, locker occupancy, or action-specific counterfactual outcomes. The reported policy results are diagnostics from a constructed simulator; they are not causal estimates or deployment evidence.

## Study design

The notebook links household, person, day, and trip tables. It builds person-day activity chains and a broad revealed-behavior target (`ooh_any`), then fits alternative supervised models on a household-grouped split.

For each held-out person-day, the simulator creates six actions:

1. home delivery;
2. an ADL in the home zone;
3. a work- or school-compatible ADL;
4. an errand- or shopping-compatible ADL;
5. a transit-compatible ADL; and
6. an ADL in the highest-opportunity visited zone.

An ADL is eligible only when its activity-chain condition is met and its opportunity score is above zero. The response model supplies a common revealed-behavior probability; hand-specified action coefficients then adjust simulated acceptance and reward. The default reward includes service value, opportunity, consolidation, behavior targeting, burden, recommendation cost, and equity-related terms. Every coefficient is visible in the notebook and in `src/equity_bandits/policy.py`.

The included run uses a survey-derived zone opportunity proxy. The notebook also contains an optional OSM/GTFS builder, but no live external opportunity data were used for the included results.

## Included aggregate results

The compact `results/` directory contains aggregate evidence only. It excludes raw survey records, identifiers, row-level predictions, action tables, model binaries, and event-level bandit logs.

### Data and supervised layer

| Quantity | Included run |
|---|---:|
| Model-ready person-days | 20,247 |
| Positive `ooh_any` observations | 716 (3.5363%) |
| Household-held-out person-days | 5,025 |
| Held-out positive rate | 3.1244% |
| Top row in the model-selection table | M1 core + opportunity, balanced random forest |
| Held-out PR-AUC for that row | 0.125855 |
| Held-out ROC-AUC for that row | 0.729422 |
| Held-out Brier score for that row | 0.044561 |

These are predictive metrics for a rare, broad revealed-behavior outcome. The current notebook ranks model alternatives on the same holdout later used for evaluation, so they should be treated as exploratory. A final study needs nested household-grouped model selection and an untouched evaluation set.

### Simulated policy layer

Static rows below report mean **expected simulator reward**. Bandit rows show both mean expected reward of chosen actions and mean realized reward sampled during 100 simulation runs. Reward is in arbitrary simulator units, so values are meaningful only under the stated assumptions.

| Policy | Metric | Value |
|---|---|---:|
| Minimum-burden visited ADL | Mean expected reward | 0.038821 |
| Simulator oracle | Mean expected reward | 0.038821 |
| Constrained LinUCB | Mean expected reward chosen | 0.022242 |
| Constrained LinUCB | Mean realized reward | 0.022365 |
| LinUCB | Mean expected reward chosen | 0.018592 |
| Epsilon-greedy linear | Mean expected reward chosen | 0.015348 |
| Diagonal linear Thompson sampling | Mean expected reward chosen | -0.006192 |

The minimum-burden visited-zone rule exactly ties the oracle in this run. That collapse is a diagnostic of the reward and burden construction, not evidence of real-world optimality. The strongest online policy, constrained LinUCB, remains below that static rule in expected simulator reward.

The source run originally combined static expected reward with bandit realized reward under one generic column. The public notebook and `results/combined_policy_comparison.csv` label each metric explicitly. The public notebook also matches the uncertainty bar in its expected-reward plot to expected reward rather than realized reward.

## Data access and privacy

The source is the [NYC Department of Transportation Citywide Mobility Survey](https://www.nyc.gov/html/dot/html/about/citywide-mobility-survey.shtml). NYC DOT provides the 2024 dataset, questionnaire, user guide, and survey-zone files and states that identifying and location information is removed from the public dataset to protect privacy.

This repository does not redistribute CMS microdata. Download the current files directly from NYC DOT, review the accompanying documentation and terms, and place local copies under `data/raw/` or configure their locations with environment variables.

The notebook's default local filenames are:

```text
data/raw/
  Citywide_Mobility_Survey_-_Trip_2024_20260707.csv
  2024_persons.csv
  2024_day.csv
  2024_household.csv
  2024_dictionary.xlsx
```

Export names can change. Rename the downloaded files or set:

- `CMS_TRIP_FILE`
- `CMS_PERSON_FILE`
- `CMS_DAY_FILE`
- `CMS_HOUSEHOLD_FILE`
- `CMS_DICTIONARY_FILE`

`CMS_DATA_DIR` changes the common input directory. `CMS_OUTPUT_DIR` changes the generated-output directory. `CMS_ZONE_OPPORTUNITY_FILE` supplies an optional zone-level opportunity table. `CMS_GTFS_DIR` points the optional GIS builder to a directory of local GTFS ZIP files.

Generated row-level files go to `artifacts/run/`, which is ignored by Git. Inspect all outputs before sharing them.

## Setup and run

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the package and notebook dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
python -m jupyter lab notebooks/equity_aware_contextual_bandits.ipynb
```

The optional live GIS cells require additional native geospatial dependencies:

```bash
python -m pip install -e ".[notebook,geo]"
```

They also depend on external APIs and feeds that may change. The live builder is disabled by default; pin or archive source snapshots for a reproducible study.

Run the extracted policy tests with:

```bash
python -m unittest discover -s tests -v
```

A complete notebook run can be computationally expensive because it fits several tree ensembles, performs permutation and zone-holdout analyses, and repeats bandit simulation 100 times. Use the visible control flags for exploratory runs, and record any changes when reporting results.

## Repository layout

```text
.
├── notebooks/
│   └── equity_aware_contextual_bandits.ipynb
├── results/
│   ├── figures/
│   ├── combined_policy_comparison.csv
│   ├── run_summary.json
│   └── supervised_model_results.csv
├── src/equity_bandits/
│   ├── __init__.py
│   └── policy.py
├── tests/
│   └── test_policy.py
├── CITATION.cff
├── QUALITY_REPORT.md
├── pyproject.toml
└── requirements.txt
```

## Limitations and next steps

- The response target measures revealed package-location behavior, not acceptance of a specific recommendation. The broad outcome also combines distinct receipt and pickup processes.
- Model alternatives are selected on the held-out set; calibration is row-stratified rather than household-grouped. Use nested, group-aware validation.
- The survey-derived opportunity proxy is demand/activity information, not a time-stamped measure of actual locker or pickup-point supply. It is constructed before the split and can leak aggregate held-out context.
- Some whole-day activity-chain and package-behavior features may not be available at recommendation time.
- Reward coefficients are assumptions and count opportunity through several terms. The minimum-burden rule tying the oracle shows that the current design needs redesign and broader sensitivity analysis.
- Intercept calibration occurs before ineligible baseline rows are masked, so achieved eligible-action acceptance can differ from the calibration target.
- Repeated simulation treats events as independent and does not refit the response model or environment. Uncertainty should be clustered by household or person.
- The equity flag, bonus, burden constraint, and service-rate gap are simulated diagnostics. They do not establish fairness.
- Optional live GIS features require source versioning and more careful treatment of boundaries, duplicate transit stops, and changing APIs.
- Field-specific missing-value rules, uniqueness assertions, and cache fingerprints should be added before a formal analysis.

The most important next step is action-response data: randomized recommendations or defensible logged propensities with outcomes. Without that evidence, this repository is best used to demonstrate a transparent modeling workflow and to identify what a deployable evaluation would require.

## Attribution and reuse status

Research and implementation: Mustafa Fardin. Survey data are provided by NYC DOT and remain subject to their source documentation and terms.

No license file is included because ownership and any prior team contributions have not yet been confirmed. Default copyright restrictions therefore apply; contact the author before reusing the code beyond what applicable law permits.
