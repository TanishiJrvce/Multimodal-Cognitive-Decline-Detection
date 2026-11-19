# v0 Branch: Understanding Guide (walkthrough edition)

This guide assumes no prior context. Each section follows the same pattern:

1. **The idea in plain words**: what we did and why anyone would do it.
2. **In formal terms**: the proper technical names and the technical reason it matters.
3. **What we found**: real numbers from the repo's outputs.
4. **Caveat / how to talk about it**: where the result has limits.

Units: everything about energy is in **Wh** (watt-hours) used in one 10-minute reading.

---

# Part A: The foundations

## A.1 What problem are we solving?

**Plain words.** A house has an electricity meter that records how much energy the appliances (fridge, oven, TV, kettle, etc.) used in every 10-minute window. The task is: *given what we know right now, predict how much energy the appliances will use in the next 10 minutes.* This is useful for things like warning about demand peaks or planning how much power a home needs.

**Formal.** Univariate-target, one-step-ahead (10-minute horizon) **time-series forecasting** / regression. The target (the thing to predict) is the column `Appliances`.

**The data.** The public UCI "Appliances Energy Prediction" dataset: one household in Belgium, 19,735 rows, one row every 10 minutes from 11 January to 27 May 2016 (137 days). Besides the target, each row has readings from indoor temperature/humidity sensors (`T1…T9`, `RH_1…RH_9`), outdoor weather (`T_out`, `RH_out`, `Windspeed`, …), `lights` (energy used by lights) and two random-number columns (`rv1`, `rv2`) that the dataset authors added as noise.

## A.2 What is the team's final version (`main`) doing?

**Plain words.** The team turns the raw table into a table of *inputs* (called features) and builds four predictors to compare:

- **Persistence.** The laziest possible predictor: "the next reading will be the same as the last one." No learning at all.
- **Linear Regression.** Draws the best straight-line-style formula through the inputs.
- **Random Forest** and **XGBoost.** Two "decision tree" methods that learn lots of if-then rules.

The 26 inputs (features) are:
- 5 **calendar** features (hour, day, day of week, month, is-weekend),
- 3 **lag** features (energy 10, 20 and 30 minutes ago),
- 3 **rolling** features (average of the last 3 or 6 readings, and their spread),
- 15 **sensor** features (temperatures, humidities, weather, lights).

**Formal.** `build_feature_matrix` creates `Appliances_lag_1..3` (shifted target), `rolling_mean_3`, `rolling_mean_6`, `rolling_std_6` (computed after `shift(1)` so the current value is never included, avoiding **leakage**), and passes through the calendar and sensor columns.

## A.3 Train / validation / test, and why order matters

**Plain words.** To judge a predictor honestly you must test it on data it never saw while learning. So the data is cut into three pieces:
- **Train** (first 70%): the model learns from this.
- **Validation** (next 15%): used while building, to compare and tune.
- **Test** (last 15%): the final exam, looked at once.

Because this is a *time* problem, the cuts are by **date, in order**: train is January to mid-April, validation is mid-April to early May, test is May. That's called a **chronological split**. It mimics real use, where you always predict the future from the past. If you instead shuffled rows randomly, a model could learn from readings 10 minutes before *and after* a test row, which is cheating you can't do in real life.

**Formal.** 70/15/15 chronological holdout: 13,810 / 2,959 / 2,960 rows after dropping 6 warm-up rows whose lags are undefined. **Leakage** means information from the future (or from the target itself) leaking into the model's inputs, producing optimistic metrics.

## A.4 How do we score a prediction?

**Plain words.**
- **MAE (mean absolute error):** on average, how many Wh was the guess off by. MAE = 26 means "typically off by 26 Wh". Easy to understand and not thrown off by a few huge misses.
- **RMSE (root mean squared error):** like MAE, but big misses count much more. A few giant spikes can dominate it.
- **R²:** how much of the ups and downs the model explains. 1 = perfect, 0 = no better than always guessing the average, **negative = worse than guessing the average**.

**Formal.** MAE = mean(|ŷ − y|); RMSE = √mean((ŷ − y)²); R² = 1 − SSE/SST. We report all three; MAE is primary (reason in A1 below).

## A.5 The headline results of the team's models (test set, seed 42)

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **Persistence** | **26.75** | 66.85 | 0.459 |
| Linear Regression | 29.24 | 61.50 | 0.542 |
| Random Forest | 45.56 | 77.02 | 0.282 |
| XGBoost | 57.10 | 84.52 | 0.135 |

**The surprise.** The "lazy" persistence guess has the lowest MAE. All the fancy models are worse on it. *The whole v0 branch is about understanding why, and checking how solid the numbers are.*

## A.6 What is v0?

**Plain words.** `v0` is your branch. It doesn't change the team's code. It holds the investigation: exploring the data, testing what drives the results, and documenting limits. Each commit below is one piece of that.

| # | Commit | What it is |
|---|---|---|
| 1 | `7081a68` updated model card | First model-card rewrite |
| 2 | `e65e138` created and ran some EDA experiments | Data exploration notebook + 7 figures (A1–A7) |
| 3 | `5b8ae40` lag feature ablation | Which features matter (L1, L2) |
| 4 | `eb2e66b` model MAEs in different target regimes | Error by situation type (E3) |
| 5 | `ab4ffba` Train/Val/Test metrics for all 4 models | Overfitting check (E4) |
| 6 | `38efa46` MAE gap analysis | How trustworthy are the gaps (E9) |
| 7 | `fbc3a16` Move figures to reports/figures | Tidy-up |
| 8 | `9a4f525` updated model card | Corrected card |

---

# Part B: Commit `e65e138`, Exploratory Data Analysis (EDA)

**Plain words.** Before building anything, you look at the data to learn what it's like. A good exploration doesn't just plot everything and hope; it first writes down guesses ("I expect X") and then checks each with numbers. Each section A0 to A7 below is one question we asked the data. The answers decide how features are designed, how the split is done, and which metric is used.

**Formal.** Hypothesis-driven **exploratory data analysis (EDA)**: priors from domain reasoning are confirmed, quantified or falsified.

## A0. Is the data clean and regular?

**Plain words.** First check the basics: are any values missing? And does every row come exactly 10 minutes after the previous one? This matters a lot because we're going to build "what was the energy 10 minutes ago" features. If a few rows were missing, "the previous row" might actually be an hour ago and our feature would silently be wrong.

**Formal.** Missing-value audit and timestamp-regularity audit (all consecutive differences == 10 minutes). Regular spacing is a precondition for index-based `shift()` lags to correspond to fixed time offsets.

**What we found.** 0 missing values; all 19,734 gaps are exactly 10 minutes. So lag features are valid.

## A1. What does the target look like? (summary statistics)

**Plain words.** Compute the basic facts about the energy readings: average, middle value (median), spread, smallest/largest, and how they're distributed (a histogram). The key question: is the "average" a good description of a typical reading?

**Formal.** Descriptive statistics, quantiles, **skewness** (asymmetry; positive = long right tail) and **kurtosis** (tail heaviness), histogram on linear and log scales, and the empirical CDF.

**Why it matters technically.** The distribution's shape decides which error metric is fair. With a long right tail, a few extreme readings dominate squared-error metrics like RMSE, so **MAE** is chosen as the primary metric. It also decides what a naive baseline would achieve.

**What we found.**
- Mean 97.7, **median 60**, std 102.5, min 10, max 1,080.
- 75th percentile 100, 95th 330, 99th 577.
- Skewness 3.39, kurtosis 13.7 (strongly right-skewed, heavy tails).
- In plain terms: most readings are low (typically 50–60), and a few are enormous, so the average (97.7) is dragged far above the typical value (60).

## A2. Does the target have distinct "modes of behaviour"? (regimes)

**Plain words.** Look at the house's behaviour: most of the time it's in **standby** (fridge hum, router, devices idling) using about 50 Wh. Occasionally someone cooks, runs the washing machine or the kettle and consumption jumps. We split readings into bands (standby, low-active, medium, high, spike) and measure how much of the time is spent in each.

**Formal.** **Regime characterisation**: partition the target by magnitude and compute proportion and summary statistics per regime. The target is effectively a *mixture* of different generating processes (a baseload floor plus intermittent appliance events).

**Why it matters technically.** If you only report one average error, the large "easy" regime hides what happens in the rare, hard regime. A model could be excellent in standby and terrible during spikes and still look good overall. So **evaluation must be stratified**: report MAE separately for each regime. (We do exactly that in commit 4.)

**What we found.**

| Regime | Share of time | Mean |
|---|---|---|
| Standby (≤60) | **54.4%** | 48.8 |
| Low-active (61–100) | 24.1% | 83.0 |
| Medium-active (101–200) | 11.7% | 131.7 |
| High-active (201–300) | 4.0% | 255.2 |
| Spike (>300) | 5.7% | 444.8 |

**Caveat: correction of a team claim.** Some of the team's READMEs say standby is ">75%" or "~76%". With the ≤60 Wh definition it is **54.4%**. The ≤90 Wh share is 73.6% and ≤100 Wh is 78.6%, so a ">75%" figure only holds for a looser "≤100 Wh" definition. In standby the readings are very tight (std 10.2): the house is basically idle at about 50.

## A3. Are there daily and weekly patterns? (time-related patterns)

**Plain words.** Average the energy by hour of the day, and separately for weekdays and weekends. People have routines: asleep at night (low), cooking in the evening (high). If the picture is flat, time of day tells you nothing. If there's a big gap between the night low and the evening high, then knowing the time of day is valuable.

**Formal.** Group-by-hour (weekday vs weekend) mean and spread, and a day-of-week boxplot. The **peak-to-trough ratio** quantifies the systematic temporal variation that calendar features (`hour`, `day_of_week`, `is_weekend`) can capture. A bigger gap means calendar features carry more value.

**What we found.**
- Weekday night (0–4 AM): 50.6 Wh. Evening peak at **18:00**: 194.9 Wh. Ratio **3.85×**.
- Weekend midday is 18% higher than weekday midday (138.3 vs 117.2), as people are home.

**Caveat.** The pattern is strong, yet the later lag ablation shows calendar features add little *once lag features are present*, because the recent past already tells the model what time-of-day state the house is in.

## A4. How much does the past tell us about the present? (ACF and PACF)

**Plain words.** Take the energy series and compare it with a copy of itself shifted back a little in time. How similar is "now" to "10 minutes ago"? To "20 minutes ago"? To "an hour ago"? The more similar, the more useful that past value is for prediction.

- **ACF (autocorrelation function):** the similarity between now and each past point.
- **PACF (partial autocorrelation function):** the *extra, direct* similarity at each past point after removing what the closer points already explain. Example: if now resembles 20 minutes ago only because both resemble 10 minutes ago, the PACF at 20 minutes will be small.

**Formal.** ACF(k) = corr(yₜ, yₜ₋ₖ). PACF(k) = the correlation at lag k after controlling for lags 1…k−1.

**Why it matters technically.** These two plots decide **which lag features to create and how many**. A big PACF spike at lag 1 followed by small values means an AR(1)-like process: lag-1 is the key feature and a few extra lags only refine. It also sets how strong the Persistence baseline will be, since persistence *is* "use lag 1".

**What we found (ACF):**

| Lag | Time back | Correlation |
|---|---|---|
| 1 | 10 min | **0.753** |
| 2 | 20 min | 0.534 |
| 3 | 30 min | 0.438 |
| 6 | 1 h | 0.324 |
| 24 | 4 h | 0.114 |
| 48 | 8 h | −0.041 |
| 144 | 24 h | 0.217 |

In plain terms: the last reading is a strong clue, the clue fades over a few hours, and there's a small echo exactly one day back (same time yesterday).

**Caveat.** The original model card said 0.83. The measured value is 0.753.

## A5. Does any single input predict the target on its own? (raw correlations)

**Plain words.** For each input (each sensor, `lights`, etc.), compute how closely it moves together with the energy reading at the same moment (correlation, from −1 to +1; 0 means unrelated). This shows how useful each input is *before* we do any clever feature work.

**Formal.** Pearson correlation coefficient between each raw feature and the target at the same timestep. It only captures **linear, contemporaneous** relationships.

**Why it matters technically.** It gives a first ranking of the inputs and flags useless columns. But low correlation doesn't prove a feature is useless: it might help non-linearly, or with a delay (A6 tests that).

**What we found.**
- Strongest: `lights` 0.197, `T2` 0.120, `T6` 0.118. Weakest: `RH_out` −0.152 (negative).
- Almost everything is below 0.1.
- `rv1` and `rv2` (the random columns) both −0.011145, which is *identical*, meaning one column is a copy of the other.

**What it means.** Even the best raw input explains under 4% of the variation (0.197² ≈ 0.039). No single sensor tells you much, which is the first hint that recent energy history will dominate.

## A6. Do sensors react after the appliances run? (thermal inertia)

**Plain words.** Physical idea: when appliances run they give off heat, but a room warms slowly. So room temperature might rise 30–90 minutes *after* energy use. If true, a temperature reading from an hour ago would help predict now. To test it, compare each sensor with the energy series shifted by different amounts, and see where the correlation is highest.

**Formal.** **Lagged cross-correlation** between a sensor and the target for lags −18…+18 (positive lag = the sensor leads the target). The hypothesis predicts a peak at positive lags of 3–9 steps.

**Why it matters technically.** If confirmed, it would justify engineering *lagged sensor* features, a whole extra feature family.

**What we found.**

| Sensor | Peak lag | Peak \|r\| | Meaning |
|---|---|---|---|
| T2 | −15 (2.5 h behind) | 0.172 | sensor lags the target |
| T6 | +1 (10 min) | 0.118 | leads, but negligible |
| T1 | −18 (3 h behind) | 0.166 | sensor lags the target |
| lights | 0 | 0.197 | same moment |

**Caveat / honest reading.** The hypothesis is **not clearly supported**. The peaks for T1 and T2 are at *negative* lags, and the biggest correlations are probably two things (energy and temperature) sharing slow daily cycles, not one causing the other. T6 "leads" by only one step with a tiny correlation. The notebook prints "CONFIRMED" for T6, which overstates it.

## A7. Does the data change over time? (stationarity)

**Plain words.** Does the energy series behave about the same in January as in May (same average, same wobble), or does it drift? This matters because we train on the past and test on the future. If the future behaves differently, the model will struggle for reasons that have nothing to do with how good it is.

**Formal.** **Stationarity** means constant statistical properties over time. Assessed by 24-hour rolling mean/std, the **ADF** test (null hypothesis: non-stationary unit root) and **KPSS** test (null hypothesis: stationary), plus per-split means.

**What we found.** ADF p ≈ 0 (rejects non-stationarity) and KPSS p = 0.10 (cannot reject stationarity): both say the target is stationary in mean. Means: train 98.8, validation 93.4, test 96.9.

**Caveat.** This is about the **target only**. The *input sensors* do drift (they're warmer in May), which comes back in the generalisation analysis. A stable target doesn't guarantee stable inputs.

---

# Part C: Commit `5b8ae40`, Lag feature ablation

**Plain words.** The model has 26 inputs from four families (calendar, lags, rolling, sensors). *Which family is actually doing the work?* The way to find out is an **ablation**: take a family away and see how much worse things get (or better!), and also try using each family *alone*. It's like removing one ingredient at a time from a recipe to see what the dish needs.

**Formal.** **Group ablation** (leave-one-group-out and only-one-group) across Linear Regression, Random Forest and XGBoost. It measures a group's contribution under the real model, which is more reliable than raw correlation (A5) because it accounts for redundancy and interactions. To make every configuration comparable, the first 144 rows are dropped in all of them (so a one-day lag can also be tested on the same rows). Persistence on these same rows scores 26.72.

## L1. Which feature families matter?

**What we found (test MAE in Wh):**

| Configuration | LR | RF | XGB |
|---|---|---|---|
| All 26 features | 29.5 | 44.6 | 78.3 |
| Without sensors | 26.9 | 26.6 | 26.5 |
| Without lags | 36.0 | 50.4 | 79.4 |
| Without rolling | 30.1 | 46.5 | 95.8 |
| **Without lags AND rolling** (no recent history) | 74.6 | 147.9 | 128.7 |
| **Only the 3 lags** | 26.8 | 26.5 | **25.7** |
| Only sensors | 74.9 | 200.8 | 182.1 |
| Only calendar | 50.6 | 57.0 | 49.5 |

**What this means in plain words:**
1. **Recent history is almost everything.** Remove it and the error multiplies 2.5 to 5.6 times. Use only three lag features and you match the lazy baseline.
2. **Sensors don't help, and for the tree models they actively hurt.** Dropping all 15 sensor columns takes Random Forest from 44.6 to 26.6, and XGBoost from 78.3 to 26.5. The sensors are an opportunity to overfit.
3. **Dropping the lags alone only hurts a little**, because the rolling averages carry the same information. In fact `rolling_mean_3` is exactly the average of lags 1, 2 and 3, so the lag and rolling features are partly duplicates (formally, an exact linear dependency in the feature matrix).
4. The best result (XGBoost, only lags, 25.7) beats persistence by about 1 Wh. That is smaller than the noise in these numbers (see E9 below).

## L2. How many lags should we use?

**Plain words.** ACF said recent past matters; how far back should the model look? We try 0, 1, 2, 3, 6 and 12 lags, and also a single value from exactly one day ago (lag 144).

**What we found (test MAE):**

| Lags | LR | RF | XGB |
|---|---|---|---|
| 0 | 74.6 | 147.9 | 128.7 |
| 1 | 30.1 | 48.8 | 96.0 |
| 3 (team's choice) | 30.1 | 46.5 | 95.8 |
| 6 | 29.4 | 42.2 | 71.7 |
| 12 | 28.8 | 36.9 | 41.8 |
| Only lag 144 | 74.3 | 148.4 | 123.2 |

**What this means.** The big jump is from 0 to 1 lag. After that, Linear Regression barely improves, but the tree models keep improving as lags are added (more lags seem to dilute the influence of the sensors). A single value from a day ago is useless on its own, despite the small daily echo in the ACF, because it adds nothing beyond the recent past.

**Caveat: XGBoost numbers are noisy.** XGBoost results move by about ±5 Wh depending on random seed and CPU thread count (see E7 below). So small XGBoost differences shouldn't be over-read. This is also one reason why the "full" XGBoost here (78.3) differs from the pipeline's 57.10 (the other reason is the 144 dropped rows).

---

# Part D: Commit `eb2e66b`, Error by regime (E3)

**Plain words.** One overall error number can hide where a model is good or bad. So we split the *test* rows by what the house was actually doing (standby, active, spike) and compute the error separately in each group. We also compute the **bias**: does the model tend to guess too high or too low?

**Formal.** **Stratified evaluation** by target regime (Standby ≤60, Active (60, 300], Spike >300). For each model: MAE, **mean signed error** (prediction − actual; positive = over-prediction) and each regime's share of the model's total absolute error. The thresholds were fixed before looking at results.

**Why it matters technically.** It implements the A2 conclusion. Signed error separates *systematic* error (bias) from random scatter, which tells you what kind of fix is needed.

**What we found (test MAE):**

| Regime (share of rows) | Persistence | LR | RF | XGB |
|---|---|---|---|---|
| Standby (50.7%) | **8.8** | 14.2 | 13.3 | 28.5 |
| Active (44.9%) | **29.5** | 29.3 | 71.6 | 80.8 |
| Spike (4.4%) | 205.8 | 202.5 | 152.5 | **145.9** |

Bias (test): Random Forest +62 and XGBoost +73 Wh in Active; XGBoost +27.6 in Standby; every model −111 to −174 on spikes (guesses too low).

**What this means:**
1. Persistence is very hard to beat when the house is quiet or steadily active: successive readings are similar, so copying the last one works.
2. The tree models' problem is a **consistent over-prediction in the Active regime** (they guess about 60 to 70 Wh too high). That single regime accounts for 63–70% of their total error.
3. Everyone under-predicts spikes (a spike averages about 445 Wh and nobody sees it coming), but the trees do less badly than persistence.
4. The first model card said trees "over-smooth spikes". The data doesn't support that as stated: trees do *relatively better* on spikes; their failure is elsewhere.

**Caveat.** The regimes are defined using the *actual* answer, so this is a diagnostic. A deployed model can't know the regime in advance.

---

# Part E: Commit `ab4ffba`, Train vs validation vs test (E4)

**Plain words.** A good model should do about as well on new data as on the data it learned from. If it does far better on the old data, it has memorised instead of learned (**overfitting**; like a student who memorised last year's exam answers). We measure each model on all three data pieces and compare.

**Formal.** **Generalisation-gap analysis**: MAE, R² and RMSE/MAE on train, validation and test, plus validation − train and test − validation differences. The helper `_common.py` rebuilds the team's exact features, split and hyperparameters so nothing in `src/` was changed. Using a chronological split makes this also a test of how well the model copes with drift over time.

**What we found:**

| Model | Train MAE | Val MAE | Test MAE | Test/Val | R² train / val / test |
|---|---|---|---|---|---|
| Persistence | 30.91 | 26.15 | 26.75 | 1.02 | 0.518 / 0.481 / 0.459 |
| Linear Regression | 31.59 | 29.10 | 29.24 | 1.00 | 0.596 / 0.566 / 0.542 |
| Random Forest | 19.34 | 33.29 | 45.56 | 1.37 | 0.845 / 0.506 / 0.282 |
| XGBoost | 20.34 | 34.25 | 57.10 | 1.67 | 0.854 / 0.501 / 0.135 |

**What this means:**
1. **Persistence and Linear Regression are stable**: about the same error everywhere (simple models, little to memorise).
2. **Random Forest and XGBoost overfit.** They fit the training period almost too well (R² 0.85) and fall to 0.28 and 0.14 on the test period. Most of the fall happens between validation and test, so validation alone would not have warned you fully.
3. The RMSE/MAE ratio (a hint whether errors come from a few huge misses or many medium ones) drops for the trees from about 2.2 to 1.5–1.7 on test, which means their errors are many moderate over-predictions, matching the bias seen in E3.

**Putting the evidence together (the explanation).**
- Sensors hurt the trees (L1).
- Trees over-predict in Active periods (E3).
- Sensor readings drift between January–April and May (E5 below), so rules learned on the training-period sensor values don't transfer.

The most consistent explanation is **overfitting to sensor patterns that don't hold later**. **Caveat:** this is supported by several pieces of evidence, but no experiment *proves* the cause.

---

# Part F: Commit `38efa46`, Are the gaps real? (E9)

**Plain words.** The test set is only about 20 days. If we'd happened to get a different 20 days, would the ranking change? To find out, we **re-score the models on thousands of re-built test sets** and see how much each number wobbles. We rebuild each test set by picking whole days at random (with repeats) and gluing them together. Whole days are used because neighbouring 10-minute readings are not independent: picking single rows would pretend we have more independent information than we do, giving falsely tight results.

**Formal.** **Moving-block bootstrap** with 144-row (one-day) blocks, 2,000 resamples, **paired** (all models scored on the same resampled rows). Reports 95% percentile confidence intervals for each model's MAE and for the *difference from persistence*. Pairing removes shared "this day was hard for everyone" variation, so the interval on the difference is tighter than the intervals on the levels.

**What we found:**

| Model | Test MAE | 95% CI | Diff vs persistence | 95% CI of diff |
|---|---|---|---|---|
| Persistence | 26.75 | 20.9–31.7 | n/a | n/a |
| Linear Regression | 29.24 | 23.5–33.7 | +2.5 | 0.4 to 4.8 |
| Random Forest | 45.56 | 38.1–51.3 | +18.8 | 13.3 to 24.3 |
| XGBoost | 57.10 | 47.5–65.7 | +30.4 | 22.7 to 39.4 |

**How to read it.** If the interval on "model minus persistence" stays above 0, the model is reliably worse. Random Forest and XGBoost clearly are. Linear Regression barely clears zero, so the safe wording is "about equal to persistence". **No model reliably beats persistence on MAE.**

**Caveat.** The intervals only capture "which days are in the test set". They don't capture training randomness (XGBoost varies by about ±5 Wh, E7) or other seasons (the window is only 20 days of May).

---

# Part G: Commit `fbc3a16`, Moving figures

**Plain words.** Housekeeping. Three of our charts had been saved beside their data files. The project convention is charts in `reports/figures/` and data tables in `reports/evidence/<experiment>/`. We moved them with `git mv` (which keeps their history) and updated the two scripts that generate them.

**Caveat.** The scripts weren't re-run after the path change.

---

# Part H: Commits `7081a68` and `9a4f525`, The model card

**Plain words.** A **model card** is a short document that says what a model is, what data it used, how well it does and where it fails. Anyone relying on the model reads it, so mistakes in it mislead people. When we reviewed the first version (`7081a68`) against our later experiments and the code, several statements were wrong, so `9a4f525` corrects them:

| Original statement | Problem | Correction |
|---|---|---|
| "The feature set has no lag features" | The team's code does build lags and rolling features (26 features) | Lags are the dominant signal (L1) |
| Random Forest: 200 trees, depth 10 | Code uses 100 trees, depth 15, min_samples_leaf 4 | Fixed; XGBoost's `colsample_bytree` and `n_jobs` added |
| Lag-1 autocorrelation ≈ 0.83 | Measured 0.753 | Fixed |
| Persistence error in standby "<5 Wh" | Measured 8.8 Wh | Fixed |
| Trees "over-smooth spikes" | E3 shows trees do better on spikes | Replaced with the over-prediction finding |
| (missing) | XGBoost results vary by seed and thread count | Reproducibility note added |

---

# Part I: Experiments run but not committed

You chose not to track these, but you should understand them, especially E7, which explains why different documents show different XGBoost numbers.

**E1: Are the random columns really useless?** (`rv1`, `rv2`.) We measured how much shuffling them hurts predictions and whether adding them changes accuracy. *Result:* essentially zero effect (inside the seed-to-seed wobble), and `rv1` equals `rv2`. This confirms the dataset's noise columns are safe to ignore.

**E2: Is one particular lag irreplaceable?** We scrambled `lag_1` and retrained. *Result:* models barely suffer (RF 41.9, LR 29.25), because lag 2, lag 3 and the rolling mean carry the same information. This shows how redundant the memory features are.

**E5: Does the data drift between train and test?** We compared the sensor and target distributions per split (Kolmogorov–Smirnov tests, and feature differences measured in training standard deviations), and each model's average signed error per split. *Result:* the target's average is stable (99 / 93 / 97), but sensors drift strongly: test temperatures are 2 to 3 training standard deviations higher (T3 +2.82, T2 +2.70, T1 +2.55). The trees over-predict on test (+28 and +42 Wh) and Linear Regression doesn't (+1). *This is the best available explanation of the tree failure, but it's consistent with the data, not proven.*

**E6: Why must the split be chronological?** Same data, same sizes, random vs time-ordered split. *Result:* with a random split Random Forest and XGBoost look much better (test MAE about 28; R² inflates by +0.34 and +0.50), while persistence and Linear Regression barely change (+0.04). A random split makes the trees look good only because they can "peek" at neighbouring rows. This justifies the team's chronological split.

**E7: How repeatable are the results?** We trained with 10 random seeds, and with different numbers of CPU threads. *Result:* Random Forest is rock-steady (45.57 ± 1.32 across seeds, identical for any thread count). **XGBoost is not**: 62.0 ± 4.7 across seeds, and at the *same* seed anywhere from 55.0 to 66.3 depending on thread count. So 57.10, 60.84 and 60.97 in different documents are the same code on different machines.

**E8: Where does persistence make its errors?** By transition type (standby→active, stay, active→standby). *Result:* sustained activity causes 70–80% of the error. Largely repeats E3.

---

# Part J: Inconsistencies found in the team's materials

Present these as observations for the next phase, not as accusations.
- Standby share stated as ">75%/~76%"; data says 54.4% for ≤60 Wh.
- READMEs say 80/20 split; code uses 70/15/15.
- `main`'s model card table doesn't match its own `model_comparison.csv`.
- No agreed final model: `sj`'s `predict.py` defaults to Random Forest while its README highlights Linear Regression.
- `n_jobs=-1` makes XGBoost results depend on the machine.
- The DVC remote is a local folder, so `dvc pull` won't work for others.
- `main`'s notebook has no executed cells.

---

# Part K: Defending the work, likely questions

**Why is the lazy persistence guess so hard to beat?**
The last reading is a very strong clue (correlation 0.753), and over half the time the house is in standby where successive readings differ by under about 9 Wh. The lag ablation shows that even advanced models only match persistence when given just the last few readings.

**Why do the tree models do worse on test?**
They fit training almost perfectly (R² 0.85) but fall to 0.14–0.28 on test (E4). They over-predict in active periods (E3). Their error nearly vanishes if the sensor columns are removed (L1). And the sensor values drift between training and test (E5). The explanation is overfitting to sensor patterns that don't hold later; supported, but not causally proven.

**Which model is best?**
It depends on the metric: persistence on MAE (Linear Regression is statistically close), Linear Regression on RMSE and R² among learned models, the tree models on spikes. There is no clear winner, and no model reliably beats persistence on MAE.

**Why a chronological split?**
Real use predicts the future. A random split lets the model interpolate between neighbouring rows, inflating tree R² by +0.34 to +0.50 (E6).

**Why MAE rather than RMSE?**
The target is heavily right-skewed (skewness 3.39), so RMSE would be dominated by rare spikes. MAE is also directly readable in Wh.

**How reliable are your numbers?**
Persistence, Linear Regression and Random Forest are stable. XGBoost carries about ±5 Wh of seed/thread noise, so only large XGBoost differences are meaningful. The test set is about 20 days of a single month.

**What are the weaknesses of your work?**
- The thermal-inertia idea wasn't supported (A6).
- The sensor-drift explanation isn't causally isolated.
- One house, one season.
- Regime analysis uses the true target to define the groups.
- Ablation XGBoost numbers differ from the pipeline because of dropped rows and thread noise.

---

# Part L: Glossary

| Term | Plain meaning | Formal |
|---|---|---|
| Target | The thing we predict | `Appliances` (Wh per 10 min) |
| Feature | An input the model uses | A column of X |
| Lag | The value k steps ago | `Appliances_lag_k` (shifted target) |
| Rolling feature | Average of recent readings | `rolling_mean_k`, `rolling_std_k` on shifted data |
| Persistence | "Same as last time" | ŷₜ = yₜ₋₁ |
| ACF / PACF | How much the past resembles now (total / direct) | (Partial) autocorrelation at each lag |
| Stationary | Behaves the same over time | Constant mean/variance; ADF and KPSS |
| MAE / RMSE / R² | Typical miss / miss weighted to big errors / share explained | See A.4 |
| Chronological split | Train on past, test on future | Time-ordered holdout |
| Leakage | Model peeks at information it wouldn't have | Future/target information in the inputs |
| Overfitting | Memorising training data | Large train vs held-out gap |
| Distribution shift | New data looks different | Input/target distribution changes over time |
| Regime | Type of situation | Standby / Active / Spike |
| Ablation | Remove something and see what breaks | Controlled removal of a feature group |
| Bias (signed error) | Tendency to guess too high/low | mean(ŷ − y) |
| Bootstrap CI | How much would this number wobble with different data? | Resampling-based confidence interval |
| Block bootstrap | Same, but resampling whole days | Preserves autocorrelation |
| Seed | Number fixing the randomness | `random_state` |

---

# Part M: File map

```
energy-monitoring-mlops/
├── docs/model_card.md                         (7081a68, 9a4f525)
├── notebooks/01_eda.ipynb                     (e65e138)
├── experiments/
│   ├── 01_lag_feature_ablation.py             (5b8ae40)
│   ├── 03_regime_analysis.py                  (eb2e66b)
│   ├── _common.py, 05_generalisation_gap.py   (ab4ffba)
│   └── 10_bootstrap_ci.py                     (38efa46)
└── reports/
    ├── figures/    A1–A7, L1, L2, E3, E4, E9
    └── evidence/   lag_ablation/, regime_analysis/,
                    generalisation_gap/, bootstrap_ci/
```
