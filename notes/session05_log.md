# Learning log, Session 5

## Part A: evaluating models (10 Nov 2026)

**Course:** Machine Learning Specialization, Course 2 Week 3, Parts A–B (training / cross-validation / test sets, model selection, bias and variance, baseline level of performance, learning curves)

**Question:** Sessions 3 and 4 never beat the simple guess. Is that because the models are too simple (high bias), too flexible for so little data (high variance), or because the inputs hold almost no information about what comes next? And would more data help?

**Setup, decided before fitting:**
- Same rows, inputs and targets as Sessions 3–4: 96 consolidation episodes in 24 countries; ΔCAPB and the four lagged controls; inflation capped at ±30 %; original and year-adjusted "weak growth" targets with cut-offs from the training countries only.
- Model family: logistic regression on polynomial features of the five inputs (degree 1, 2 or 3 → 5, 20 or 55 features), z-scored on the training countries only, λ from 0.01 to 1000 (C = 1/λ).
- J_train and J_cv are log loss without the λ term. The baseline level of performance is the simple guess (the training share of "weak").
- Cross-validation: the 20 training countries in 4 groups (15 to 20 episodes each). Each fold trains on about 53 episodes and scores only its left-out group.

**Train / cross-validation / test in this project**
- Training set: the countries a model learns from. Cross-validation set: training countries held back to choose the degree and λ. Test set: the six hidden countries (Belgium, Greece, Lithuania, Malta, Slovakia, Spain).
- The test set should be used once. Ours has been read in every session since Session 1, so it is no longer fully clean, even though nothing was tuned on it on purpose (apart from the Session 3 leak, since fixed). That is why "every country hidden in turn" and the 500 random splits matter more.

**J_train and J_cv against λ (cross-validation on the 20 training countries)**

| λ | Original, degree 1: J_train / J_cv | Original, degree 2 | Original, degree 3 | Year-adjusted, degree 1 | Year-adjusted, degree 2 | Year-adjusted, degree 3 |
|---|---|---|---|---|---|---|
| 0.01 | 0.617 / 0.877 | 0.338 / 2.486 | 0.116 / 2.881 | 0.629 / 0.901 | 0.379 / 2.047 | 0.125 / 2.887 |
| 1 | 0.619 / 0.823 | 0.459 / 1.077 | 0.372 / 1.201 | 0.631 / 0.860 | 0.482 / 0.898 | 0.377 / 1.365 |
| 10 | 0.640 / 0.732 | 0.560 / 0.760 | 0.504 / 0.787 | 0.646 / 0.764 | 0.563 / 0.660 | 0.504 / 0.926 |
| 30 | 0.659 / 0.706 | 0.607 / 0.696 | 0.565 / 0.682 | 0.663 / 0.722 | 0.608 / **0.643** | 0.564 / 0.763 |
| 100 | 0.676 / 0.698 | 0.649 / **0.680** | 0.618 / 0.702 | 0.678 / 0.703 | 0.649 / 0.658 | 0.619 / 0.750 |
| 1000 | 0.687 / 0.698 | 0.683 / 0.694 | 0.675 / 0.697 | 0.688 / 0.697 | 0.684 / 0.690 | 0.676 / 0.701 |
| Simple guess | J_cv 0.698 | | | J_cv 0.696 | | |

- **The textbook picture.** With little regularization, J_train is low and J_cv is far above it: **high variance**. Degree 3 with λ = 0.01 fits the training countries almost perfectly (J_train 0.12) and is terrible on the cross-validation countries (J_cv 2.9). With a lot of regularization, J_train and J_cv both rise or fall to the simple guess.
- **Degree 1 (the Session 3 model) never gets below the guess** in cross-validation: its best J_cv is 0.698 at λ = 100–1000, the same as the guess.
- **Degree 2 dips below the guess:** J_cv 0.680 vs 0.698 (original, λ = 100) and 0.643 vs 0.696 (year-adjusted, λ = 30). Cross-validation picks these. But the lowest J_cv among 27 options scored on the same 4 groups is optimistic: that is the course's reason for a separate test set.

**Learning curves (train on 2 to 16 training countries, score 4 others, 200 repeats)**

| At 16 training countries (about 57 episodes) | J_train | J_cv | Simple guess |
|---|---|---|---|
| Original, chosen by cross-validation (degree 2, λ = 100) | 0.650 | 0.695 | 0.694 |
| Original, flexible (degree 3, λ = 0.1) | 0.255 | 3.18 | 0.694 |
| Year-adjusted, chosen (degree 2, λ = 30) | 0.598 | 0.714 | 0.694 |
| Year-adjusted, flexible (degree 3, λ = 0.1) | 0.243 | 2.76 | 0.694 |

- **The chosen models:** J_cv falls and J_train rises as countries are added, and they close in on the simple guess from both sides. More data makes them *as good as* the guess, not better.
- **The flexible models:** a huge, slowly closing gap (high variance). More data would help them, but they start so far above the guess that 26 EU countries could never be enough.
- **Diagnosis:** in course terms, the well-regularized models have no variance problem, and their J_train is only a little below the guess. If the best achievable performance were much better than the guess, that would be high bias, and a more flexible model would fix it. But the more flexible models (degree 3 here, the network in Session 4) do not help. The limit is most likely the information in the data: these inputs say little about TFP growth over the next years. More countries do not exist; more years would help only slowly.

**The test, once: the cross-validation choice through the three tests**

| Test | Original (degree 2, λ = 100) | Year-adjusted (degree 2, λ = 30) |
|---|---|---|
| Six hidden: log loss vs baseline | 0.667 vs 0.710 | 0.624 vs 0.718 |
| Six hidden: accuracy vs baseline | 52 % vs 40 % | 52 % vs 40 % |
| Each hidden in turn: log loss vs baseline | 0.919 vs 0.694 (−32.4 %) | 0.704 vs 0.700 (−0.6 %) |
| Each hidden in turn: accuracy vs baseline | 58.3 % vs 49.0 % | 60.4 % vs 42.7 % |
| Countries won | 15 of 24 | 14 of 24 |
| 500 random splits: log loss better | 77 % (median +2.9 %) | 69 % (median +3.5 %) |
| Settings chosen across the 24 rounds | (2, 30) ×13, (2, 10) ×8, (1, 300), (2, 100), (3, 10) | (2, 30) ×11, (2, 10) ×5, (2, 100) ×5, (2, 300), (1, 10), (1, 300) |

- **Luxembourg again.** With the original target, Luxembourg's round chose degree 3 with λ = 10 and scored a log loss of 5.91 against 0.674 for the guess: that one round costs 39.3 points, more than the whole −32.4 %. With the year-adjusted target its round chose degree 1 with λ = 10 (1.313 vs 0.691, −4.6 points). The next worst rounds (Hungary, Estonia, Spain) cost less than 1 point each.
- Without Luxembourg's round the overall result would be roughly +7 % (original) and +4 % (year-adjusted). Together with 69–77 % of random splits, this is the most frequent small gain so far. But one unstable tuning round still decides the sign, the gain is small, the 500 splits use the setting chosen once (slightly optimistic), and no placebo check has been run on degree 2. A lead to check, not a usable predictor.

**Placebo test for the degree-2 model (added 10 Nov 2026)**
- Within each country, the ΔCAPB values were shuffled across years, so each country keeps its number of episodes and their sizes but *which years* count as consolidations becomes random. The model was rerun on each shuffled panel with every country hidden in turn.
- Fast version: 500 shuffles, settings fixed at the section 3 choice (degree 2; λ = 100 / 30). Slightly optimistic and may favour the real episodes, as in Session 3. Clean version: 100 shuffles, degree and λ chosen again by cross-validation inside every round, so nothing is tuned on the real episodes.

| | Original, fast | Original, clean | Year-adjusted, fast | Year-adjusted, clean |
|---|---|---|---|---|
| Shuffles | 500 | 100 | 500 | 100 |
| Real: log loss vs baseline | +4.1 % | −32.4 % | +5.6 % | −0.6 % |
| Shuffled: median (middle 90 %) | +0.4 % (−6.4 to +6.2) | −6.5 % (−40.0 to +5.2) | −3.5 % (−19.5 to +7.9) | −7.6 % (−29.4 to +4.1) |
| Shuffles at least as good (log loss) | 15 % | 94 % | 10 % | 15 % |
| Real: accuracy minus baseline | +11.5 points | +9.4 | +16.7 | +17.7 |
| Shuffled accuracy: median (middle 90 %) | +3.1 (−3.1 to +18.7) | +1.0 (−4.2 to +18.8) | +5.2 (−2.1 to +18.8) | +1.0 (−2.1 to +13.6) |
| Shuffles at least as good (accuracy) | 12 % | 9 % | 6 % | **2 %** |
| Countries won: real vs shuffled median | 16 vs 14 | 15 vs 14 | 13 vs 14 | 14 vs 14 |

- **On log loss, the real episodes are not unusual** in any version: 10–15 % of shuffles do at least as well, and with the original target and tuning in every round, 94 % do (the real result is worse than most shuffles, because of Luxembourg's round). The small degree-2 gain is not specific to the consolidation years.
- **On accuracy, one version stands out:** with the year-adjusted target and tuning in every round, only 2 % of shuffles get as many yes/no calls right. The other versions are 6–12 %. This matches the Session 3 placebo (2–3 % on accuracy, not unusual on log loss): a faint pattern in which side of 50 % the predictions fall, tied to the real years, but not in how confident they are. With 8 comparisons, one 2 % is weak evidence.
- **Unstable tuning is general, not only Luxembourg:** the shuffled panels with tuning in every round have a median of −6.5 % and −7.6 %, and some fall below −40 %. When the best setting has to be chosen on about 50 episodes, a badly wrong round is common.
- **Conclusion:** the degree-2 lead does not survive the placebo on log loss. Still no usable predictor.

**Why Luxembourg goes wrong (checked 10 Nov):** its trade openness in the five episodes is 240 to 376 % of GDP (z-scores 1.9 to 4.0 among the 96 episodes); the median episode is 109 %, and the next most open countries average 150 to 207 % (Hungary, Malta, Ireland). When Luxembourg is hidden, the model has to extrapolate far outside what it was trained on, and squared or cubed terms make that extreme. The thesis does not have this problem because country fixed effects absorb each country's usual openness. Dropping Luxembourg would choose the sample by the result; a rule for all countries (log of trade openness, or capping inputs at the training range when predicting) is the better fix, decided before rerunning.

**A rule for every country: the log of trade openness (added 10 Nov 2026)**
- Decided before running: use the natural log of trade openness for every country; nothing else changes (same degrees, λ values, cross-validation groups, targets and tests). Luxembourg stays in. Its episodes' z-scores fall from 1.9–4.0 to 1.7–2.7.

| Test | Original, raw | Original, log | Year-adjusted, raw | Year-adjusted, log |
|---|---|---|---|---|
| Chosen on the training countries (degree, λ) | (2, 100) | (2, 30) | (2, 30) | (2, 10) |
| Lowest J_cv vs simple guess | 0.680 vs 0.698 | 0.674 vs 0.698 | 0.643 vs 0.696 | 0.614 vs 0.696 |
| Six hidden: log loss vs baseline | 0.667 vs 0.710 | 0.620 vs 0.710 | 0.624 vs 0.718 | 0.594 vs 0.718 |
| Six hidden: accuracy vs baseline | 52 % vs 40 % | 72 % vs 40 % | 52 % vs 40 % | 60 % vs 40 % |
| Each hidden in turn: log loss vs baseline | 0.919 vs 0.694 (−32.4 %) | **0.682 vs 0.694 (+1.7 %)** | 0.704 vs 0.700 (−0.6 %) | **0.686 vs 0.700 (+1.9 %)** |
| Each hidden in turn: accuracy vs baseline | 58.3 % vs 49.0 % | 62.5 % vs 49.0 % | 60.4 % vs 42.7 % | 61.5 % vs 42.7 % |
| Countries won | 15 of 24 | 16 of 24 | 14 of 24 | 13 of 24 |
| 500 random splits: log loss better | 77 % (median +2.9 %) | **84 % (median +6.8 %)** | 69 % (median +3.5 %) | 64 % (median +3.3 %) |
| Luxembourg's round (effect on overall) | −39.3 points | −6.6 points | −4.6 points | −3.0 points |

- **The rule fixes most of the Luxembourg problem.** Its round still goes worst (log loss 1.554 vs 0.674 on the original target), but costs 6.6 points instead of 39.3. Tuning is also steadier: degree 2 is chosen in 22 of 24 rounds on the original target.
- **For the first time a model beats the simple guess in all three tests**, on both targets: +1.7 % and +1.9 % with every country hidden in turn, and better in 84 % and 64 % of random splits.
- **But read it carefully.**
  - The gain is small: about 2 % in log loss with every country hidden in turn, and the year-adjusted model wins only 13 of 24 countries.
  - The rule is principled and was fixed before running, but it was chosen *after* seeing Luxembourg's problem, and across Sessions 3–5 many versions have been tried. Among that many tries, one small positive result can appear by chance.
  - The placebo above was run on raw trade openness, where the gain was not unusual. It has not been run on this version, so there is no evidence yet that the gain is about consolidations rather than about the controls predicting TFP growth in general.
- **Conclusion:** a small, promising result, not yet a finding. The next check is the placebo on the log version.

**Placebo test on the log version (added 10 Nov 2026)**
- The same two versions as for raw trade openness: 500 shuffles with the settings fixed at the section 7 choice (slightly optimistic, may favour the real episodes), and 100 shuffles with degree and λ chosen again in every round (no such caveat).

| | Original, fast | Original, clean | Year-adjusted, fast | Year-adjusted, clean |
|---|---|---|---|---|
| Shuffles | 500 | 100 | 500 | 100 |
| Real: log loss vs baseline | +8.8 % | +1.7 % | +7.8 % | +1.9 % |
| Shuffled: median (middle 90 %) | −1.9 % (−15.5 to +8.8) | −3.8 % (−20.8 to +6.2) | −7.1 % (−33.9 to +8.3) | −3.2 % (−14.9 to +6.9) |
| Shuffles at least as good (log loss) | 5 % | 16 % | 6 % | 21 % |
| Real: accuracy minus baseline | +16.7 points | +13.5 | +19.8 | +18.8 |
| Shuffles at least as good (accuracy) | 8 % | 9 % | 5 % | **2 %** |
| Countries won: real vs shuffled median | 16 vs 13 | 16 vs 14 | 12 vs 13 | 13 vs 14 |

- **The clean version decides it:** with the settings chosen in every round, 16 % and 21 % of shuffled panels do at least as well on log loss. The small gain is not unusual, so it mostly comes from the controls predicting TFP growth in general, not from which years were consolidations.
- The fast version looks better (5–6 % of shuffles as good), but its settings were tuned on the real episodes, which favours them. That gap is exactly why the clean version is the one to trust.
- **The same accuracy pattern for the third time:** with the year-adjusted target and tuning in every round, only 2 % of shuffles get as many yes/no calls right (Session 3 placebo: 2–3 %; raw trade openness above: 2 %). It is one faint pattern seen three times on the same data, not three independent confirmations, and it shows up in which side of 50 % the predictions fall, not in how confident they are.
- **Conclusion:** the log rule fixes the Luxembourg problem and gives the first model that beats the guess in all three tests, but the placebo shows the gain is not about consolidations. Still no usable predictor of which consolidations are followed by weak growth.

**Main lessons**
- J_train and J_cv together show what a single score cannot: low J_train with high J_cv is overfitting; both stuck at the guess means nothing is being learned.
- The J_cv of the chosen model is optimistic, because it won a contest on the same folds (0.643 in cross-validation, 0.704 with every country hidden in turn).
- Learning curves answer "would more data help?". Here the regularized models meet at the guess, and the flexible ones would need far more countries than the EU has.
- A test set read in every session slowly stops being a test set.
- Unstable tuning is the weak point: in Sessions 3, 4 and 5 the worst round is Luxembourg, where tuning picks a less regularized model and is badly wrong.

- A placebo is what turns "it beats the guess" into "it beats the guess *because of the consolidations*". Here the first win in all three tests did not pass it.
- A placebo with tuning inside every round needs no caveat; it costs time (here 100 shuffles in parallel, about 10 minutes), but it is the clean version.

- Fix an outlier's problem with a rule for every country, not by dropping the country. Here the log of trade openness turned −32.4 % into +1.7 %.

**Next:** Part B after Course 2 Week 4: decision trees, to test whether there is a large-consolidation threshold, using the log of trade openness and the clean placebo from the start.
