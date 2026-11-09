# Learning log, Session 3 (5 Nov 2026, corrected 9 Nov 2026)

**Course:** Machine Learning Specialization, Course 1 Week 3 (classification, sigmoid, decision boundary, logistic cost, gradient descent for logistic regression, overfitting, regularization)

> **Correction, 9 Nov 2026.** The code that chose λ had a leak. Inside the loop that leaves one training country out at a time, it also scored the hidden countries' episodes, so the hidden countries helped choose λ. That made every result that chooses λ look better than it was. The code is fixed, every number below comes from the fixed code, and the main conclusion changes: the model does **not** beat the simple guess in the fairer test. Results that do not choose λ (the logits, the timing and pre-trend tables, the check against scikit-learn) are unchanged.

**Setup, decided before fitting:**
- Rows: the 96 consolidation episodes (ΔCAPB ≥ 1.5 pp) in Session 2's rows, from 24 countries. Some countries have only one or two episodes.
- Target: 1 if TFP growth from t−1 to t+2 is below the median of all country-years in the training countries. The median is recomputed without the hidden countries every time. 51 % of all episodes are "weak".
- Inputs: ΔCAPB and the four lagged controls. The episode flag is always 1 here, so it is dropped.
- Inflation is capped at ±30 %, as Session 2 recommended deciding up front.
- Baseline: always predict the most common training class. For log loss it uses the training share of "weak" as its probability.

**Method**
- Vectorized gradient descent for logistic regression.
- With λ = 0 it matches scikit-learn and statsmodels to 4 decimals (largest gap 3.5e-05). With λ = 1 it matches scikit-learn with C = 1/λ (gap 3.3e-04).
- α was chosen as in Session 2: values about ×3 apart, on training data only. α = 3 for λ up to 10, but λ = 100 needs α = 1. My first run reused one α for every λ, and gradient descent blew up at λ = 100, so α is now picked for each λ.
- λ ∈ {0, 0.1, 1, 10, 100}, chosen by leaving one training country out at a time and scoring only that country. On the Session 1 training split, λ = 100 wins (log loss 0.702, against 0.960 with no regularization). The strongest penalty wins: the weights shrink until the model is close to the simple guess.

**Prediction: no gain in the fairer test**

| Model | Six hidden: log loss vs baseline | Six hidden: accuracy vs baseline | Each hidden in turn: log loss vs baseline | Each hidden in turn: accuracy vs baseline | Countries where log loss beats baseline |
|---|---|---|---|---|---|
| **ΔCAPB + controls** | **0.695 vs 0.710 (+2.1 %)** | **40 % vs 40 %** | **0.730 vs 0.694 (−5.2 %)** | **63.5 % vs 49.0 %** | **17 of 24** |
| ΔCAPB only | 0.696 vs 0.710 (+1.9 %) | 40 % vs 40 % | 0.685 vs 0.694 (+1.3 %) | 57.3 % vs 49.0 % | 13 of 24 |
| Controls only | 0.698 vs 0.710 (+1.6 %) | 40 % vs 40 % | 0.734 vs 0.694 (−5.8 %) | 63.5 % vs 49.0 % | 14 of 24 |

- On the six hidden countries the model is a little better on log loss, but it makes the same yes/no calls as the baseline (40 % accuracy each).
- With every country hidden in turn, the pooled log loss is *worse* than the baseline (−5.2 %).
- After the fix, λ barely varies: 23 of the 24 rounds choose λ = 100 and only Luxembourg's round chooses λ = 10, but that one round is so wrong (log loss 1.705 against 0.674 for the baseline) that it turns the pooled result negative; the other 23 rounds together score 0.677 against 0.695.
- Over 500 random sets of six hidden countries (λ fixed at 100), log loss beats the baseline in 76 % of splits but only by a median of +1.6 %; accuracy wins in 60 % (median +4.8 points).
- The baseline is weak here, because the classes are almost 50/50 and the majority class flips between rounds. Always guessing "weak" on all 96 episodes would be right 51 % of the time.
- Without the inflation cap, the results barely change (each hidden in turn: −4.8 %, accuracy 62.5 %), so the cap was not decisive.

**What the model says**
- Logit on all 96 episodes, country-clustered standard errors: only lagged inflation is significant at 5 % (0.081, p = 0.049). ΔCAPB is positive but not significant (0.108, p = 0.199).
- Timing matters: 64 % of episodes in 2008–2013 were followed by weak growth, against 41 % in 1997–2007 and 48 % in 2014–2022. Because "weak" compares each episode with all years, part of any signal is the period, not the consolidation.

**Year-adjusted check (added 5 Nov 2026)**
- Target redefined: 1 if the episode's TFP growth is below the median of all training-country observations in the **same year**. The yearly medians are recalculated without the hidden countries each time (they range from −4.24 % in 2007 to 6.64 % in 1998). Everything else is identical, and λ = 100 is chosen again.

| Test | Original: median of all years | Year-adjusted: median of the same year |
|---|---|---|
| Share of the 96 episodes that are weak | 51 % | 52 % |
| Six hidden: log loss vs baseline | 0.695 vs 0.710 (+2.1 %) | 0.704 vs 0.718 (+2.0 %) |
| Six hidden: accuracy vs baseline | 40 % vs 40 % | 40 % vs 40 % |
| Each hidden in turn: log loss vs baseline | 0.730 vs 0.694 (−5.2 %) | 0.717 vs 0.700 (−2.5 %) |
| Each hidden in turn: accuracy vs baseline | 63.5 % vs 49.0 % | 55.2 % vs 42.7 % |
| Each hidden in turn: countries where log loss beats baseline | 17 of 24 | 13 of 24 |
| 500 random splits: log loss better than baseline | 76 % of splits (median +1.6 %) | 72 % of splits (median +0.9 %) |
| 500 random splits: accuracy better than baseline | 60 % of splits (median +4.8 points) | 54 % of splits (median +3.8 points) |

- The timing pattern is gone: the share of weak episodes is now 56 % in 1997–2007, 52 % in 2008–2013 and 48 % in 2014–2022.
- The year-adjusted model is also worse than the baseline with every country hidden in turn, and wins in only 13 of 24 countries.
- Logit on all 96 episodes: no input is significant at 5 % any more. Inflation is the closest (p = 0.092); ΔCAPB has p = 0.366.

**Pre-trend check (added 5 Nov 2026), building on the year-adjusted version**
- One extra input: TFP growth before the episode, 100 × (ln TFP[t−1] − ln TFP[t−3]). It ends where the target window starts, so the two do not overlap. It is year-adjusted like the target.
- **5 of the 96 episodes are lost** because TFP at t−3 is missing: Austria, Bulgaria, Italy, Romania and Slovakia, all in 1997 (the panel starts in 1995). 91 episodes in 24 countries remain, and all three models use these 91.

| Test | Year-adjusted original, 91 episodes | Pre-trend only, 91 episodes | Year-adjusted original + pre-trend, 91 episodes |
|---|---|---|---|
| λ chosen | 100 | 10 | 10 |
| Six hidden: log loss vs baseline | 0.689 vs 0.702 (+1.8 %) | 0.678 vs 0.702 (+3.4 %) | 0.658 vs 0.702 (+6.2 %) |
| Six hidden: accuracy vs baseline | 42 % vs 42 % | 54 % vs 42 % | 58 % vs 42 % |
| Each hidden in turn: log loss vs baseline | 0.722 vs 0.700 (−3.1 %) | 0.697 vs 0.700 (+0.5 %) | 0.717 vs 0.700 (−2.3 %) |
| Each hidden in turn: accuracy vs baseline | 53.8 % vs 48.4 % | 51.6 % vs 48.4 % | 50.5 % vs 48.4 % |
| Countries where log loss beats baseline | 11 of 24 | 15 of 24 | 12 of 24 |
| 500 random splits: log loss better | 55 % (median +0.2 %) | 77 % (median +2.1 %) | 41 % (median −1.4 %) |
| 500 random splits: accuracy better | 45 % (median +0.0 points) | 65 % (median +5.3 points) | 63 % (median +5.4 points) |

- The pre-trend alone is the only model here that is not worse than the baseline with every country hidden in turn, but its gain is tiny (+0.5 %) and its accuracy is about a coin flip.
- Adding it to the other inputs does not help: the combined model looks good on the six hidden countries, but it loses with every country hidden in turn and in most random splits.
- Direction: faster TFP growth before the episode goes with a lower chance of weak growth after (logit coefficient −0.052, p = 0.483, not significant). By thirds: 53 % weak after the slowest third, 60 % after the middle third, and 39 % after the fastest third.

**Placebo test (added 9 Nov 2026)**
- Within each country, the ΔCAPB values were shuffled across years, 500 times. Each country keeps its number of episodes and their sizes; only *which years* count as consolidations becomes random (in a typical shuffle, 23 of the 96 real episodes happen to stay in their own year). The model was rerun on each shuffled panel with every country hidden in turn, λ fixed at 100 as chosen on the training countries. The real result was recomputed the same way.

| | Original target | Year-adjusted target |
|---|---|---|
| Real episodes: log loss vs baseline | +0.9 % | +1.4 % |
| Shuffled: median (middle 90 %) | −0.4 % (−2.5 to +2.2) | −0.8 % (−2.8 to +1.8) |
| Shuffles at least as good on log loss | 21 % | 8 % |
| Real episodes: accuracy minus baseline | +14.5 points | +12.5 points |
| Shuffled: median (middle 90 %) | +0.0 (−3.2 to +9.4) | +0.0 (−3.2 to +9.3) |
| Shuffles at least as good on accuracy | 2 % | 3 % |
| Countries won: real vs shuffled median | 17 vs 14 of 24 | 13 vs 14 of 24 |

- On log loss, the real episodes are not unusual: 21 % and 8 % of the fake versions do at least as well.
- On accuracy, they are: only 2–3 % of the fake versions get the yes/no calls right as often. With λ = 100 the model's chances stay close to 50 %, so its calls depend on small tilts. For the real episodes those tilts point the right way more often than for random years.
- So the real consolidation years carry a faint pattern that random years do not. It is far too weak to predict with (the model is barely more confident than a coin), it shows up in one measure but not the other, and four comparisons were made, so one or two small shares can happen by chance.
- It is also not causal. Countries choose when to consolidate, and the thesis pre-trend shows they were already diverging, so the pattern can come from what goes with consolidating, not from the consolidation itself.
- **Caveat:** the placebo uses λ = 100 fixed in every round, chosen on the real episodes' training countries. That makes it slightly optimistic: those countries are also hidden in turn here, so λ was not chosen fully out of sample. It may also slightly favour the real episodes, because the shuffled versions do not get to choose their own λ. A fully clean version would choose λ again inside every round for every shuffle (about 7 hours). It was not run, because the honest headline (−5.2 %, with λ chosen inside each round) already shows no usable predictor.

**Main lessons**
- Leakage can hide in the tuning step. Choosing λ looked "inside the training countries", but one line let the hidden countries in, and it turned a loss into an apparent win. A result that beats every fixed setting is a warning sign.
- Accuracy and log loss can tell different stories: log loss punishes confident mistakes.
- Strong regularization (λ = 100) won once the leak was fixed: with this little signal, the best model is one that stays close to the simple guess.
- Choosing α depends on λ. An α that works without regularization can blow up with a large λ.
- A baseline that is wrong more than half the time is easy to beat. Check the accuracy against 50 % as well. A placebo test handles this automatically, because the fake versions face the same baseline.
- Small samples are fragile. Read the steadiness tests (countries won, random splits), not one pooled number.

**Next:** Course 1 is done. Course 2 (neural networks, decision trees) can be tested the same way, with the year-adjusted target, these honest baselines and a placebo check.
