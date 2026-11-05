# Learning log, Session 3 (5 Nov 2026)

**Course:** Machine Learning Specialization, Course 1 Week 3 (classification, sigmoid, decision boundary, logistic cost, gradient descent for logistic regression, overfitting, regularization)

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
- λ ∈ {0, 0.1, 1, 10, 100}, chosen by leaving one training country out at a time, without the hidden countries. On the Session 1 training split, λ = 10 wins (log loss 0.680, against 0.741 with no regularization). Regularization helps, which fits a small sample with weak signal.

**Prediction: a small, unsteady gain**

| Model | Six hidden: log loss vs baseline | Six hidden: accuracy vs baseline | Each hidden in turn: log loss vs baseline | Each hidden in turn: accuracy vs baseline | Countries where log loss beats baseline |
|---|---|---|---|---|---|
| **ΔCAPB + controls** | **0.670 vs 0.710 (+5.6 %)** | **60 % vs 40 %** | **0.647 vs 0.694 (+6.8 %)** | **64.6 % vs 49.0 %** | **17 of 24** |
| ΔCAPB only | 0.693 vs 0.710 (+2.3 %) | 44 % vs 40 % | 0.664 vs 0.694 (+4.3 %) | 59.4 % vs 49.0 % | 13 of 24 |
| Controls only | 0.670 vs 0.710 (+5.6 %) | 68 % vs 40 % | 0.640 vs 0.694 (+7.8 %) | 61.5 % vs 49.0 % | 14 of 24 |

- Unlike Sessions 1 and 2, the model does better than the baseline in the fairer test too, not only on one split. But the gain is small.
- Over 500 random sets of six hidden countries, log loss beats the baseline in 60 % of splits (median +2.3 %), and accuracy in 74 % (median +10 points).
- The baseline is weak here, because the classes are almost 50/50 and the majority class flips between rounds. Always guessing "weak" on all 96 episodes would be right 51 % of the time.
- Results vary a lot by country: Luxembourg and Latvia go wrong, Italy and Romania go well.
- Without the inflation cap, the results barely change (each hidden in turn: +7.6 %, accuracy 65.6 %), so the cap was not decisive.

**What the model says**
- Logit on all 96 episodes, country-clustered standard errors: only lagged inflation is significant at 5 % (0.081, p = 0.049). ΔCAPB is positive but not significant (0.108, p = 0.199).
- Controls alone predict as well as controls plus ΔCAPB. So the size of the consolidation adds little; the economic conditions around it matter more.
- Timing matters: 64 % of episodes in 2008–2013 were followed by weak growth, against 41 % in 1997–2007 and 48 % in 2014–2022. Because "weak" compares each episode with all years, part of what the model picks up is the period, not the consolidation.

**Year-adjusted check (added 5 Nov 2026)**
- Target redefined: 1 if the episode's TFP growth is below the median of all training-country observations in the **same year**. The yearly medians are recalculated without the hidden countries each time (they range from −4.24 % in 2007 to 6.64 % in 1998). Everything else is identical: the same 96 episodes, inputs, ±30 % inflation cap, α rule (same α for every λ as before), λ chosen inside the training countries (λ = 10 again), majority-class baseline, and all three tests.

| Test | Original: median of all years | Year-adjusted: median of the same year |
|---|---|---|
| Share of the 96 episodes that are weak | 51 % | 52 % |
| Six hidden: log loss vs baseline | 0.670 vs 0.710 (+5.6 %) | 0.674 vs 0.718 (+6.1 %) |
| Six hidden: accuracy vs baseline | 60 % vs 40 % | 56 % vs 40 % |
| Each hidden in turn: log loss vs baseline | 0.647 vs 0.694 (+6.8 %) | 0.638 vs 0.700 (+8.8 %) |
| Each hidden in turn: accuracy vs baseline | 64.6 % vs 49.0 % | 60.4 % vs 42.7 % |
| Each hidden in turn: countries where log loss beats baseline | 17 of 24 | 13 of 24 |
| 500 random splits: log loss better than baseline | 60 % of splits (median +2.3 %) | 52 % of splits (median +0.2 %) |
| 500 random splits: accuracy better than baseline | 74 % of splits (median +10 points) | 72 % of splits (median +10 points) |

- The timing pattern is gone: the share of weak episodes is now 56 % in 1997–2007, 52 % in 2008–2013 and 48 % in 2014–2022.
- The pooled log loss still looks better than the baseline, but the steadiness tests do not hold up: the model wins in only 13 of 24 countries and in 52 % of random splits, with a median gain of +0.2 %. That is a coin flip.
- The accuracy gains look steadier, but the baseline sets a low bar. With classes near 50/50, the majority class flips between rounds and its accuracy falls below 50 % (42.7 % with each country hidden in turn). Against a fair coin (50 %), the model's 60 % on 96 episodes is only about two standard errors better.
- Logit on all 96 episodes: no input is significant at 5 % any more. Inflation is the closest (p = 0.092); ΔCAPB has p = 0.366.
- **Conclusion:** most of the Session 3 signal was timing. Once each episode is compared with its own year, what is left is weak and unsteady, and ML still adds nothing reliable over the econometrics.

**Pre-trend check (added 5 Nov 2026), building on the year-adjusted version**
- One extra input: TFP growth before the episode, 100 × (ln TFP[t−1] − ln TFP[t−3]). It ends where the target window starts, so the two do not overlap. It is year-adjusted like the target: relative to the median of training-country observations in the same year, recalculated without the hidden countries each time.
- **5 of the 96 episodes are lost** because TFP at t−3 is missing: Austria, Bulgaria, Italy, Romania and Slovakia, all in 1997 (the panel starts in 1995). 91 episodes in 24 countries remain. All three models use these 91, so they can be compared directly. Everything else is as in the year-adjusted check (α the same for every λ).

| Test | Section 11 reference: year-adjusted original, 96 episodes | Year-adjusted original, 91 episodes | Pre-trend only, 91 episodes | Year-adjusted original + pre-trend, 91 episodes |
|---|---|---|---|---|
| λ chosen | 10 | 10 | 0.1 | 10 |
| Six hidden: log loss vs baseline | 0.674 vs 0.718 | 0.663 vs 0.702 (+5.5 %) | 0.668 vs 0.702 (+4.8 %) | 0.658 vs 0.702 (+6.2 %) |
| Six hidden: accuracy vs baseline | 56 % vs 40 % | 58 % vs 42 % | 54 % vs 42 % | 58 % vs 42 % |
| Each hidden in turn: log loss vs baseline | 0.638 vs 0.700 (+8.8 %) | 0.671 vs 0.700 (+4.3 %) | 0.668 vs 0.700 (+4.7 %) | 0.670 vs 0.700 (+4.3 %) |
| Each hidden in turn: accuracy vs baseline | 60.4 % vs 42.7 % | 60.4 % vs 48.4 % | 52.7 % vs 48.4 % | 56.0 % vs 48.4 % |
| Countries where log loss beats baseline | 13 of 24 | 12 of 24 | 15 of 24 | 12 of 24 |
| 500 random splits: log loss better | 52 % (median +0.2 %) | 38 % (median −2.1 %) | 67 % (median +2.4 %) | 41 % (median −1.4 %) |
| 500 random splits: accuracy better | 72 % (median +10.0 points) | 62 % (median +5.6 points) | 69 % (median +8.7 points) | 63 % (median +5.4 points) |

- **Dropping five 1997 episodes alone moves the results a lot.** The year-adjusted original goes from +8.8 % to +4.3 % with each country hidden in turn, and from 52 % to 38 % of random splits. The earlier pooled gain was fragile.
- **The pre-trend alone is the steadiest of the three, but small.** It beats the baseline in 15 of 24 countries and in 67 % of random splits, with a median gain of +2.4 %. Its accuracy (52.7 % with each country hidden in turn) is barely above a coin flip.
- **Adding it to the other inputs does not help.** The combined model wins in 12 of 24 countries and 41 % of random splits, about the same as without it. With only 91 episodes, the extra inputs cost more than they add.
- **Direction: persistence, not a bounce back.** Faster TFP growth before the episode goes with a lower chance of weak growth after (logit coefficient −0.052, p = 0.483, not significant). By thirds: 53 % weak after the slowest third, 60 % after the middle third, and 39 % after the fastest third, so the pattern is not even smooth. The shared t−1 endpoint would, if anything, push the other way (a blip at t−1 raises one window and lowers the other), so the mechanical overlap does not explain this sign.
- No input is significant at 5 % in the combined logit (inflation p = 0.533, ΔCAPB p = 0.243).
- **Conclusion:** past TFP growth carries a faint signal of its own, in line with the thesis caveat that countries diverge before they consolidate. But it is not significant, and it does not turn the model into a reliable predictor. ML still adds nothing reliable over the econometrics.

**Main lessons**
- Accuracy and log loss can tell different stories: log loss punishes confident mistakes.
- Regularization (λ = 10) helped in a small, noisy sample.
- Choosing α depends on λ. An α that works without regularization can blow up with a large λ.
- A predictor can work for the wrong reason. Here it partly recognised the crisis years; the year-adjusted check removed most of the gain. The thesis pre-trend caveat still applies, so this is not evidence about which consolidations *cause* weaker growth.
- A baseline that is wrong more than half the time is easy to beat. Check the accuracy against 50 % as well.
- Small samples are fragile: removing five episodes halved the pooled gain. Read the steadiness tests (countries won, random splits), not one pooled number.

**Next:** Course 1 is done. Course 2 (neural networks, decision trees) can be tested the same way, with the year-adjusted target and these honest baselines.
