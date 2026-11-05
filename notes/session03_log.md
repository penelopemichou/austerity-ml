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

**Main lessons**
- Accuracy and log loss can tell different stories: log loss punishes confident mistakes.
- Regularization (λ = 10) helped in a small, noisy sample.
- Choosing α depends on λ. An α that works without regularization can blow up with a large λ.
- A predictor can work for the wrong reason. Here it partly recognises the crisis years, and the thesis pre-trend caveat still applies, so this is not evidence about which consolidations *cause* weaker growth.

**Next ideas:** compare each episode with other countries in the same year (removing timing), or add TFP growth before the episode to test the pre-trend.
