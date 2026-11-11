# austerity-ml

Learning machine learning step by step on a real research dataset: the EU country panel from my MSc thesis at TU Delft (*Austerity and Europe's Structural Trajectory*, 2026). I am working through Andrew Ng's Machine Learning Specialization alongside it, and every notebook applies what that week of the course covers.

## The question

Can a simple ML model predict which fiscal consolidations are followed by weaker long-run growth? And is a good predictor the same thing as a cause?

My thesis answers the causal side with econometrics (local projections with country and year fixed effects). This repo asks what machine learning adds to that, and says so plainly when the answer is "nothing".

## Data

Annual panel of 26 EU member states (EU minus Czech Republic), 1995–2024.

- Fiscal shock: change in the cyclically adjusted primary balance (ΔCAPB, pp of GDP); a consolidation *episode* is ΔCAPB ≥ 1.5 pp
- Outcomes: TFP, output per worker, public and private investment, unemployment, real compensation, R&D intensity, exports
- Sources: AMECO (European Commission), Eurostat, SWIID, DG ECFIN

## How I evaluate every model

- Compare against a simple baseline ("always predict the average")
- Test on countries the model has never seen (split by country, never by random row, to avoid leakage from overlapping time windows)
- Check for overfitting
- Compare with what the fixed-effects econometrics already says

## Sessions

A plain-language overview with a roadmap and the results so far is in [PROGRESS.md](PROGRESS.md).

| # | Notebook | Course link | What I found |
|---|---|---|---|
| 1 | `ML01_dCAPB_to_TFP_linear_regression` | C1 Week 1: linear regression, cost function, gradient descent | See below |
| 2 | `ML02_multiple_inputs_vectorized_gradient_descent` | C1 Week 2: multiple features, vectorization, feature scaling, learning rate | See below |
| 3 | `ML03_logistic_regression_episode_classifier` | C1 Week 3: logistic regression, decision boundary, regularization | See below |
| 3b | `ML03b_logistic_regression_practice` | C1 Week 3: practice exercises in the style of the course labs | Exercises with PASS/FAIL tests; solutions in `notes/ML03b_solutions.py` |
| 3b | `ML03b_logistic_regression_practice_completed` | C1 Week 3: my own answers to the practice exercises | All 25 checks pass |
| 4 | `ML04_neural_network_episode_classifier` | C2 Weeks 1–2: neural networks, TensorFlow, ReLU, from_logits, Adam, softmax | See below |
| 5a | `ML05a_evaluating_models_bias_variance` | C2 Week 3: train / cross-validation / test, bias and variance, learning curves | See below |

### Session 1: one-variable linear regression (ΔCAPB → TFP growth, t−1 to t+2)

- Gradient descent and OLS give the same line (slope −0.198 per pp of tightening).
- With country and year fixed effects the slope becomes −0.268: cross-country differences were masking part of the relationship.
- Scaled to an average episode (3.3 pp) that is about −0.9 % TFP, versus about −1.5 % in my thesis local projection. A straight line dilutes what looks more like a threshold effect.
- **Significant is not the same as predictive.** Within-country R² is 0.022. On six held-out countries the model beats "predict the average" by only 1.2 %, and across all 26 leave-one-country-out splits it wins in 13: a coin flip.
- Verdict: ML adds nothing over the econometrics yet. A real effect can still be a useless predictor.

### Session 2: several inputs (ΔCAPB + four lagged controls + episode flag)

- Vectorized gradient descent matches scikit-learn. Z-score scaling uses the training countries only, and α = 1.0 was chosen on training data from values about ×3 apart.
- On the same six hidden countries the model beats "predict the average" by 8.6 %, against 1.1 % for Session 1's model on the same rows. **That is a lucky split:** with each country hidden in turn it wins 13 of 26 and is 48.7 % *worse* overall, mostly because of Bulgaria's 1997 hyperinflation, and still 7.4 % worse with inflation capped. Across 500 random splits it beats the average only 38 % of the time.
- The effect per average episode moves close to the thesis: −1.38 %, against −0.93 % for ΔCAPB alone on the same rows. Flag + controls, the thesis design, gives −1.52 % (thesis: −1.50 %).
- Adding ΔCAPB² does not catch the threshold in a useful way. It fits better within the sample, but much of that comes from Ireland 2010–11 (t falls from −7.1 to −1.6), a parabola wrongly says large loosening hurts too, and prediction does not improve.
- **Data correction:** this session exposed two wrongly built controls, trade openness and inflation. They are now fixed. Re-running my thesis code on the corrected panel leaves the main findings in place (TFP at h = 2: −1.50 % becomes −1.53 %). See [the correction note](notes/controls_correction.md).
- Verdict: ML still adds nothing reliable over the econometrics.

### Session 3: a yes/no question (is a consolidation followed by below-median TFP growth?)

- **Correction, 9 Nov 2026:** the first version had a leak. The loop that chose λ also scored the hidden countries, so they helped pick λ. The numbers below come from the fixed code, and they reverse the earlier "small gain" result.
- Logistic regression on the 96 consolidation episodes, with ΔCAPB and the four lagged controls. Gradient descent matches scikit-learn and statsmodels. λ = 100, the strongest penalty tried, is chosen inside the training countries.
- On the six hidden countries the log loss is slightly lower than "always predict the most common class" (0.695 vs 0.710), but the yes/no calls are the same (accuracy 40 % vs 40 %).
- **It does not hold up in the fairer test:** with each country hidden in turn, log loss is *worse* than the baseline (0.730 vs 0.694, −5.2 %). Accuracy is 63.5 % vs 49 %, but that baseline is wrong more than half the time. That −5.2 % hinges on one round: λ is re-chosen in each round, 23 of 24 rounds pick λ = 100, and only Luxembourg's round picks λ = 10 and is badly wrong (log loss 1.705 vs 0.674). The other 23 rounds together are slightly better than the baseline (0.677 vs 0.695), a gain too small to be useful. Across 500 random splits the log loss is better in 76 % of them, by a median of only +1.6 %.
- Only lagged inflation is significant in a logit on all episodes (p = 0.049). Timing also matters: 64 % of episodes in 2008–2013 were followed by weak growth, against 41 % in 1997–2007.
- **Year-adjusted check:** comparing each episode with the median of training countries *in the same year* removes the timing pattern. The model is still worse than the baseline with each country hidden in turn (0.717 vs 0.700) and wins in 13 of 24 countries. No input is significant at 5 %.
- **Pre-trend check:** adding TFP growth before the episode (t−3 to t−1, year-adjusted the same way) costs 5 episodes from 1997, leaving 91. The pre-trend alone just about matches the baseline (+0.5 %, 15 of 24 countries) and is not significant (p = 0.48). Adding it to the other inputs does not help (−2.3 %, 12 of 24).
- **Placebo test:** within each country, I shuffled which years count as consolidations 500 times and reran the model (each country hidden in turn, λ = 100). On log loss the real episodes are not unusual (21 % of shuffles do as well; 8 % with the year-adjusted target). On accuracy they are: only 2–3 % of shuffles match them (+14.5 and +12.5 points over the baseline). So the real consolidation years carry a faint pattern that random years do not, but it is far too weak to predict with, and it says nothing about cause.
- Verdict: ML still adds nothing reliable over the econometrics.

### Session 4: a small neural network on the same yes/no question

- TensorFlow/Keras: one ReLU hidden layer (2, 4 or 8 units), a linear output with `from_logits=True`, binary cross-entropy, Adam and L2. The size and λ are chosen on the training countries only (4 groups of training countries; only the left-out group is scored). Logistic regression runs side by side with the same tuning and reproduces the Session 3 results.
- Size: 15 / 29 / 57 parameters against 71 training episodes.
- **No gain in the fair tests.** With each country hidden in turn the network's log loss is worse than the baseline: 0.763 vs 0.694 (−10.0 %) on the original target and 0.713 vs 0.700 (−1.9 %) year-adjusted, against −5.2 % and −3.7 % for logistic regression. It wins in 14 and 12 of 24 countries, and in 71 % and 49 % of 500 random splits.
- **One lucky result:** on the six hidden countries with the year-adjusted target it scores 0.639 vs 0.718 and 72 % vs 40 % accuracy, but only with seed 0. Seeds 1–4 give 0.659 to 0.751, and the settings chosen change from round to round (8 different ones in 24 rounds).
- Optional three-class softmax version (weak / middle / strong, year-adjusted): no better than the most common class (each hidden in turn −6.1 %, accuracy 33 % vs 35 %).
- Verdict: a more flexible model does not find what a simple one could not. ML still adds nothing reliable over the econometrics.

### Session 5, Part A: evaluating models (bias, variance, learning curves)

- Logistic regression on polynomial features of the five inputs (degree 1–3: 5, 20 or 55 features), λ from 0.01 to 1000, J_train and J_cv from 4 groups of training countries, the simple guess as the baseline level.
- **Classic high variance with little regularization:** degree 3 with λ = 0.01 has J_train 0.12 and J_cv 2.9 (guess about 0.70). With more regularization both converge to the guess. Degree 1 never gets below it in cross-validation; degree 2 dips slightly (J_cv 0.680 vs 0.698 original, 0.643 vs 0.696 year-adjusted).
- **Learning curves:** for the regularized models, J_train and J_cv close in on the simple guess as countries are added (at 16 countries: J_cv 0.695 and 0.714 vs 0.694). For the flexible model the gap is huge and closes slowly. The limit looks like the information in these inputs, not model size or the amount of data.
- **The cross-validation choice (degree 2) through the three tests:** six hidden 0.667 vs 0.710 and 0.624 vs 0.718; random splits better in 77 % and 69 % (median +2.9 % and +3.5 %), the most frequent small gain so far. But each hidden in turn gives −32.4 % and −0.6 %, almost entirely from Luxembourg's round, where tuning picked a more flexible setting (log loss 5.91 vs 0.674 on the original target).
- The six hidden countries have been read in every session, so they are no longer a clean test set; the repeated tests matter more.
- **Placebo test** (episode years shuffled within each country; 500 shuffles with settings fixed, 100 with degree and λ chosen again in every round): on log loss the real episodes are not unusual (10–15 % of shuffles do as well; 94 % with the original target and tuning in every round). On accuracy, the year-adjusted target with tuning in every round stands out (2 % of shuffles match it), the same pattern as the Session 3 placebo. The degree-2 lead does not survive on log loss.
- **Why Luxembourg goes wrong:** its trade openness is 240–376 % of GDP against a median episode of 109 %, so when it is hidden the model extrapolates far outside its training data. Dropping it would choose the sample by the result; a rule for all countries (log of trade openness, or capping inputs at the training range) is the better fix.
- **Log of trade openness, as a rule for every country** (decided before running; Luxembourg kept): Luxembourg's round costs 6.6 points instead of 39.3, and the degree-2 model beats the simple guess in all three tests for the first time: each hidden in turn 0.682 vs 0.694 (+1.7 %) and 0.686 vs 0.700 (+1.9 %), random splits better in 84 % and 64 %. The gain is small, and the rule was chosen after seeing the problem.
- **Placebo on the log version:** with degree and λ chosen in every round, 16 % and 21 % of shuffled panels do at least as well on log loss, so the gain is not about the consolidation years. On accuracy the year-adjusted target again stands out (2 % of shuffles), the third time on the same data.
- **Error analysis and precision / recall** (after Course 2 Week 3), for the log version with every country hidden in turn: precision for "weak" 0.71 but recall only 0.44–0.45. The model is cautious about saying "weak" and gains mostly by recognising "not weak" episodes (recall 0.80–0.81). Luxembourg's episodes are 4 of its 5 worst misses on the original target (all weak, predicted 13–19 %), adding 4.40 of excess loss against 0.59 for the next country. Deep recessions (Latvia 2010, Estonia 2009) and high-debt episodes that were not weak (Greece 2005 and 2022) come next. On the original target the gain is again mostly the 2008–2013 crisis years.
- Verdict: the first model to beat the guess in all three tests, but not because of the consolidations. Still no usable predictor.

## Structure

```
notebooks/   one Colab notebook per session (ML01_, ML02_, ...), plus practice notebooks (ML03b_)
data/        analysis panel and source notes
notes/       learning logs
progress/    session plan, results and log (feed PROGRESS.md and my dashboard)
```

## Author

Pinelopi (Penelope) Michou · MSc Management of Technology, TU Delft

The code in this repo was written with help from [Claude Code](https://claude.com/claude-code).
