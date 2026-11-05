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

- Logistic regression on the 96 consolidation episodes, with ΔCAPB and the four lagged controls. Gradient descent matches scikit-learn and statsmodels. λ = 10 was chosen inside the training countries only.
- On the six hidden countries the model beats "always predict the most common class": log loss 0.670 vs 0.710, accuracy 60 % vs 40 %.
- **This time it holds up in the fairer test, but only just:** with each country hidden in turn, log loss is 0.647 vs 0.694 (+6.8 %) and accuracy 64.6 % vs 49 %. Across 500 random splits it beats the baseline on log loss in 60 % of them.
- ΔCAPB adds little: controls alone do as well. Only lagged inflation is significant (p = 0.049).
- Part of the signal is timing: 64 % of episodes in 2008–2013 were followed by weak growth, against 41 % in 1997–2007. So this is a weak predictor, not evidence about which consolidations *cause* weaker growth.
- **Year-adjusted check:** comparing each episode with the median of training countries *in the same year* removes the timing pattern. The pooled log loss still looks better (0.638 vs 0.700), but the model now wins in only 13 of 24 countries and in 52 % of 500 random splits (median gain +0.2 %). No input is significant at 5 %. Most of the Session 3 signal was timing.
- **Pre-trend check:** adding TFP growth before the episode (t−3 to t−1, year-adjusted the same way) costs 5 episodes from 1997, leaving 91. On those, the pre-trend alone is the steadiest model (15 of 24 countries, 67 % of random splits), but its gain is small (median +2.4 %) and not significant (p = 0.48). Faster growth before goes with less weak growth after. Adding it to the other inputs does not help (12 of 24, 41 % of splits).
- Verdict: ML still adds nothing reliable over the econometrics.

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
