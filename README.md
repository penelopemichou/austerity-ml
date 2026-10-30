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

| # | Notebook | Course link | What I found |
|---|---|---|---|
| 1 | `ML01_dCAPB_to_TFP_linear_regression` | C1 Week 1: linear regression, cost function, gradient descent | See below |

### Session 1: one-variable linear regression (ΔCAPB → TFP growth, t−1 to t+2)

- Gradient descent and OLS give the same line (slope −0.198 per pp of tightening).
- With country and year fixed effects the slope becomes −0.268: cross-country differences were masking part of the relationship.
- Scaled to an average episode (3.3 pp) that is about −0.9 % TFP, versus about −1.5 % in my thesis local projection. A straight line dilutes what looks more like a threshold effect.
- **Significant is not the same as predictive.** Within-country R² is 0.022. On six held-out countries the model beats "predict the average" by only 1.2 %, and across all 26 leave-one-country-out splits it wins in 13: a coin flip.
- Verdict: ML adds nothing over the econometrics yet. A real effect can still be a useless predictor.

## Structure

```
notebooks/   one Colab notebook per session (ML01_, ML02_, ...)
data/        analysis panel and source notes
notes/       learning logs
```

## Author

Pinelopi (Penelope) Michou · MSc Management of Technology, TU Delft
