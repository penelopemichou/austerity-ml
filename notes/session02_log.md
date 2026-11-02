# Learning log, Session 2 (2 Nov 2026)

**Course:** Machine Learning Specialization, Course 1 Week 2 (multiple features, vectorization, feature scaling, choosing the learning rate, feature engineering)

**Setup:** ΔCAPB + four lagged thesis controls + episode flag (ΔCAPB ≥ 1.5 pp). 669 rows: the controls cost 22 rows, because inflation starts in 1996 and its one-year lag therefore starts in 1997. Same target, same six hidden countries and same baseline as Session 1.

**Data correction:** building this session exposed two wrongly built controls, trade openness and inflation. They are now fixed in `data/panel_master.csv`, and the numbers below use the corrected panel. Details: [controls_correction.md](controls_correction.md).

**Method**
- Z-score scaling uses the training countries only, and is redone inside every leave-one-out fold.
- Without scaling, α has to stay tiny and training is slow: after 5,000 steps the cost is 15.02, versus 13.67 with scaling.
- α values about ×3 apart, chosen on training data only: 1.0 is the largest whose cost falls at every step, and 3.0 blows up.
- Vectorized gradient descent matches scikit-learn's `LinearRegression` (largest gap 9e-16).

**Prediction: no reliable gain**

| Model | Six hidden countries: better than baseline | Each country hidden in turn: wins | Each country hidden in turn: overall |
|---|---|---|---|
| Session 1 model, on its own 691 rows | 1.2 % | 13 of 26 | +0.4 % |
| Session 1 model, on the same 669 rows | 1.1 % | 15 of 26 | +0.5 % |
| Controls only | 7.9 % | 13 of 26 | −46.7 % |
| ΔCAPB + controls | 8.7 % | 13 of 26 | −48.4 % |
| **Session 2: ΔCAPB + controls + flag** | **8.6 %** | **13 of 26** | **−48.7 %** |
| Session 2 + ΔCAPB² | 8.5 % | 13 of 26 | −49.3 % |
| Session 2, inflation capped at ±30 % | 7.1 % | 14 of 26 | −7.4 % |
| Session 2, before the data correction | 10.0 % | 16 of 26 | −1.1 % |

- The 8.6 % is a lucky split. Over 500 random sets of six hidden countries, the median is −7.2 %, the middle 90 % runs from −274 % to +11 %, and the model beats the baseline in only 38 % of them.
- The −48.7 % comes mostly from Bulgaria, whose 1997 hyperinflation (about 1,040 %) lies far outside what the model saw in training. Bulgaria's error is about 12 times the baseline's (−1,067 %). But it is not only Bulgaria: without it, the result is still −6.6 % overall (13 of 25 wins), and capping inflation gives −7.4 %.
- Session 1's simple model is the only one here that does not lose to the baseline overall (+0.5 %), and that is close to zero.
- Lesson: correcting the data made the prediction result look *worse*. That is what honest testing is for. A wrong input can look harmless by accident.

**Effect per episode: close to the thesis (−1.50 %)**

| Model, country and year effects | Effect per episode | Main coefficient |
|---|---|---|
| ΔCAPB only, same rows | −0.93 % | slope −0.281 (t = −3.0) |
| ΔCAPB + controls | −1.01 % | slope −0.307 (t = −3.9) |
| Session 2: ΔCAPB + controls + flag | −1.38 % | slope −0.255 (t = −3.7), flag −0.536 (t = −1.3) |
| Session 2 + ΔCAPB² | −1.49 % | ΔCAPB² −0.021 (t = −7.1) |
| Flag + controls (the thesis design) | −1.52 % | flag −1.518 (t = −3.3) |

The controls and the flag bring the estimate close to the thesis. The design check is one horizon on this sample, not a full replication. ΔCAPB and the flag partly measure the same thing, so the flag alone is not significant once ΔCAPB is in.

**Extra: ΔCAPB² catches the shape only partly, and does not help prediction**
- Within the sample, it looks useful: the coefficient has t = −7.1 and the flag shrinks to −0.214 (t = −0.5).
- But much of that comes from Ireland 2010–11: without those two years, ΔCAPB² has t = −1.6.
- The actual averages are flat for loosening and fall sharply only from about 3 pp of tightening (−0.19 % at 1.5–3 pp, −1.86 % at 3–5 pp, −4.23 % above 5 pp). A parabola wrongly says big loosening hurts too, and the 1.5 pp flag may sit too low to mark the real break.
- On unseen countries it is no better: 8.5 %, 13 of 26, −49.3 % overall.

**Main lessons**
- Check one test against others: a single split can make a model look much better than it is.
- Look at the inputs before trusting a model. A strange number led to a real error in the data.
- Controls improve the effect estimate, but not the prediction. A better estimate of a cause is not the same as a better predictor.
- ML still adds nothing reliable over the econometrics.

**Next:** Course 1 Week 3 (logistic regression): which episodes are followed by below-median TFP growth? Decide up front how to handle extreme inflation values.
