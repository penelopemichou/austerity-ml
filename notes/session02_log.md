# Learning log, Session 2 (2 Nov 2026)

**Course:** Machine Learning Specialization, Course 1 Week 2 (multiple features, vectorization, feature scaling, choosing the learning rate, feature engineering)

**Setup:** ΔCAPB + four lagged thesis controls + episode flag (ΔCAPB ≥ 1.5 pp). 669 rows: the controls cost 22 rows, because inflation starts in 1996 and its one-year lag therefore starts in 1997. Same target, same six hidden countries and same baseline as Session 1.

**Method**
- Z-score scaling uses the training countries only, and is redone inside every leave-one-out fold.
- Without scaling, α has to stay tiny and training is slow: after 5,000 steps the cost is 15.15, versus 13.57 with scaling.
- α values about ×3 apart, chosen on training data only: 1.0 is the largest whose cost falls at every step, and 3.0 blows up.
- Vectorized gradient descent matches scikit-learn's `LinearRegression` (largest gap 4e-15).

**Prediction: no reliable gain**

| Model | Six hidden countries: better than baseline | Each country hidden in turn: wins | Each country hidden in turn: overall |
|---|---|---|---|
| Session 1 model, on its own 691 rows | 1.2 % | 13 of 26 | +0.4 % |
| Session 1 model, on the same 669 rows | 1.1 % | 15 of 26 | +0.5 % |
| Controls only | 9.3 % | 17 of 26 | −1.1 % |
| ΔCAPB + controls | 10.1 % | 16 of 26 | −0.8 % |
| **Session 2: ΔCAPB + controls + flag** | **10.0 %** | **16 of 26** | **−1.1 %** |
| Session 2 + ΔCAPB² | 9.8 % | 16 of 26 | −1.6 % |

- The 10 % is mostly a lucky split. Over 500 random sets of six hidden countries, the median is +1.6 %, the middle 90 % runs from −24.7 % to +12.0 %, and the model beats the baseline in only 54 % of them.
- It helps some countries a lot (Belgium, Italy) and hurts others badly (Luxembourg, Denmark, Ireland, Sweden). Overall, those losses cancel the gains.
- The controls do the predicting. Adding ΔCAPB and the flag to them changes almost nothing.

**Effect per episode: closer to the thesis (−1.50 %)**
- On the same rows, the effect moves from −0.93 % (ΔCAPB only) to −1.36 % (Session 2). Session 1's −0.89 % was on its own rows.
- Flag + controls, the thesis design, gives −1.49 % (t = −3.4), almost exactly the thesis number. This is not a full replication: it is one horizon, on this sample.
- In the Session 2 model, ΔCAPB stays significant (−0.233, t = −3.2) while the flag does not (−0.593, t = −1.4). ΔCAPB and the flag partly measure the same thing.

**Extra: ΔCAPB² catches the shape only partly, and does not help prediction**
- Within the sample, it looks useful: ΔCAPB² = −0.020 (t = −5.4), the flag shrinks to −0.283 (t = −0.6), and the effect becomes −1.46 %.
- But much of that comes from Ireland 2010–11: without those two years, ΔCAPB² has t = −1.9.
- The actual averages are flat for loosening and fall sharply only from about 3 pp of tightening (−0.19 % at 1.5–3 pp, −1.86 % at 3–5 pp, −4.23 % above 5 pp). A parabola wrongly says big loosening hurts too, and the 1.5 pp flag may sit too low to mark the real break.
- On unseen countries it is no better: 9.8 %, 16 of 26, −1.6 % overall.

**Data problem found: check this in the thesis too**
- `trade_open` is far too low in five non-euro countries (median: Hungary 0.4, Sweden 7.7, Romania 11.2, Denmark 11.7, Poland 17.8). The values look too small by about the exchange rate, which would fit euro trade divided by national-currency GDP.
- `inflation` also looks off in non-euro countries, for example Poland 2009 at −16.5 %, possibly because it reflects exchange-rate moves.
- This is a pattern in the numbers. I have not confirmed the cause or changed the data.
- Without these two controls, the Session 2 model goes from −1.1 % to +2.7 % overall (15 of 26 wins) on the hide-each-country test, and its effect barely moves (−1.35 %). The thesis-design estimate barely moves either (−1.48 %). Because I chose this test after seeing the results, the +2.7 % is a warning about the data, not a better model.

**Main lessons**
- Check one test against others: a single split can make a model look better than it is.
- Controls improve the effect estimate, but not reliably the prediction. A better estimate of a cause is not the same as a better predictor.
- ML still adds nothing reliable over the econometrics.

**Next:** check `trade_open` and `inflation` at the source, then Course 1 Week 3 (logistic regression): which episodes are followed by below-median TFP growth?
