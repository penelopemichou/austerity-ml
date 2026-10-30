# Learning log, Session 1 (30 Oct 2026)

**Course:** Machine Learning Specialization, Course 1 Week 1 (linear regression with one input, cost function, gradient descent)

- Gradient descent finds the same line as OLS (w = −0.198). Fixed effects make it −0.268.
- Significant ≠ predictive: within-country R² is 0.022, and on 6 unseen countries the model beats the mean by only 1.2 %.
- A straight line gives −0.89 % per episode vs −1.50 % in my thesis LP: a threshold effect gets diluted by a linear model.
- Data note: Ireland 2010 (−18.8 pp) and 2011 (+18.2 pp) are bank-rescue accounting, not fiscal policy. Extreme, but they sit near the line, so they barely move the slope.

**Next:** Course 1 Week 2 (multiple features, feature scaling, vectorization). Add the four thesis controls and the episode dummy as inputs and re-run the unseen-country test.
