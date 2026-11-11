# CLAUDE.md — austerity-ml

## What this repo is

My machine-learning project on the data from my MSc thesis (TU Delft, "Austerity and Europe's Structural Trajectory", 2026): EU fiscal consolidations and what happens to growth afterwards. There is one session per week of Andrew Ng's Machine Learning Specialization, and each session applies that week to the thesis panel.

The question: can a simple model predict which consolidations are followed by weaker growth? And is a good predictor the same as a cause?

You (Claude) write the code. I'm a beginner in Python (comfortable with statistics), so explain simply and step by step:
- Link each step to what I just learned in the course.
- Before I run something, give me a predictions cell with a few questions, so I predict the result first.
- I work in Google Colab. Notebooks load the data from the repo's raw GitHub URL, so "Run all" works for anyone.
- Say plainly when ML adds nothing over my econometrics.

## Data

- `data/panel_master.csv`: 26 EU countries (EU minus Czech Republic), 1995–2024. Definitions are in `data/DATA_DICTIONARY.csv` and sources and licences in `data/README.md`.
- A consolidation episode is ΔCAPB ≥ 1.5 pp of GDP (`dcapb_main`; + = tightening).
- The main target is TFP growth from t−1 to t+2: 100 × (ln TFP[t+2] − ln TFP[t−1]), built within country.
- Controls: GDP growth, debt, inflation and trade openness, each lagged one year.
- Trade openness and inflation were corrected on 2 Nov 2026; see `notes/controls_correction.md`. The thesis itself used the old panel (commit `5787f6a`).
- Watch out for:
  - Ireland 2010–11 (ΔCAPB −18.8 / +18.2) is bank-rescue accounting, not fiscal policy.
  - Cyprus 2013 is +12.0.
  - Bulgaria's 1997 inflation is about 1,040 %. Session 3 caps inflation at ±30 %, decided before fitting.

## Rules we've used so far

Taken from PROGRESS.md and the session logs in `notes/`.

**Testing on unseen countries**
- Split by country, never by random row. The targets use overlapping time windows, so a random split leaks.
- Run every model through three tests:
  - **Six hidden countries:** Belgium, Greece, Lithuania, Malta, Slovakia and Spain (seed 42, the same in every session).
  - **Each country hidden in turn:** report the pooled result *and* how many countries the model wins.
  - **500 random sets of six hidden countries:** shows how lucky one split was.
- Read the steadiness tests (countries won, share of random splits), not one pooled number. Small samples are fragile.
- Where it helps, add a placebo test: shuffle which years count as episodes within each country, rerun, and see where the real result falls among the shuffled ones.

**Using only the training countries**
- Fit scaling (z-scores) on the training countries only, then apply it to the hidden ones.
- Compute medians that define a target (for example "below-median growth") on the training countries only, again in every round.
- Choose α and λ with the training countries only. Pick α from values about ×3 apart: the largest whose cost falls at every step, with a separate α for each λ. Choose λ by leaving one *training* country out at a time. That inner loop must score only the left-out training country, never the hidden ones (getting this wrong inflated the first Session 3 results). A tuned result that beats every fixed setting is a warning sign of leakage.

**Comparing with a simple baseline**
- Numbers: always predict the training mean.
- Yes/no: always predict the most common training class. For log loss, use the training share of "yes" as the probability.
- When classes are near 50/50, the majority-class baseline can be right less than half the time. Check accuracy against 50 % as well.

**Checking against my thesis method**
- The thesis method: Jordà local projections, h = −2…+4, country and year fixed effects, the four controls lagged one year, country-clustered standard errors. Code and results live in the thesis folder, outside this repo. Never modify it.
- Check each model's effect estimate against the thesis (TFP at h = 2: −1.50 % in the thesis, −1.53 % on the corrected panel).
- Check my own gradient descent against scikit-learn (and statsmodels where useful) before trusting it.
- Timing can fake a signal. Compare each episode with other countries in the same year (the year-adjusted target) before claiming the model learned anything about consolidations.

**After each session:** update the notebook (`notebooks/MLxx_...`), `notes/sessionNN_log.md`, `README.md`, `PROGRESS.md` (plain words, new terms explained in "Words used here") and `progress/*.json`, then commit and push.

## Where I am in my learning

- ML Specialization: Course 1 done (5 Nov 2026). Course 2 Weeks 1–2 done (9 Nov): neural networks, forward propagation, TensorFlow (Sequential, compile, fit), ReLU / sigmoid / linear / softmax, from_logits, Adam, backprop. Course 2 Week 3 Parts A–B done (10 Nov): train / cross-validation / test sets, model selection, bias and variance, baselines, learning curves. Course 2 Week 3 done (11 Nov): also error analysis, adding data, transfer learning, the ML project cycle, fairness, and precision / recall / F1 for skewed classes. Next: Week 4 (decision trees).
- Causal Inference: The Mixtape: Chapters 1–3 done (potential outcomes, selection bias, ATE / ATT / ATU, randomization, SUTVA, randomization inference).
- Use only methods I've covered so far. When a session needs something new, explain it simply first.

## From my learning notes

I'll paste new lines under this heading over time.

- Prediction isn't cause: never call a model's prediction a causal effect.
- My thesis estimates are closest to an ATT (the effect for the consolidations that happened), not an ATE.
- Countries choose austerity (selection bias), consolidations spill over between EU countries, and the ΔCAPB ≥ 1.5 pp flag mixes tax rises and spending cuts.
- With a few hundred country-years, prefer simple models (logistic regression, regularization, later trees). A neural network would likely overfit.
- Placebo idea: shuffle the episode years within each country to see how unusual the real effect is.
- Neural network setup if one is ever used: ReLU in hidden layers, a linear output with from_logits=True, Adam. Report the parameter count against the number of training rows.
- Before changing a model, do error analysis: list the episodes it gets most wrong and group them by cause (country, period, extreme inputs). For yes/no results, report precision and recall next to accuracy and log loss.
