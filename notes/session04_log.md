# Learning log, Session 4 (9 Nov 2026)

**Course:** Machine Learning Specialization, Course 2 Weeks 1–2 (neural networks, forward propagation, TensorFlow `Sequential` / `compile` / `fit`, ReLU, sigmoid, linear and softmax outputs, `from_logits`, Adam, backprop)

**Question:** does a small neural network beat Session 3's logistic regression on the same yes/no target (TFP growth from t−1 to t+2 below the median), and on the year-adjusted one?

**Setup, decided before fitting:**
- Same rows, inputs and targets as Session 3: 96 consolidation episodes in 24 countries; ΔCAPB and the four lagged controls, z-scored on the training countries only; inflation capped at ±30 %.
- Network: 5 inputs → one ReLU hidden layer → one linear output unit, `BinaryCrossentropy(from_logits=True)`, Adam (learning rate 0.01), 500 full-batch steps. L2 on the weights of both layers, with Keras `l2 = λ/(2m)` so that λ means the same as in Course 1.
- The settings are chosen on the training countries only, by splitting them into 4 groups and scoring each left-out group (never the hidden countries). Logistic regression gets the same treatment.
  - Network: hidden units ∈ {2, 4, 8} and λ ∈ {0.3, 1, 3, 10}. A check on the training countries showed that λ = 10 already shrinks the network to an almost flat guess, so 100 was dropped.
  - Logistic regression: λ ∈ {0, 0.1, 1, 10, 100}.
- Same three tests and the same majority-class baseline as Session 3. Random seed 0 for the main results; seeds 0–4 to see how much they move.

**Checks**
- Size: 5 inputs and 2 / 4 / 8 hidden units give 15 / 29 / 57 parameters, against 71 training episodes (0.21 / 0.41 / 0.80 parameters per episode). Even the "tiny" 8-unit network has almost one parameter per episode.
- Speed trick: repeating the data 500 times inside one `fit` call (`steps_per_execution`) gives exactly the same weights as `epochs=500` (difference 0.0), about 6 times faster.
- 500 steps are enough: for the least regularized, largest network (8 units, λ = 0.3) the loss is 0.4468 after 500 steps and 0.4449 after 2000.
- With the 4-group tuning, logistic regression reproduces Session 3's original-target results exactly (six hidden 0.695 vs 0.710; each hidden in turn −5.2 %, 17 of 24). On the year-adjusted target it comes out a little worse than in Session 3 (−3.7 % instead of −2.5 %), because 2 rounds now choose λ = 10.

**Results**

| Test | Logistic regression, original | Neural network, original | Logistic regression, year-adjusted | Neural network, year-adjusted |
|---|---|---|---|---|
| Chosen on the six-hidden training countries | λ = 100 | 8 units, λ = 10 | λ = 100 | 8 units, λ = 3 |
| Six hidden: log loss vs baseline | 0.695 vs 0.710 | 0.705 vs 0.710 | 0.704 vs 0.718 | **0.639 vs 0.718** |
| Six hidden: accuracy vs baseline | 40 % vs 40 % | 40 % vs 40 % | 40 % vs 40 % | **72 % vs 40 %** |
| Each hidden in turn: log loss vs baseline | 0.730 vs 0.694 (−5.2 %) | 0.763 vs 0.694 (−10.0 %) | 0.726 vs 0.700 (−3.7 %) | 0.713 vs 0.700 (−1.9 %) |
| Each hidden in turn: accuracy vs baseline | 63.5 % vs 49.0 % | 57.3 % vs 49.0 % | 55.2 % vs 42.7 % | 55.2 % vs 42.7 % |
| Countries won | 17 of 24 | 14 of 24 | 13 of 24 | 12 of 24 |
| 500 random splits: log loss better | 76 % (median +1.6 %) | 71 % (median +1.2 %) | 72 % (median +0.9 %) | 49 % (median −0.5 %) |

- **The answer is no.** With every country hidden in turn, the network is worse than the baseline on both targets, and worse than logistic regression on the original target. In the random splits it wins less often than logistic regression.
- **The one bright spot is luck.** On the year-adjusted target the network looks clearly better on the six hidden countries (log loss 0.639, accuracy 72 %). But that is seed 0 on one split: with seeds 1–4 the same test gives 0.659 to 0.751, the fairer tests show no gain, and the random splits are a coin flip (49 %).
- **The network's settings jump around.** Across the 24 rounds the original-target network chose 5 different settings (8 or 4 units with λ = 3 most often), and the year-adjusted one chose 8 different settings. Logistic regression chose λ = 100 in 23 of 24 rounds (22 of 24 year-adjusted). When the best setting changes with every small change in the data, there is no stable pattern to find.

**How much the seed matters (network, seeds 0–4)**

| Target | Six hidden: log loss (settings chosen again per seed) | Each hidden in turn: log loss gain (settings fixed) | Countries won |
|---|---|---|---|
| Original | 0.644 to 0.705 | −0.4 % to +0.8 % | 15 or 16 of 24 |
| Year-adjusted | 0.639 to 0.751 | −5.9 % to +0.8 % | 11 to 13 of 24 |

- Changing only the seed moves the six-hidden log loss by up to 0.11, more than the gap between any model and the baseline. The seed even changes which settings get chosen. A single run of a network on this little data can look good or bad by chance.
- (With the settings fixed at the section 5 choice, every country hidden in turn gives −0.1 % for seed 0 on both targets, not the −10.0 % and −1.9 % above. Choosing the settings again in each round costs a lot, because the choice is unstable.)

**Optional extra: three classes (weak / middle / strong, year-adjusted)**
- The cut-offs are the thirds of the training countries' growth in the same year. Class shares in the training episodes are 34 / 35 / 31 %. Output: 3 linear units with softmax, `SparseCategoricalCrossentropy(from_logits=True)`. Chosen: 8 units, λ = 10.
- Six hidden: log loss 1.113 vs 1.117, accuracy 20 % vs 24 %. Each hidden in turn: 1.184 vs 1.116 (−6.1 %), accuracy 33.3 % vs 35.4 %, 11 of 24 countries. 500 splits: log loss better in 57 % (median +0.4 %).
- No gain: with three classes the network does not even beat always guessing the most common class.

**Main lessons**
- A more flexible model does not help when there is almost no signal. The network adds parameters, not information.
- With about 70 training rows, even a "tiny" network has nearly one parameter per row, and its settings and seed change the result more than the model beats the baseline by.
- Always report the seed spread for a network. One good seed on one split (72 % accuracy here) is exactly the kind of result that does not survive the fairer tests.
- Regularization pushes the network towards a flat guess, just as λ = 100 did for logistic regression: on this data the best a model can do is stay close to the baseline.
- This matches the learning note in CLAUDE.md: with a few hundred country-years, prefer simple models. ML still adds nothing reliable over the econometrics.

**Next:** Session 5, Course 2 Weeks 3–4: evaluating models (train / cross-validation / test, bias and variance), then decision trees, to test whether there is a large-consolidation threshold.
