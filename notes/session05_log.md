# Learning log, Session 5

## Part A: evaluating models (10 Nov 2026)

**Course:** Machine Learning Specialization, Course 2 Week 3, Parts A–B (training / cross-validation / test sets, model selection, bias and variance, baseline level of performance, learning curves)

**Question:** Sessions 3 and 4 never beat the simple guess. Is that because the models are too simple (high bias), too flexible for so little data (high variance), or because the inputs hold almost no information about what comes next? And would more data help?

**Setup, decided before fitting:**
- Same rows, inputs and targets as Sessions 3–4: 96 consolidation episodes in 24 countries; ΔCAPB and the four lagged controls; inflation capped at ±30 %; original and year-adjusted "weak growth" targets with cut-offs from the training countries only.
- Model family: logistic regression on polynomial features of the five inputs (degree 1, 2 or 3 → 5, 20 or 55 features), z-scored on the training countries only, λ from 0.01 to 1000 (C = 1/λ).
- J_train and J_cv are log loss without the λ term. The baseline level of performance is the simple guess (the training share of "weak").
- Cross-validation: the 20 training countries in 4 groups (15 to 20 episodes each). Each fold trains on about 53 episodes and scores only its left-out group.

**Train / cross-validation / test in this project**
- Training set: the countries a model learns from. Cross-validation set: training countries held back to choose the degree and λ. Test set: the six hidden countries (Belgium, Greece, Lithuania, Malta, Slovakia, Spain).
- The test set should be used once. Ours has been read in every session since Session 1, so it is no longer fully clean, even though nothing was tuned on it on purpose (apart from the Session 3 leak, since fixed). That is why "every country hidden in turn" and the 500 random splits matter more.

**J_train and J_cv against λ (cross-validation on the 20 training countries)**

| λ | Original, degree 1: J_train / J_cv | Original, degree 2 | Original, degree 3 | Year-adjusted, degree 1 | Year-adjusted, degree 2 | Year-adjusted, degree 3 |
|---|---|---|---|---|---|---|
| 0.01 | 0.617 / 0.877 | 0.338 / 2.486 | 0.116 / 2.881 | 0.629 / 0.901 | 0.379 / 2.047 | 0.125 / 2.887 |
| 1 | 0.619 / 0.823 | 0.459 / 1.077 | 0.372 / 1.201 | 0.631 / 0.860 | 0.482 / 0.898 | 0.377 / 1.365 |
| 10 | 0.640 / 0.732 | 0.560 / 0.760 | 0.504 / 0.787 | 0.646 / 0.764 | 0.563 / 0.660 | 0.504 / 0.926 |
| 30 | 0.659 / 0.706 | 0.607 / 0.696 | 0.565 / 0.682 | 0.663 / 0.722 | 0.608 / **0.643** | 0.564 / 0.763 |
| 100 | 0.676 / 0.698 | 0.649 / **0.680** | 0.618 / 0.702 | 0.678 / 0.703 | 0.649 / 0.658 | 0.619 / 0.750 |
| 1000 | 0.687 / 0.698 | 0.683 / 0.694 | 0.675 / 0.697 | 0.688 / 0.697 | 0.684 / 0.690 | 0.676 / 0.701 |
| Simple guess | J_cv 0.698 | | | J_cv 0.696 | | |

- **The textbook picture.** With little regularization, J_train is low and J_cv is far above it: **high variance**. Degree 3 with λ = 0.01 fits the training countries almost perfectly (J_train 0.12) and is terrible on the cross-validation countries (J_cv 2.9). With a lot of regularization, J_train and J_cv both rise or fall to the simple guess.
- **Degree 1 (the Session 3 model) never gets below the guess** in cross-validation: its best J_cv is 0.698 at λ = 100–1000, the same as the guess.
- **Degree 2 dips below the guess:** J_cv 0.680 vs 0.698 (original, λ = 100) and 0.643 vs 0.696 (year-adjusted, λ = 30). Cross-validation picks these. But the lowest J_cv among 27 options scored on the same 4 groups is optimistic: that is the course's reason for a separate test set.

**Learning curves (train on 2 to 16 training countries, score 4 others, 200 repeats)**

| At 16 training countries (about 57 episodes) | J_train | J_cv | Simple guess |
|---|---|---|---|
| Original, chosen by cross-validation (degree 2, λ = 100) | 0.650 | 0.695 | 0.694 |
| Original, flexible (degree 3, λ = 0.1) | 0.255 | 3.18 | 0.694 |
| Year-adjusted, chosen (degree 2, λ = 30) | 0.598 | 0.714 | 0.694 |
| Year-adjusted, flexible (degree 3, λ = 0.1) | 0.243 | 2.76 | 0.694 |

- **The chosen models:** J_cv falls and J_train rises as countries are added, and they close in on the simple guess from both sides. More data makes them *as good as* the guess, not better.
- **The flexible models:** a huge, slowly closing gap (high variance). More data would help them, but they start so far above the guess that 26 EU countries could never be enough.
- **Diagnosis:** in course terms, the well-regularized models have no variance problem, and their J_train is only a little below the guess. If the best achievable performance were much better than the guess, that would be high bias, and a more flexible model would fix it. But the more flexible models (degree 3 here, the network in Session 4) do not help. The limit is most likely the information in the data: these inputs say little about TFP growth over the next years. More countries do not exist; more years would help only slowly.

**The test, once: the cross-validation choice through the three tests**

| Test | Original (degree 2, λ = 100) | Year-adjusted (degree 2, λ = 30) |
|---|---|---|
| Six hidden: log loss vs baseline | 0.667 vs 0.710 | 0.624 vs 0.718 |
| Six hidden: accuracy vs baseline | 52 % vs 40 % | 52 % vs 40 % |
| Each hidden in turn: log loss vs baseline | 0.919 vs 0.694 (−32.4 %) | 0.704 vs 0.700 (−0.6 %) |
| Each hidden in turn: accuracy vs baseline | 58.3 % vs 49.0 % | 60.4 % vs 42.7 % |
| Countries won | 15 of 24 | 14 of 24 |
| 500 random splits: log loss better | 77 % (median +2.9 %) | 69 % (median +3.5 %) |
| Settings chosen across the 24 rounds | (2, 30) ×13, (2, 10) ×8, (1, 300), (2, 100), (3, 10) | (2, 30) ×11, (2, 10) ×5, (2, 100) ×5, (2, 300), (1, 10), (1, 300) |

- **Luxembourg again.** With the original target, Luxembourg's round chose degree 3 with λ = 10 and scored a log loss of 5.91 against 0.674 for the guess: that one round costs 39.3 points, more than the whole −32.4 %. With the year-adjusted target its round chose degree 1 with λ = 10 (1.313 vs 0.691, −4.6 points). The next worst rounds (Hungary, Estonia, Spain) cost less than 1 point each.
- Without Luxembourg's round the overall result would be roughly +7 % (original) and +4 % (year-adjusted). Together with 69–77 % of random splits, this is the most frequent small gain so far. But one unstable tuning round still decides the sign, the gain is small, the 500 splits use the setting chosen once (slightly optimistic), and no placebo check has been run on degree 2. A lead to check, not a usable predictor.

**Main lessons**
- J_train and J_cv together show what a single score cannot: low J_train with high J_cv is overfitting; both stuck at the guess means nothing is being learned.
- The J_cv of the chosen model is optimistic, because it won a contest on the same folds (0.643 in cross-validation, 0.704 with every country hidden in turn).
- Learning curves answer "would more data help?". Here the regularized models meet at the guess, and the flexible ones would need far more countries than the EU has.
- A test set read in every session slowly stops being a test set.
- Unstable tuning is the weak point: in Sessions 3, 4 and 5 the worst round is Luxembourg, where tuning picks a less regularized model and is badly wrong.

**Next:** Part B after Course 2 Week 4: decision trees, to test whether there is a large-consolidation threshold. Worth checking first: a placebo test for the degree-2 model, and whether Luxembourg's episodes are unusual.
