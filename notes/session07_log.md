# Learning log, Session 7 (14 Nov 2026)

**Course:** Machine Learning Specialization, Course 3 Week 2 (collaborative filtering: learned features, cost function, mean normalization, cold start, TensorFlow `GradientTape` and Adam; content-based filtering; PCA: principal components, explained variance, projection, reconstruction)

**Questions:** what do the 26 countries look like in one picture? Can collaborative filtering predict missing one-year TFP growth better than simple averages?

**Setup, decided before fitting:**
- **PCA on countries:** the same z-scored 1995–2024 averages as Session 6 (log trade openness, debt, inflation capped at ±30 %, GDP growth), coloured by Session 6's K = 2 groups. scikit-learn's k-means reproduces those two groups exactly.
- **PCA on country-years:** 754 country-years, Session 6's transformed inputs (log trade openness, square-root debt, signed-log inflation, GDP growth, one-year TFP growth), z-scored. ΔCAPB is left out, so episodes sit where their conditions put them.
- **Collaborative filtering:** the 26 × 29 table of one-year TFP growth (754 known cells); countries = users, years = items. 20 % of the known cells hidden at random, 10 times. Baseline = country average + year average on the training cells, which is also the mean normalization: the learned part models what is left. My own cost (half the squared error on known cells + λ/2 times the squares of all learned features), minimised with `GradientTape` and Adam (learning rate 0.05, 500 steps). 1, 2 or 3 learned features and λ ∈ {0.3, 1, 3, 10, 30} chosen by 5-fold cross-validation inside the training cells.
- This fills gaps in a country's own record; it is not an unseen-country test. A country with no known years would be a cold start.
- Rerun without the flagged artefacts: Bulgaria 1997, Ireland 2015, 2020 and 2021, Romania 2002. The notebook runs in about 6 minutes.

**PCA on the 26 countries: one picture**
- **Explained variance:** PC1 55.5 %, PC2 26.0 %, together 81.4 %; PC3 10.3 %, PC4 8.2 %. Reconstructing the four inputs from two components leaves an average squared error of 0.186 (in z-scores), which is exactly the 18.6 % not explained.
- **PC1 (left to right):** from high-debt, slow-growing, closed economies to fast-growing, low-debt, open ones (weights: GDP growth +0.57, debt −0.56, log trade openness +0.51, inflation +0.30). It is the "catching-up vs older core" axis.
- **PC2 (bottom to top):** higher inflation and less trade openness (inflation +0.83, log trade openness −0.50).
- **Where countries sit:** Session 6's two groups separate cleanly along PC1, with Croatia, Netherlands and Cyprus closest to the middle. Greece and Italy are furthest left (high debt, slow growth). Romania is alone at the top (inflation). Bulgaria, Hungary and Poland are above the catching-up group. Ireland, Luxembourg, Malta and Cyprus are at the bottom right (very open, low inflation).

**PCA on country-years**
- **Explained variance:** PC1 41.5 %, PC2 23.5 % (together 65 %).
- **PC1:** the boom-to-bust axis (GDP growth +0.62, one-year TFP growth +0.58, debt −0.40).
- **PC2:** more debt and less trade openness (debt +0.55, trade openness −0.54), with faster TFP and GDP growth.
- **Consolidation episodes** are spread through the whole cloud, slightly towards lower growth and higher debt (average position PC1 −0.42, PC2 −0.34, against +0.08 and +0.06 for other years). They are not a separate kind of country-year.
- **The flagged artefacts sit at the edges:** Ireland 2015 far top right (the most extreme point), Ireland 2020 and 2021 and Romania 2002 to the right (TFP jumps), Bulgaria 1997 far bottom left (inflation, collapse). Ireland 2010–11 sit in the middle, because their oddity is ΔCAPB, which is not an input here. Greece 2010 is to the left (a bust year).

**Collaborative filtering: does it beat simple averages?**

| Mean squared error on the hidden cells (average of 10 hides) | With all cells | Without the flagged artefacts |
|---|---|---|
| Overall mean | 8.18 | 5.94 |
| Country average only | 7.90 | 5.82 |
| **Baseline: country average + year average** | **6.03** | **4.08** |
| Collaborative filtering (features and λ chosen by cross-validation) | 6.11 | 3.93 |
| Collaborative filtering beats the baseline | 4 of 10 hides, median gain −0.0 % (−8.3 % to +2.7 %) | **8 of 10 hides, median +2.1 % (−0.1 % to +9.8 %)** |
| Chosen number of features | 1 ×5, 2 ×3, 3 ×2 | 3 ×4, 1 ×4, 2 ×2 |

- **The year average does most of the work.** Adding the year average to the country average cuts the error by about a quarter (7.90 → 6.03), because TFP growth moves together across countries in the same year (the 2009 crash, for example).
- **With all cells, learned features add nothing.** Cross-validation often picks λ = 30, which shrinks the learned features almost to zero, so the prediction becomes the baseline (several hides show a gain of exactly 0.0 %). In two hides the learned features made things worse (−8.3 %, −7.2 %).
- **Without the artefacts, a small, fairly steady gain appears:** better than the baseline in 8 of 10 hides, by a median of 2.1 %. The few huge artefact cells (Ireland 2015's +19.8 %, Bulgaria 1997's −13.7 %) dominate the squared error and hide the small pattern the learned features can find.
- In ML terms, the learned features catch a little of how countries' growth moves together beyond the common year effect. It is small, and it is about filling gaps in known countries, not about unseen ones or consolidations.

**Related countries**
- **By learned features** (3 features, λ = 30, all cells): the lists are **not stable**. Only 19 % of the nearest countries are the same across three random starts, and 22 % after dropping the artefacts. With λ = 30 the features are close to zero, so the distances between them are mostly noise. These lists should not be read as "related countries".
- **By PCA position** (average conditions): stable, with 94 % of neighbours unchanged without the artefacts, and easy to read. Greece's nearest are Italy, France and Portugal. Ireland's are Luxembourg, Malta and Slovakia. Estonia's are Lithuania, Latvia and Slovakia. Romania's are Bulgaria, Hungary and Poland.
- Only 17 % overlap between the two kinds of list, as expected given that the learned ones are mostly noise.

**Without the flagged artefacts: does anything important change?**
- **PCA on countries:** almost nothing. Explained variance moves from 55.5 % / 26.0 % to 57.3 % / 25.3 %. Positions correlate 0.997 (PC1) and 0.994 (PC2) with the original, and no weight changes by more than 0.05. Each artefact is one year in a 30-year average.
- **PCA on country-years:** the components keep their meaning (41.5 % → 42.3 % and 23.5 % → 23.4 %). Trade openness weighs more on PC2 (−0.54 → −0.66) and inflation less (−0.33 → −0.14).
- **Collaborative filtering: the one thing that changes.** It goes from no gain to a small, steady gain (above). So the artefacts matter for anything scored by squared error on single years.
- **Related countries:** the PCA neighbours barely move (94 % the same); the learned-feature neighbours are unstable either way.

**Main lessons**
- Two principal components capture 81 % of how the countries differ in average conditions, and the first one is Session 6's two-group split.
- Consolidation episodes do not form their own region of the data; they happen in all kinds of conditions, a little more often in bad ones.
- For one-year TFP growth, "country average + year average" is a strong baseline, and the year average is most of it.
- Learned features help only a little, and only once a handful of artefact cells are removed: squared error lets a few extreme cells drown a weak pattern.
- When regularization pushes learned features towards zero, "nearest neighbours" in that feature space are noise. Check stability across random starts before reading them.
- PCA and the data-artefact checks say the same as Session 6: a few country-years are not real economics and should be handled before any model is judged on single years.

**Next:** Course 3 Week 3 (reinforcement learning).
