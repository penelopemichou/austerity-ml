# Learning log, Session 6 (12 Nov 2026)

**Course:** Machine Learning Specialization, Course 3 Week 1 (k-means clustering: random initialisation, cost, choosing K by purpose; anomaly detection: a Gaussian per feature, p(x) < ε, tuning ε on a few known anomalies, transforming features to look more Gaussian)

**Questions:** do EU countries fall into natural groups? Does the data flag the country-years we already know are strange?

**Setup, decided before fitting:**
- **Clustering:** one row per country, its 1995–2024 average of log trade openness, debt, inflation capped at ±30 % and GDP growth, z-scored across the 26 countries. My own k-means (as in the course lab), 100 random starts, lowest cost kept; K = 2 to 5.
- **Anomaly detection:** one row per country-year (743 with all six features): ΔCAPB, GDP growth, inflation (uncapped), debt, trade openness and one-year TFP growth. Skewed features transformed: inflation with a signed log, trade openness with a log, debt with a square root (a log over-corrected it, leaving a long low tail from Estonia; decided from the histograms only). A Gaussian per feature on 60 % of the ordinary country-years; ε tuned by F1 on the other 40 % plus the five known anomalies from CLAUDE.md (Bulgaria 1997, Ireland 2010, 2011 and 2015, Greece 2010). No separate test set: five labels are too few.

**Checks**
- My k-means reaches exactly scikit-learn's lowest cost for every K (K = 2: 2.3641; 3: 1.7778; 4: 1.3882; 5: 1.0403).
- The notebook runs in about 10 seconds.

**The groups, in plain words**
- **Elbow plot:** the cost falls from 4.00 (K = 1) to 2.36, 1.78, 1.39 and 1.04 (K = 5). The only clear bend is at K = 2; after that it falls steadily. As the course says, the elbow is only a hint.
- **K = 2: old core vs catching-up and small open economies.**
  - Group 1 (13): Austria, Belgium, Croatia, Denmark, Finland, France, Germany, Greece, Italy, Netherlands, Portugal, Spain, Sweden. Less open to trade (84 % of GDP), higher debt (78 %), low inflation (2.1 %), slow growth (1.7 %).
  - Group 2 (13): Bulgaria, Cyprus, Estonia, Hungary, Ireland, Latvia, Lithuania, Luxembourg, Malta, Poland, Romania, Slovakia, Slovenia. Very open (142 %), low debt (42 %), higher inflation (4.5 %), fast growth (3.6 %).
- **K = 3:** the high-inflation transition economies split off: Bulgaria, Hungary, Romania (inflation 8.2 %).
- **K = 4:** Greece and Italy also form their own group: debt 132 %, growth 1.0 %, the least open (56 %).
- **K = 5:** the catching-up group splits into the very open small economies (Cyprus, Ireland, Luxembourg, Malta: trade 201 % of GDP, growth 4.3 %) and Central and Eastern Europe (Bulgaria, Estonia, Hungary, Latvia, Lithuania, Poland, Slovakia, Slovenia), with Romania alone (inflation 11.5 %).
- Croatia sits with the old core in every K, despite joining the EU in 2013: its averages (trade 84 %, debt 53 %, growth 2.6 %) are closer to that group.

**Are the groups stable?**

| | K = 2 | K = 3 | K = 4 | K = 5 |
|---|---|---|---|---|
| 20 other seeds (100 starts each): lowest pair agreement | 1.00 | 1.00 | 0.89 | 0.86 |
| A single random start: median (lowest) | 1.00 (0.48) | 0.84 (0.61) | 0.80 (0.66) | 0.80 (0.70) |
| Each country hidden in turn: median (lowest) | 1.00 (0.92) | 1.00 (0.87) | 0.96 (0.72) | 1.00 (0.86) |
| Rounds giving exactly the same grouping | 23 of 26 | 16 of 26 | 12 of 26 | 15 of 26 |

- **K = 2 is the stable grouping.** Every seed finds it, and hiding any one country changes it only for Belgium, Luxembourg or Romania, the countries between the two groups.
- **From K = 4, the answer depends on the random seed** even with 100 starts each (pair agreement 0.89 and 0.86): there are several almost equally good groupings. One random start alone can be far off (pair agreement as low as 0.48 for K = 2), which is why the course uses many starts.
- **Choosing K by purpose:** for comparing growth after consolidations, K = 2 is the honest choice (stable, 42 and 64 episodes). K = 3 and 4 add interpretable but small groups (3 and 2 countries) that are less stable.

**TFP growth after consolidations, by group (a description, not a cause)**

| | Episodes | Growth after episodes (t−1 to t+2) | Median | Year-adjusted | All country-years |
|---|---|---|---|---|---|
| K = 2, old core | 42 | 1.44 % | 2.28 % | −1.57 | 1.68 % |
| K = 2, catching-up and small open | 64 | 3.87 % | 4.22 % | +1.30 | 5.55 % |
| K = 4, Bulgaria, Hungary, Romania | 17 | 0.88 % | 2.64 % | −1.65 | 5.50 % |
| K = 4, Greece and Italy | 10 | −2.53 % | 0.01 % | −4.82 | 0.65 % |

- After consolidations, growth is weaker in the old core (1.44 %) than in the catching-up group (3.87 %), but that mostly reflects their normal growth (1.68 % vs 5.55 % over all years). Compared with their own normal, the catching-up group's growth falls more after consolidations (−1.7 points vs −0.2).
- The weakest: Greece and Italy (−2.53 % after their 10 episodes, against 0.65 % normally) and Bulgaria, Hungary and Romania (0.88 % vs 5.50 %). Small groups, few episodes, and crisis-era timing for many of them, so this describes where growth was weak after consolidating, not why.

**Anomaly detection: does it flag the known strange years?**
- At the ε chosen by F1 (log ε = −48.7), it flags **4 of the 5 known anomalies and nothing else** in the cross-validation set: precision 1.00, recall 0.80.
- **Caught:**
  - Bulgaria 1997 (inflation 1,040 %; most unusual of 301)
  - Ireland 2015 (GDP +24.6 %, TFP +19.8 %; second)
  - Ireland 2010 (ΔCAPB −18.8; third)
  - Ireland 2011 (ΔCAPB +18.2; fourth)
- **Missed: Greece 2010** (10th of 301). It is unusual as a *combination* (ΔCAPB +7.4, GDP −5.7 %, debt 148 %), but not extreme on any single feature, and a Gaussian per feature cannot see combinations. The course's fix is to add a feature that combines them (for example a ratio or product), decided before looking.
- **Precision and recall trade-off** on the cross-validation set: flagging the lowest 1 % catches 4 of 5 with precision 1.00; 5 % catches all 5 with precision 0.31; 10 % catches all 5 with precision 0.16. Loosening ε on the whole panel just enough to catch Greece 2010 flags 19 country-years (2.6 %), a precision of 0.26.

**The most unusual years I had not labelled, grouped by cause**
- **The 2009 Baltic crash:** Estonia 2009 (GDP −14.5 %), Lithuania 2009 (−14.9 %), Latvia 2009 (−16.0 %). Real events, not data errors.
- **Transition-era inflation:** Romania 1997 (134 %), Bulgaria 1996 (113 %).
- **The Greek crisis:** Greece 2011 (TFP −7.9 %), 2013 (ΔCAPB −5.7), 2014 (ΔCAPB +8.4), 2020 (GDP −9.2 %, debt 209 %). Greece has 6 of the 30 most unusual unlabelled years.
- **Large budget swings:** Cyprus 2013 (ΔCAPB +12.0, already in CLAUDE.md's watch-outs) and Lithuania 1997 (ΔCAPB −9.6).
- **Possible data quirks worth checking:** one-year TFP jumps of +11.0 % in Ireland 2020 and +10.9 % in 2021 (likely the same multinational accounting as 2015), +12.2 % in Romania 2002 and +10.5 % in Estonia 1997.

**Main lessons**
- K-means always finds groups; whether they mean anything is shown by stability. Here K = 2 is solid, and from K = 4 the groups depend on the random start.
- Choosing K by purpose matters more than the elbow, which had no clear bend after K = 2.
- A Gaussian per feature catches years that are extreme on one feature (Bulgaria 1997, Ireland), but misses years that are unusual only as a combination (Greece 2010).
- With five labelled anomalies, precision and recall rest on very few cases; ε is a judgement call as much as a calculation.
- Anomaly detection is a useful data check: it confirmed the known quirks and pointed to new ones (Ireland 2020–21, Romania 2002) to look at before any future model uses those years.
- Clustering and anomaly detection describe the data; they say nothing about whether consolidations cause weaker growth.

**Next:** Course 3 Week 2 (recommender systems). Before that, possibly check the new TFP jumps (Ireland 2020–21, Romania 2002, Estonia 1997) against the AMECO source.
