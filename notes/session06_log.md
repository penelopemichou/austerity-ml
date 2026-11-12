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

**Checking the new flags against the source (added 12 Nov 2026)**
- The panel's TFP matches the AMECO series (ZVGDF) exactly in the thesis's raw files, so none of these jumps comes from processing. Splitting GDP growth into output per worker and employment shows what drives each:

| | TFP growth | Output per worker | GDP growth | Implied employment change | Verdict |
|---|---|---|---|---|---|
| Ireland 2020 | +11.0 % | +9.4 % | +7.2 % | −2.3 % | multinational accounting |
| Ireland 2021 | +10.9 % | +8.7 % | +16.3 % | +7.5 % | multinational accounting |
| Romania 2002 | +12.2 % | +16.3 % | +5.7 % | −10.6 % | most likely an employment-series break |
| Estonia 1997 | +10.5 % | +11.8 % | +12.3 % | +0.5 % | real catch-up growth |

- **Ireland 2020–21:** the same problem as 2015. Irish GDP reflects multinationals' accounts (intellectual property, contract manufacturing, pharma exports during COVID), not domestic productivity; GDP grew 7 % in 2020 when most EU economies shrank. Smaller jumps in 2014 (+6.3 %) and 2017 (+5.5 %) follow the same pattern.
- **Romania 2002:** GDP grew a normal 5.7 %, but employment would have had to fall 10.6 % in one year for TFP to jump 12 %. That fits a statistical revision of the employment figures (Romania's 2002 census), not a real fall; not yet confirmed against Eurostat's notes.
- **Estonia 1997:** GDP really grew about 12 % with employment flat: genuine fast catch-up growth, not a data problem.
- Both artefacts are now in CLAUDE.md's data watch-outs. They feed into the TFP growth target for Ireland's 2018–2021 windows and Romania's 2000–2002 windows.

**Quick check (12 Nov 2026): the Session 5B threshold without the flagged windows, and the K = 2 groups against their normal growth**

*1. ΔCAPB bands and the stump cut* (Session 5B sample of 669 country-years; average TFP growth, t−1 to t+2; dropping Ireland's target windows from 2013 on and Romania 2000–2002, 13 country-years, then Greece as well). No earlier notebook was changed.

| ΔCAPB band (pp) | < −1.5 | −1.5 to 0 | 0 to 1.5 | 1.5 to 3 | 3 to 4.5 | > 4.5 |
|---|---|---|---|---|---|---|
| All (Session 5B) | 3.78 % (120) | 3.56 % (247) | 3.78 % (209) | 3.57 % (54) | 2.09 % (21) | 0.16 % (18) |
| Without flagged windows | 3.56 % (118) | 3.17 % (242) | 3.73 % (206) | 2.78 % (51) | 2.09 % (21) | 0.16 % (18) |
| Without flagged windows and Greece | 3.76 % (107) | 3.17 % (236) | 3.73 % (203) | 2.77 % (50) | 1.95 % (20) | 2.59 % (14) |

| Stump (best single ΔCAPB cut) | Original: cut, below vs above | Gap | Placebo: shuffles as large | Year-adjusted: cut, gap | Placebo |
|---|---|---|---|---|---|
| All (Session 5B) | 3.45 pp: +3.69 % vs +0.28 % | 3.40 | 1.6 % | 3.45 pp, 2.78 | 2.6 % |
| Without flagged windows | 3.45 pp: +3.41 % vs +0.28 % | 3.13 | 0.4 % | 3.45 pp, 2.55 | 1.0 % |
| Without flagged windows and Greece | **1.85 pp**: +3.52 % vs +2.04 % | 1.47 | **18.2 %** | **0.75 pp**, 0.94 | **25.0 %** |

- **The flagged windows do not matter:** without Ireland 2013 onwards and Romania 2000–2002, the bands above 3 pp and the 3.45 pp cut are unchanged, and the placebo is even clearer (0.4 % and 1.0 %).
- **Greece does:** without it, the stump's best cut moves to 1.85 pp (original) and 0.75 pp (year-adjusted), the gap shrinks to 1.47 and 0.94 points, and the placebo no longer stands out (18 % and 25 % of shuffles do as well). The top band rises from 0.16 % to 2.59 %, because Greece 2010 and 2011 (−16.9 % and −16.5 %) are in it.
- At a fixed 3.45 pp cut, growth above it is still lower without Greece (mean 1.48 % vs 3.45 %, median 0.72 % vs 3.03 %, year-adjusted −0.48 vs +0.71), so large consolidations were still followed by weaker growth. But without Greece the bands look more like a general step down for all consolidations (2–2.8 % from 1.5 pp up, against 3.2–3.8 % for other years) than a sharp drop above about 3.5 pp.
- **This corrects Session 5B:** its robustness check measured the gap at a fixed 3.45 pp cut and found it held without Greece, but did not let the cut move. Letting it move shows that the 3.45 pp threshold, and its placebo pass, rest on Greece's crisis years.

*2. Session 6's K = 2 groups: growth after consolidations minus each group's normal growth* (Session 6 sample; "normal" = average over all country-years of the group; year-adjusted = minus the median across countries in the same year).

| | Episodes | After episodes | Normal | Difference | Year-adjusted: after | Normal | Difference |
|---|---|---|---|---|---|---|---|
| Catching-up and small open | 64 | 3.87 % | 5.55 % | **−1.68** | +1.30 | +2.70 | **−1.41** |
| Older core | 42 | 1.44 % | 1.68 % | **−0.24** | −1.57 | −1.15 | **−0.42** |
| Catching-up, without flagged windows | 61 | 3.22 % | 5.11 % | −1.89 | +0.66 | +2.28 | −1.62 |

- Compared with its own normal growth, the catching-up group's growth falls much more after consolidations (−1.7 points, −1.4 year-adjusted) than the older core's (−0.2, −0.4 year-adjusted). Dropping the flagged Irish and Romanian windows makes the catching-up gap slightly larger (−1.9, −1.6). The older core has no flagged windows.
- A description, not a cause: the catching-up economies grow fast normally, so a pause after consolidating shows up as a larger gap, and their consolidations cluster in crisis years.

**Main lessons**
- K-means always finds groups; whether they mean anything is shown by stability. Here K = 2 is solid, and from K = 4 the groups depend on the random start.
- Choosing K by purpose matters more than the elbow, which had no clear bend after K = 2.
- A Gaussian per feature catches years that are extreme on one feature (Bulgaria 1997, Ireland), but misses years that are unusual only as a combination (Greece 2010).
- With five labelled anomalies, precision and recall rest on very few cases; ε is a judgement call as much as a calculation.
- Anomaly detection is a useful data check: it confirmed the known quirks and found two new ones that the source data confirms (Ireland 2020–21, Romania 2002), plus one that turned out to be real (Estonia 1997).
- Clustering and anomaly detection describe the data; they say nothing about whether consolidations cause weaker growth.
- A robustness check should let the result move, not only test it at a fixed value: the 3.45 pp cut held at a fixed cut without Greece, but the best cut moved away once it could.

**Next:** Course 3 Week 2 (recommender systems).
