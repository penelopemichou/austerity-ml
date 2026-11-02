# Correction to two thesis controls (2 Nov 2026)

While building Session 2, I found that two of the four control variables in my thesis panel were built from mismatched AMECO series. This note records what was wrong, what changed, and how much it matters.

## What was wrong

| Control | How it was built | Problem |
|---|---|---|
| `trade_open` | Exports + imports (current prices, EUR) ÷ GDP (constant prices, national currency) | Currencies and price bases did not match. A file named `UVGD.xlsx` actually held OVGD. Sweden showed about 8 % of GDP and Hungary about 0.4 %. |
| `inflation` | % change in the private consumption deflator (PCPH) in EUR | Exchange-rate moves counted as inflation. Poland 2009 showed −16.5 %; Bulgaria 1997 showed 35 %. |

`net_exports_gdp` had the same fault as `trade_open`, but it is not used as a control. Rebuilding the old columns from the raw files matched the panel exactly, which confirms this was the cause.

## The fix

`data/panel_master.csv` now computes these three columns from AMECO's national-currency series (release of 2 June 2026), so units match. `data/fix_controls.py` rebuilds them. All other columns and the sample are unchanged. The panel as used in the submitted thesis is at commit `5787f6a`.

After the fix, Sweden is about 84 % of GDP, Hungary about 155 %, Poland 2009 inflation is +2.9 %, and Bulgaria 1997 inflation is about 1,040 %, which is a real hyperinflation.

## Effect on the thesis results

I re-ran the thesis code on the corrected panel: baseline, robustness and extensions. Run unchanged on the old panel, the same code reproduced the saved results (largest gap about 1e-12). The original thesis files were not changed.

**The main findings hold.** TFP at h = 2 moves from −1.50 % to −1.53 %.

| TFP, cumulative change | h = −2 | h = 0 | h = 2 | h = 4 |
|---|---|---|---|---|
| Submitted | −0.56 % | −1.53 % | −1.50 % | −0.71 % |
| Corrected | −0.54 % | −1.53 % | −1.53 % | −0.73 % |

Results that cross a significance line (out of 50 baseline cells at h ≥ 0 and 12 composition cells):

| Result | Submitted | Corrected | Change |
|---|---|---|---|
| Unemployment, h = 1 | p = 0.051 | p = 0.044 | now significant at 5 % |
| Unemployment, h = 2 | p = 0.131 | p = 0.098 | now significant at 10 % |
| Social benefits, h = 2 (composition) | p = 0.059 | p = 0.101 | no longer significant at 10 % |

- Driscoll-Kraay standard errors, the wild cluster bootstrap and the debt-heterogeneity test have no changes at the 10 % line.
- Coefficient signs change only in results that are far from significant.
- Capping inflation at ±30 % leaves all baseline results on the same side of the 10 % line as submitted, and TFP at h = 2 is −1.48 %.

Every baseline number is in `controls_correction_baseline.csv`.

## Teulings–Zubanov, threshold and sample checks

I re-ran these three checks on both panels. **No main conclusion changes:** only 6 of the 120 results change significance stars, all marginally. Coefficients are in % for log outcomes and native units otherwise. The code stores only coefficients for the threshold and sample tables, so I added p-values using the baseline method (country-clustered, t with 25 degrees of freedom), and applied the same method to Teulings–Zubanov.

On the old panel, the threshold and sample results match printed Tables 7.4 and 7.5 of the thesis in all 90 cells, for both coefficients (to two decimals) and stars. So the model and the p-value method are the same as in the thesis for these two tables. I did not make this comparison for Teulings–Zubanov (Table 7.1), so its stars may differ from the printed ones.

| Check | Results | Biggest coefficient change | Sign flips | Star changes |
|---|---|---|---|---|
| Teulings–Zubanov | 30 | 0.57 | 0 | 3 |
| Threshold (1.0 / 1.5 / 2.0) | 45 | 0.73 | 1 | 1 |
| Sample (full / no Greece / no CEE) | 45 | 0.70 | 0 | 2 |

The largest coefficient changes are all in public investment, on estimates of about −9 to −14, and those stay significant at 1 %. The one sign flip is consumer confidence at threshold 1.0, h = 2 (+0.03 to −0.01, p about 0.98).

| TFP at h = 2 | Submitted | Corrected | Corrected p |
|---|---|---|---|
| Baseline | −1.50 % | −1.53 % | 0.004 |
| Teulings–Zubanov | −1.67 % | −1.69 % | 0.003 |
| Threshold 1.0 | −1.64 % | −1.64 % | 0.001 |
| Threshold 2.0 | −2.14 % | −2.22 % | 0.001 |
| No Greece | −1.28 % | −1.32 % | 0.009 |
| No CEE | −0.75 % | −0.78 % | 0.262 |

The six star changes:

| Check | Result | Submitted | Corrected |
|---|---|---|---|
| Teulings–Zubanov | Unemployment, h = 2 | +0.67, p = 0.056 | +0.73, p = 0.046 |
| Teulings–Zubanov | Real compensation, h = 4 | −1.89 %, p = 0.008 | −1.78 %, p = 0.017 |
| Teulings–Zubanov | Gini, h = 4 | +0.31, p = 0.011 | +0.31, p = 0.009 |
| Threshold 1.0 | Consumer confidence, h = 4 | +1.58, p = 0.088 | +1.45, p = 0.110 |
| No Greece | Real compensation, h = 4 | −1.03 %, p = 0.038 | −0.90 %, p = 0.070 |
| No CEE | Gini, h = 2 | +0.21, p = 0.055 | +0.21, p = 0.045 |

One sensitivity was already there before the correction: without the CEE countries, TFP at h = 2 is not significant (−0.78 %, p = 0.262), and neither is real compensation (−0.96 %, p = 0.115). So the medium-term TFP result leans on the CEE countries.

Every number is in `controls_correction_robustness.csv`.

## A separate label issue

The `exports` column is AMECO's *market performance of exports on export-weighted imports* (VXGSP), not an export volume index as the thesis text and the old data dictionary describe it. The dictionary here is now corrected. Re-running with true export volume (OXGS) gives the same conclusion: no significant effect at any horizon.

This is a check of the reproduced code on corrected data. It is not a re-review of the thesis itself.
