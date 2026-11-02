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

## A separate label issue

The `exports` column is AMECO's *market performance of exports on export-weighted imports* (VXGSP), not an export volume index as the thesis text and the old data dictionary describe it. The dictionary here is now corrected. Re-running with true export volume (OXGS) gives the same conclusion: no significant effect at any horizon.

This is a check of the reproduced code on corrected data. It is not a re-review of the thesis itself.
