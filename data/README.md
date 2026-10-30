# Data

`panel_master.csv` is the analysis panel from my MSc thesis (TU Delft, 2026): 26 EU member states (EU minus Czech Republic), 1995–2024, 780 country-years. `DATA_DICTIONARY.csv` describes every column: unit, source and source code.

The notebooks load it directly from this repo:

```python
DATA_URL = "https://raw.githubusercontent.com/penelopemichou/austerity-ml/main/data/panel_master.csv"
```

## Sources and licences

| Columns | Source | Licence |
|---|---|---|
| `capb_main`, `dcapb_main`, `tfp`, `outputperworker`, `gfcf_public`, `gfcf_private`, `unemployment`, `real_compensation`, `exports`, `gdp_growth`, `debt_gdp`, `inflation`, `trade_open`, `long_rate`, `net_exports_gdp` | [AMECO](https://economy-finance.ec.europa.eu/economic-research-and-databases/economic-databases/ameco-database_en), European Commission (DG ECFIN) | [CC BY 4.0](https://commission.europa.eu/legal-notice_en) (Commission reuse policy, Decision 2011/833/EU) |
| `rd_intensity` | [Eurostat](https://ec.europa.eu/eurostat/databrowser/view/sdg_09_10/default/table), dataset `sdg_09_10` (GERD, % of GDP) | Reuse allowed with acknowledgement ([Eurostat copyright notice](https://ec.europa.eu/eurostat/help/copyright-notice)) |
| `gini_disp`, `gini_disp_se` | [SWIID](https://fsolt.org/swiid/), Frederick Solt | [CC0 1.0](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LM4OWF) |
| `consumer_confidence`, `n_months` | [Business and Consumer Surveys](https://economy-finance.ec.europa.eu/economic-forecast-and-surveys/business-and-consumer-surveys_en), European Commission (DG ECFIN) | [CC BY 4.0](https://commission.europa.eu/legal-notice_en) |

SWIID citation: Solt, Frederick. 2020. "Measuring Income Inequality Across Countries and Over Time: The Standardized World Income Inequality Database." *Social Science Quarterly* 101(3): 1183–1199.

## Changes from the original sources

The panel is my own compilation, not the original releases. I merged the series into one country-year table, computed `dcapb_main` as the year-on-year change in `capb_main`, computed `gfcf_private` as 100 × UIGP / UVGD, and averaged monthly consumer confidence to annual values (`n_months` = months averaged). The European Commission, Eurostat and the SWIID author are not responsible for these changes or for any errors in them.
