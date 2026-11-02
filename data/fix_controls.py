"""
Rebuilds the corrected controls in panel_master.csv from AMECO's bulk file.

    python data/fix_controls.py <thesis_panel_master.csv> <folder with unzipped ameco0.zip> <output.csv>

AMECO bulk file: https://ec.europa.eu/economy_finance/db_indicators/ameco/documents/ameco0.zip
(release used here: 2 June 2026). The thesis panel is this repo's data/panel_master.csv at commit 5787f6a.

  trade_open      = 100 * (UXGS + UMGS) / UVGD   all at current prices, in national currency
  net_exports_gdp = 100 * (UXGS - UMGS) / UVGD   same units
  inflation       = % change in PCPH, national currency
Every other column stays as it was. Inflation stays blank where it was blank, so the sample does not change.
"""
import csv, glob, os, sys
import pandas as pd

ISO = {"AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria", "HRV": "Croatia", "CYP": "Cyprus", "DNK": "Denmark",
       "EST": "Estonia", "FIN": "Finland", "FRA": "France", "DEU": "Germany", "GRC": "Greece", "HUN": "Hungary",
       "IRL": "Ireland", "ITA": "Italy", "LVA": "Latvia", "LTU": "Lithuania", "LUX": "Luxembourg", "MLT": "Malta",
       "NLD": "Netherlands", "POL": "Poland", "PRT": "Portugal", "ROM": "Romania", "SVK": "Slovakia",
       "SVN": "Slovenia", "ESP": "Spain", "SWE": "Sweden"}
SERIES = {("1.0.0.0", "UXGS"): "ux", ("1.0.0.0", "UMGS"): "um", ("1.0.0.0", "UVGD"): "gdp", ("3.1.0.0", "PCPH"): "prices"}

def read_ameco(folder):
    rows = []
    for path in glob.glob(os.path.join(folder, "*.TXT")):
        with open(path, encoding="latin-1", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            years = [(i, int(h)) for i, h in enumerate(next(reader)) if h.isdigit()]
            for r in reader:
                code = r[0].split(".")
                if len(code) < 6 or code[0] not in ISO:
                    continue
                name = SERIES.get((".".join(code[1:5]), code[5]))
                if name:
                    rows += [(ISO[code[0]], y, name, pd.to_numeric(r[i], errors="coerce")) for i, y in years if i < len(r)]
    table = pd.DataFrame(rows, columns=["country", "year", "series", "value"])
    return table.pivot_table(index=["country", "year"], columns="series", values="value").reset_index()

def main(panel_path, ameco_folder, out_path):
    panel = pd.read_csv(panel_path)
    a = read_ameco(ameco_folder).sort_values(["country", "year"])
    a["inflation_new"] = a.groupby("country")["prices"].pct_change(fill_method=None) * 100
    m = panel[["country", "year"]].merge(a, on=["country", "year"], how="left")
    panel["trade_open"] = 100 * (m["ux"] + m["um"]) / m["gdp"]
    panel["net_exports_gdp"] = 100 * (m["ux"] - m["um"]) / m["gdp"]
    panel["inflation"] = m["inflation_new"].where(panel["inflation"].notna())
    panel.to_csv(out_path, index=False)

if __name__ == "__main__":
    main(*sys.argv[1:4])
