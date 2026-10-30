"""
Session 1 (30 Oct 2026): one-variable linear regression, dCAPB -> TFP growth (t-1 to t+2).
Same steps as the Colab notebook ML01, but loading the panel from disk.
Course link: Andrew Ng ML Specialization, Course 1 Week 1.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PANEL = "~/Documents/Fiscal_Consolidation_Project/04_analysis_data/panel_master.csv"
COUNTRY, YEAR, DCAPB, TFP = "country", "year", "dcapb_main", "tfp"
H = 2

# 1. Load data and build the target (cumulative % TFP growth, t-1 -> t+H, within country)
df = pd.read_csv(PANEL).sort_values([COUNTRY, YEAR]).copy()
df["log_tfp"] = np.log(df[TFP])
lead = df.groupby(COUNTRY)["log_tfp"].shift(-H)
lag1 = df.groupby(COUNTRY)["log_tfp"].shift(1)
df["y"] = 100 * (lead - lag1)
data = df[[COUNTRY, YEAR, DCAPB, "y"]].dropna()
x, y = data[DCAPB].to_numpy(), data["y"].to_numpy()
print(len(data), "usable rows")  # 691

# 2. Gradient descent, written as in the course labs
def compute_cost(x, y, w, b):
    m = len(x)
    return np.sum((w * x + b - y) ** 2) / (2 * m)

def compute_gradient(x, y, w, b):
    m = len(x)
    err = (w * x + b) - y
    return np.sum(err * x) / m, np.sum(err) / m

def gradient_descent(x, y, w=0.0, b=0.0, alpha=0.01, iters=5000):
    J_history = []
    for _ in range(iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w, b = w - alpha * dj_dw, b - alpha * dj_db
        J_history.append(compute_cost(x, y, w, b))
    return w, b, J_history

w, b, J_hist = gradient_descent(x, y)
w_ols, b_ols = np.polyfit(x, y, 1)
r2 = 1 - np.sum((w * x + b - y) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"GD w={w:.3f} b={b:.3f} | OLS w={w_ols:.3f} b={b_ols:.3f} | R2={r2:.3f}")  # -0.198, 3.605, 0.007

# 3. Plot: cost curve + fitted line
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(J_hist); ax[0].set_title("Cost J during gradient descent")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("J(w,b)")
ax[1].scatter(x, y, s=12, alpha=0.5)
xs = np.linspace(x.min(), x.max(), 100)
ax[1].plot(xs, w * xs + b, color="red", label=f"slope w = {w:.2f}")
ax[1].axvline(1.5, ls="--", color="grey", label="episode threshold 1.5 pp")
ax[1].set_xlabel("dCAPB at t (pp of GDP, + = tightening)")
ax[1].set_ylabel(f"TFP growth t-1 -> t+{H} (%)"); ax[1].legend()
plt.tight_layout(); plt.show()

# 4. Two-way fixed effects by iterative demeaning (unbalanced panel), same GD code
d = data.copy()
for col in [DCAPB, "y"]:
    v = d[col].copy()
    for _ in range(50):
        v = v - v.groupby(d[COUNTRY]).transform("mean")
        v = v - v.groupby(d[YEAR]).transform("mean")
    d[col + "_fe"] = v
x_fe, y_fe = d[DCAPB + "_fe"].to_numpy(), d["y_fe"].to_numpy()
w_fe, b_fe, _ = gradient_descent(x_fe, y_fe)
r2_fe = 1 - np.sum((w_fe * x_fe + b_fe - y_fe) ** 2) / np.sum((y_fe - y_fe.mean()) ** 2)
print(f"FE w={w_fe:.3f} within R2={r2_fe:.3f}")  # -0.268, 0.022

# 5. Put on the thesis scale (effect per average episode)
ep = data[DCAPB] >= 1.5
avg_ep = data.loc[ep, DCAPB].mean()
print(f"episodes={ep.sum()} avg size={avg_ep:.2f} pp -> {w_fe * avg_ep:.2f} % (thesis LP h=2: -1.50 %)")  # 106, 3.31, -0.89

# 6. Unseen-country test vs mean baseline (split by country to avoid leakage)
rng = np.random.default_rng(42)
test_countries = rng.choice(data[COUNTRY].unique(), size=6, replace=False)
train = data[~data[COUNTRY].isin(test_countries)]
test = data[data[COUNTRY].isin(test_countries)]
w_tr, b_tr, _ = gradient_descent(train[DCAPB].to_numpy(), train["y"].to_numpy())
x_te, y_te = test[DCAPB].to_numpy(), test["y"].to_numpy()
mse_model = np.mean((w_tr * x_te + b_tr - y_te) ** 2)
mse_base = np.mean((train["y"].mean() - y_te) ** 2)
print(sorted(test_countries))
print(f"MSE model={mse_model:.2f} baseline={mse_base:.2f} improvement={100*(1-mse_model/mse_base):.1f} %")  # 30.34, 30.72, 1.2 %
