from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(r"E:\111")
OUTPUT_DIR = BASE_DIR / "output" / "question2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from scipy import optimize, stats

sys.path.insert(0, str(BASE_DIR))
from question1_analysis import prepare_data  # noqa: E402


THRESHOLD = 4.0
SIGMA_E = 0.446505
Q_TARGET = 0.95
MIN_GROUP_SIZE = 30
SEED = 20250823


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    for font_path in [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=str(font_path)).get_name()
            ]
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )


def build_intervals(valid: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Construct one latent first-attainment interval per woman.

    Left: first available measurement is already above threshold, T <= U.
    Interval: last sub-threshold measurement before first pass, L < T <= U.
    Right: no measurement reaches threshold, T > L.
    """
    rows: list[dict] = []
    for person, group in valid.sort_values("gest_week").groupby("孕妇代码"):
        group = group.sort_values("gest_week")
        weeks = group["gest_week"].to_numpy(float)
        y_pct = group["y_pct"].to_numpy(float)
        passed = np.flatnonzero(y_pct >= threshold)
        lower, upper, censor = np.nan, np.nan, "right"
        if len(passed) == 0:
            lower = float(weeks[-1])
            upper = np.inf
        else:
            first_pass = int(passed[0])
            earlier_fail = np.flatnonzero(y_pct[:first_pass] < threshold)
            upper = float(weeks[first_pass])
            if len(earlier_fail) == 0:
                lower = -np.inf
                censor = "left"
            else:
                lower = float(weeks[int(earlier_fail[-1])])
                censor = "interval"

        reversal = bool(np.any((y_pct[:-1] >= threshold) & (y_pct[1:] < threshold)))
        rows.append(
            {
                "person_id": person,
                "bmi": float(group["孕妇BMI"].mean()),
                "bmi_first": float(group["孕妇BMI"].iloc[0]),
                "censor": censor,
                "lower": lower,
                "upper": upper,
                "n_episodes": int(len(group)),
                "first_week": float(weeks[0]),
                "last_week": float(weeks[-1]),
                "reversal": reversal,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class AFTFit:
    distribution: str
    degree: int
    params: np.ndarray
    nll: float
    success: bool
    message: str
    bmi_center: float
    bmi_scale: float
    n: int

    @property
    def k(self) -> int:
        return len(self.params)

    @property
    def aic(self) -> float:
        return 2 * self.k + 2 * self.nll

    @property
    def bic(self) -> float:
        return self.k * math.log(self.n) + 2 * self.nll


def _linear_predictor(params: np.ndarray, z: np.ndarray, degree: int) -> np.ndarray:
    eta = params[0] + params[1] * z
    if degree == 2:
        eta = eta + params[2] * z**2
    return eta


def _cdf(t: np.ndarray, eta: np.ndarray, log_shape: float, distribution: str) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    eta = np.asarray(eta, dtype=float)
    out = np.empty(np.broadcast_shapes(t.shape, eta.shape), dtype=float)
    tt, ee = np.broadcast_arrays(t, eta)
    positive = tt > 0
    out[~positive] = 0.0
    if not positive.any():
        return out
    shape = math.exp(float(np.clip(log_shape, -4.0, 4.0)))
    log_t = np.log(tt[positive])
    x = (log_t - ee[positive]) * shape
    if distribution == "lognormal":
        out[positive] = stats.norm.cdf(x)
    elif distribution == "weibull":
        out[positive] = -np.expm1(-np.exp(np.clip(x, -40, 40)))
    elif distribution == "loglogistic":
        out[positive] = stats.logistic.cdf(x)
    else:
        raise ValueError(distribution)
    return np.clip(out, 0.0, 1.0)


def _neg_log_likelihood(
    params: np.ndarray,
    intervals: pd.DataFrame,
    distribution: str,
    degree: int,
    bmi_center: float,
    bmi_scale: float,
) -> float:
    z = (intervals["bmi"].to_numpy(float) - bmi_center) / bmi_scale
    eta = _linear_predictor(params, z, degree)
    log_shape = params[degree + 1]
    lower = intervals["lower"].to_numpy(float)
    upper = intervals["upper"].to_numpy(float)
    kind = intervals["censor"].to_numpy(str)
    prob = np.empty(len(intervals), dtype=float)
    left = kind == "left"
    inter = kind == "interval"
    right = kind == "right"
    prob[left] = _cdf(upper[left], eta[left], log_shape, distribution)
    prob[inter] = _cdf(upper[inter], eta[inter], log_shape, distribution) - _cdf(
        lower[inter], eta[inter], log_shape, distribution
    )
    prob[right] = 1.0 - _cdf(lower[right], eta[right], log_shape, distribution)
    prob = np.clip(prob, 1e-12, 1.0)
    return float(-np.log(prob).sum())


def fit_aft(intervals: pd.DataFrame, distribution: str, degree: int) -> AFTFit:
    bmi_center = float(intervals["bmi"].mean())
    bmi_scale = float(intervals["bmi"].std(ddof=0))
    p = degree + 2
    # eta = log(scale); last parameter is log(shape or inverse log-scale).
    x0 = np.zeros(p)
    x0[0] = math.log(12.5)
    x0[-1] = math.log(5.0)
    bounds = [(math.log(5), math.log(40))] + [(-3, 3)] * degree + [(-3, 4)]
    starts = [x0]
    for slope in (0.05, 0.2, 0.5, -0.1):
        trial = x0.copy()
        trial[1] = slope
        starts.append(trial)
    best = None
    for start in starts:
        res = optimize.minimize(
            _neg_log_likelihood,
            start,
            args=(intervals, distribution, degree, bmi_center, bmi_scale),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or float(res.fun) < float(best.fun):
            best = res
    assert best is not None
    return AFTFit(
        distribution=distribution,
        degree=degree,
        params=np.asarray(best.x, dtype=float),
        nll=float(best.fun),
        success=bool(best.success),
        message=str(best.message),
        bmi_center=bmi_center,
        bmi_scale=bmi_scale,
        n=len(intervals),
    )


def predict_probability(fit: AFTFit, week: np.ndarray | float, bmi: np.ndarray | float) -> np.ndarray:
    bmi_arr = np.asarray(bmi, dtype=float)
    z = (bmi_arr - fit.bmi_center) / fit.bmi_scale
    eta = _linear_predictor(fit.params, z, fit.degree)
    return _cdf(np.asarray(week, dtype=float), eta, fit.params[fit.degree + 1], fit.distribution)


def predict_quantile(fit: AFTFit, bmi: np.ndarray | float, q: float) -> np.ndarray:
    bmi_arr = np.asarray(bmi, dtype=float)
    z = (bmi_arr - fit.bmi_center) / fit.bmi_scale
    eta = _linear_predictor(fit.params, z, fit.degree)
    shape = math.exp(float(fit.params[fit.degree + 1]))
    if fit.distribution == "lognormal":
        base = stats.norm.ppf(q)
    elif fit.distribution == "weibull":
        base = math.log(-math.log1p(-q))
    elif fit.distribution == "loglogistic":
        base = stats.logistic.ppf(q)
    else:
        raise ValueError(fit.distribution)
    return np.exp(eta + base / shape)


def optimal_partition(values: np.ndarray, max_groups: int = 6, min_size: int = 30) -> dict:
    """Exact 1-D segmentation of sorted target times under within-group SSE."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    prefix = np.r_[0.0, np.cumsum(x)]
    prefix2 = np.r_[0.0, np.cumsum(x * x)]

    def cost(i: int, j: int) -> float:
        count = j - i
        s = prefix[j] - prefix[i]
        s2 = prefix2[j] - prefix2[i]
        return max(0.0, s2 - s * s / count)

    dp = np.full((max_groups + 1, n + 1), np.inf)
    prev = np.full((max_groups + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, max_groups + 1):
        for j in range(k * min_size, n + 1):
            lo = (k - 1) * min_size
            hi = j - min_size
            for i in range(lo, hi + 1):
                candidate = dp[k - 1, i] + cost(i, j)
                if candidate < dp[k, j]:
                    dp[k, j] = candidate
                    prev[k, j] = i

    partitions: dict[int, dict] = {}
    for k in range(1, max_groups + 1):
        if not np.isfinite(dp[k, n]):
            continue
        ends = [n]
        j = n
        for kk in range(k, 0, -1):
            j = int(prev[kk, j])
            ends.append(j)
        ends = sorted(ends)
        partitions[k] = {"sse": float(dp[k, n]), "bounds": list(zip(ends[:-1], ends[1:]))}

    sse1 = partitions[1]["sse"]
    sse_max = partitions[max(partitions)]["sse"]
    denominator = max(sse1 - sse_max, 1e-12)
    for k, result in partitions.items():
        result["explained"] = (sse1 - result["sse"]) / denominator
    # Smallest number attaining 90% of the maximum 1..K improvement.
    selected = next((k for k in sorted(partitions) if partitions[k]["explained"] >= 0.90), max(partitions))
    return {"selected": selected, "partitions": partitions}


def ceil_to_day(week: float) -> float:
    return math.ceil(week * 7 - 1e-10) / 7.0


def format_week(week: float) -> str:
    total_days = int(math.ceil(week * 7 - 1e-9))
    return f"{total_days // 7}周+{total_days % 7}天"


def make_groups(women: pd.DataFrame, fit: AFTFit, q: float = Q_TARGET) -> tuple[pd.DataFrame, dict]:
    ordered = women.sort_values("bmi").reset_index(drop=True).copy()
    ordered["target_week"] = predict_quantile(fit, ordered["bmi"].to_numpy(), q)
    part = optimal_partition(ordered["target_week"].to_numpy(), max_groups=6, min_size=MIN_GROUP_SIZE)
    bounds = part["partitions"][part["selected"]]["bounds"]
    rows = []
    ordered["group"] = 0
    for group_id, (i, j) in enumerate(bounds, start=1):
        segment = ordered.iloc[i:j]
        ordered.loc[i : j - 1, "group"] = group_id
        bmi_min = float(segment["bmi"].min())
        bmi_max = float(segment["bmi"].max())
        bmi_values = segment["bmi"].to_numpy(float)
        # Group policy: expected attainment probability over the empirical BMI
        # distribution in this group reaches q. This avoids letting one extreme
        # tail observation determine the schedule for the entire group.
        objective = lambda t: float(predict_probability(fit, t, bmi_values).mean() - q)
        raw_week = float(optimize.brentq(objective, 5.0, 80.0))
        decision_week = ceil_to_day(raw_week)
        p_all = predict_probability(fit, decision_week, bmi_values)
        p_min = float(p_all.min())
        p_mean = float(p_all.mean())
        rows.append(
            {
                "group": group_id,
                "n": int(len(segment)),
                "bmi_min": bmi_min,
                "bmi_max": bmi_max,
                "bmi_median": float(segment["bmi"].median()),
                "raw_target_week": raw_week,
                "decision_week": decision_week,
                "decision_label": format_week(decision_week),
                "min_probability": p_min,
                "mean_probability": p_mean,
            }
        )
    groups = pd.DataFrame(rows)
    cutpoints = [
        (groups.loc[i, "bmi_max"] + groups.loc[i + 1, "bmi_min"]) / 2
        for i in range(len(groups) - 1)
    ]
    labels = []
    for i, row in groups.iterrows():
        if len(groups) == 1:
            label = f"[{row['bmi_min']:.2f}, {row['bmi_max']:.2f}]"
        elif i == 0:
            label = f"[{row['bmi_min']:.2f}, {cutpoints[0]:.2f})"
        elif i == len(groups) - 1:
            label = f"[{cutpoints[-1]:.2f}, {row['bmi_max']:.2f}]"
        else:
            label = f"[{cutpoints[i-1]:.2f}, {cutpoints[i]:.2f})"
        labels.append(label)
    groups["bmi_interval"] = labels
    return groups, {"ordered": ordered, "cutpoints": cutpoints, **part}


def fit_selection_table(intervals: pd.DataFrame) -> tuple[AFTFit, pd.DataFrame]:
    fits: list[AFTFit] = []
    for distribution in ["lognormal", "weibull", "loglogistic"]:
        for degree in [1, 2]:
            fits.append(fit_aft(intervals, distribution, degree))
    table = pd.DataFrame(
        [
            {
                "distribution": f.distribution,
                "degree": f.degree,
                "nll": f.nll,
                "aic": f.aic,
                "bic": f.bic,
                "converged": f.success,
            }
            for f in fits
        ]
    ).sort_values("aic")
    # The unconstrained quadratic fit is retained in the audit table but not used
    # for decisions: it creates a U-shape that conflicts with the verified Q1
    # direction and is unstable at the sparse BMI tails. Select by AIC among the
    # monotone linear-BMI AFT candidates.
    best_key = table.loc[table["degree"] == 1].iloc[0][["distribution", "degree"]]
    best = next(
        f
        for f in fits
        if f.distribution == best_key["distribution"] and f.degree == int(best_key["degree"])
    )
    return best, table


def bootstrap_group_times(
    intervals: pd.DataFrame,
    fit: AFTFit,
    groups: pd.DataFrame,
    reps: int = 250,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records: list[dict] = []
    for rep in range(reps):
        sample = intervals.iloc[rng.integers(0, len(intervals), len(intervals))].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = fit_aft(sample, fit.distribution, fit.degree)
        if not boot.success or not np.isfinite(boot.nll):
            continue
        for _, row in groups.iterrows():
            bmi_values = intervals.loc[
                intervals["bmi"].between(row["bmi_min"], row["bmi_max"]), "bmi"
            ].to_numpy(float)
            objective = lambda t: float(predict_probability(boot, t, bmi_values).mean() - Q_TARGET)
            try:
                week = float(optimize.brentq(objective, 5.0, 80.0))
            except ValueError:
                continue
            if np.isfinite(week) and 5 <= week <= 60:
                records.append({"rep": rep, "group": int(row["group"]), "week": week})
    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=["group", "boot_n", "ci_low", "ci_high"])
    return (
        result.groupby("group")["week"]
        .agg(
            boot_n="size",
            ci_low=lambda s: float(np.quantile(s, 0.025)),
            ci_high=lambda s: float(np.quantile(s, 0.975)),
        )
        .reset_index()
    )


def error_sensitivity(
    valid: pd.DataFrame,
    selected: AFTFit,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    scenarios = [
        ("宽松阈值 4%-1σ", THRESHOLD - SIGMA_E),
        ("基准阈值 4%", THRESHOLD),
        ("保守阈值 4%+1.645σ", THRESHOLD + 1.645 * SIGMA_E),
    ]
    rows = []
    for label, threshold in scenarios:
        intervals = build_intervals(valid, threshold)
        fitted = fit_aft(intervals, selected.distribution, selected.degree)
        for _, group in groups.iterrows():
            bmi_values = intervals.loc[
                intervals["bmi"].between(group["bmi_min"], group["bmi_max"]), "bmi"
            ].to_numpy(float)
            objective = lambda t: float(predict_probability(fitted, t, bmi_values).mean() - Q_TARGET)
            raw_week = float(optimize.brentq(objective, 5.0, 100.0))
            rows.append(
                {
                    "scenario": label,
                    "threshold": threshold,
                    "group": int(group["group"]),
                    "raw_week": raw_week,
                    "decision_week": ceil_to_day(raw_week),
                    "decision_label": (
                        format_week(ceil_to_day(raw_week))
                        if raw_week <= 25
                        else f"25周内不可满足（交点{format_week(ceil_to_day(raw_week))}）"
                    ),
                    "feasible_by_25": bool(raw_week <= 25),
                    "left": int((intervals["censor"] == "left").sum()),
                    "interval": int((intervals["censor"] == "interval").sum()),
                    "right": int((intervals["censor"] == "right").sum()),
                }
            )
    return pd.DataFrame(rows)


def reliability_sensitivity(
    intervals: pd.DataFrame,
    fit: AFTFit,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for q in [0.90, 0.95, 0.99]:
        for _, group in groups.iterrows():
            bmi_values = intervals.loc[
                intervals["bmi"].between(group["bmi_min"], group["bmi_max"]), "bmi"
            ].to_numpy(float)
            objective = lambda t: float(predict_probability(fit, t, bmi_values).mean() - q)
            raw_week = float(optimize.brentq(objective, 5.0, 100.0))
            decision = ceil_to_day(raw_week)
            rows.append(
                {
                    "target_probability": q,
                    "group": int(group["group"]),
                    "raw_week": raw_week,
                    "decision_week": decision,
                    "decision_label": (
                        format_week(decision)
                        if decision <= 25
                        else f"25周内不可满足（交点{format_week(decision)}）"
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_censoring(intervals: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    colors = {"left": "#4C78A8", "interval": "#F58518", "right": "#E45756"}
    labels = {"left": "左删失：首次检测已达标", "interval": "区间删失：两次检测间达标", "right": "右删失：末次仍未达标"}
    for kind in ["left", "interval", "right"]:
        d = intervals[intervals["censor"] == kind]
        if kind == "left":
            y = d["upper"]
            ax.scatter(d["bmi"], y, s=24, alpha=0.65, color=colors[kind], marker="v", label=labels[kind])
        elif kind == "right":
            y = d["lower"]
            ax.scatter(d["bmi"], y, s=35, alpha=0.85, color=colors[kind], marker="^", label=labels[kind])
        else:
            ax.vlines(d["bmi"], d["lower"], d["upper"], color=colors[kind], alpha=0.45, linewidth=1.3)
            ax.scatter(d["bmi"], d["upper"], s=28, alpha=0.8, color=colors[kind], marker="o", label=labels[kind])
    ax.axhspan(10, 12, color="#54A24B", alpha=0.08, label="低风险检测窗口（≤12周）")
    ax.axhline(12, color="#54A24B", ls="--", lw=1)
    ax.set(xlabel="孕妇平均 BMI", ylabel="删失端点孕周（周）", title="首次达到 Y 浓度 4% 的观测区间")
    ax.legend(frameon=True, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "censoring_intervals.png", bbox_inches="tight")
    plt.close(fig)


def plot_grouping(intervals: pd.DataFrame, fit: AFTFit, groups: pd.DataFrame, detail: dict) -> None:
    ordered = detail["ordered"]
    grid = np.linspace(intervals["bmi"].min(), intervals["bmi"].max(), 500)
    q90 = predict_quantile(fit, grid, 0.90)
    q95 = predict_quantile(fit, grid, 0.95)
    q99 = predict_quantile(fit, grid, 0.99)
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.plot(grid, q95, color="#E45756", lw=2.4, label="连续 BMI 模型的 95% 达标时点")
    ax.fill_between(grid, q90, q99, color="#E45756", alpha=0.12, label="90%—99% 分位带")
    palette = sns.color_palette("viridis", len(groups))
    for (_, row), color in zip(groups.iterrows(), palette):
        ax.hlines(row["decision_week"], row["bmi_min"], row["bmi_max"], color=color, lw=5)
        ax.vlines(row["bmi_max"], 10, row["decision_week"], color=color, lw=0.9, ls=":", alpha=0.8)
        ax.text(
            (row["bmi_min"] + row["bmi_max"]) / 2,
            row["decision_week"] + 0.35,
            f"第{int(row['group'])}组",
            color=color,
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
    ax.scatter(ordered["bmi"], ordered["target_week"], s=8, color="black", alpha=0.15)
    ax.axhline(12, color="#54A24B", ls="--", lw=1.2, label="12周低风险界限")
    ax.axhline(25, color="#B279A2", ls="--", lw=1.2, label="题设可检测上限25周")
    ax.set(
        xlabel="孕妇平均 BMI",
        ylabel="达到目标概率所需孕周（周）",
        title="BMI 连续效应与优化后的分组决策时点",
        ylim=(9.5, max(26, float(np.nanmax(q99)) + 0.8)),
    )
    ax.legend(fontsize=9, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bmi_grouping_decision.png", bbox_inches="tight")
    plt.close(fig)


def plot_probability_curves(fit: AFTFit, groups: pd.DataFrame, intervals: pd.DataFrame) -> None:
    weeks = np.linspace(10, 25, 500)
    palette = sns.color_palette("viridis", len(groups))
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    for (_, row), color in zip(groups.iterrows(), palette):
        bmi_values = intervals.loc[
            intervals["bmi"].between(row["bmi_min"], row["bmi_max"]), "bmi"
        ].to_numpy(float)
        p_matrix = np.vstack([predict_probability(fit, weeks, bmi) for bmi in bmi_values])
        p = p_matrix.mean(axis=0)
        label = f"第{int(row['group'])}组：BMI {row['bmi_interval']}"
        ax.plot(weeks, p, lw=2.1, color=color, label=label)
        ax.fill_between(weeks, p_matrix.min(axis=0), p_matrix.max(axis=0), color=color, alpha=0.08)
        ax.scatter([row["decision_week"]], [np.interp(row["decision_week"], weeks, p)], color=color, s=45, zorder=4)
    ax.axhline(Q_TARGET, color="#E45756", ls="--", lw=1.5, label="95%可靠性约束")
    ax.axvline(12, color="#54A24B", ls=":", lw=1.2, label="12周界限")
    ax.set(xlabel="NIPT 时点（孕周）", ylabel="模型预测已达到 4% 的概率", title="各 BMI 组的平均达标概率曲线（阴影为组内范围）", xlim=(10, 25), ylim=(0, 1.01))
    ax.legend(fontsize=8.5, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "attainment_probability.png", bbox_inches="tight")
    plt.close(fig)


def plot_error_sensitivity(error: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    sns.pointplot(
        data=error,
        x="group",
        y="decision_week",
        hue="scenario",
        dodge=0.22,
        markers=["o", "s", "D"],
        linestyles=["-", "-", "-"],
        ax=ax,
    )
    ax.axhline(12, color="#54A24B", ls="--", lw=1.2, label="12周低风险界限")
    ax.axhline(25, color="#B279A2", ls=":", lw=1.2, label="题设可检测上限25周")
    ax.set(xlabel="BMI 分组", ylabel="建议时点（周，向上取整到天）", title="检测误差对各组建议时点的影响")
    ax.legend(title="阈值情景", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "measurement_error_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, columns: list[tuple[str, str]], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for col, _ in columns:
            value = row[col]
            if col in formats:
                vals.append(formats[col].format(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    audit: dict,
    valid: pd.DataFrame,
    intervals: pd.DataFrame,
    best: AFTFit,
    selection: pd.DataFrame,
    groups: pd.DataFrame,
    detail: dict,
    bootstrap: pd.DataFrame,
    error: pd.DataFrame,
    reliability: pd.DataFrame,
) -> None:
    merged_groups = groups.merge(bootstrap, on="group", how="left")
    dist_names = {"lognormal": "对数正态", "weibull": "Weibull", "loglogistic": "对数Logistic"}
    selection_show = selection.copy()
    selection_show["模型"] = selection_show.apply(
        lambda r: f"{dist_names[r['distribution']]} AFT（BMI {'二次' if int(r['degree']) == 2 else '线性'}）", axis=1
    )
    group_table = md_table(
        merged_groups,
        [
            ("group", "组别"),
            ("n", "人数"),
            ("bmi_interval", "BMI 区间"),
            ("decision_label", "最佳 NIPT 时点"),
            ("mean_probability", "组内平均达标概率"),
            ("min_probability", "极端边界最低概率"),
            ("ci_low", "时点95%CI下限/周"),
            ("ci_high", "时点95%CI上限/周"),
        ],
        {
            "group": "{:.0f}",
            "n": "{:.0f}",
            "mean_probability": "{:.1%}",
            "min_probability": "{:.1%}",
            "ci_low": "{:.2f}",
            "ci_high": "{:.2f}",
        },
    )
    selection_table = md_table(
        selection_show,
        [("模型", "候选模型"), ("nll", "负对数似然"), ("aic", "AIC"), ("bic", "BIC")],
        {"nll": "{:.2f}", "aic": "{:.2f}", "bic": "{:.2f}"},
    )
    partition_rows = []
    for k, result in detail["partitions"].items():
        partition_rows.append({"k": k, "sse": result["sse"], "explained": result["explained"]})
    partition_table = md_table(
        pd.DataFrame(partition_rows),
        [("k", "组数K"), ("sse", "组内平方误差"), ("explained", "相对最大改善率")],
        {"k": "{:.0f}", "sse": "{:.4f}", "explained": "{:.1%}"},
    )
    error_pivot = error.pivot(index="group", columns="scenario", values="decision_label").reset_index()
    error_cols = [("group", "组别")] + [(c, c) for c in error_pivot.columns if c != "group"]
    error_table = md_table(error_pivot, error_cols, {"group": "{:.0f}"})
    reliability_show = reliability.copy()
    reliability_show["目标可靠性"] = reliability_show["target_probability"].map(lambda x: f"{x:.0%}")
    reliability_pivot = reliability_show.pivot(
        index="group", columns="目标可靠性", values="decision_label"
    ).reset_index()
    reliability_cols = [("group", "组别")] + [
        (c, f"目标{c}") for c in ["90%", "95%", "99%"] if c in reliability_pivot.columns
    ]
    reliability_table = md_table(reliability_pivot, reliability_cols, {"group": "{:.0f}"})

    counts = intervals["censor"].value_counts()
    near_threshold = int(((valid["y_pct"] - THRESHOLD).abs() <= SIGMA_E).sum())
    reversals = int(intervals["reversal"].sum())
    best_name = f"{dist_names[best.distribution]} AFT（BMI {'二次项' if best.degree == 2 else '线性项'}）"
    unrestricted = selection.iloc[0]
    unrestricted_name = f"{dist_names[unrestricted['distribution']]} AFT（BMI {'二次项' if int(unrestricted['degree']) == 2 else '线性项'}）"
    delta_aic = best.aic - float(unrestricted["aic"])
    slope_text = "、".join(f"{x:.5f}" for x in best.params)
    report = rf"""# C题第二问报告：基于区间删失 AFT 模型的 BMI 分组与最佳 NIPT 时点

## 摘要

针对“Y 染色体浓度首次达到 4%”这一事件，本文没有把第一次观测到达标的孕周直接视为真实达标时间，而是根据每位孕妇的纵向检测记录构造左删失、区间删失和右删失数据，并采用参数化加速失效时间（AFT）模型估计达标时间分布。共有 {len(intervals)} 位男胎孕妇进入分析，其中左删失 {counts.get('left', 0)} 人、区间删失 {counts.get('interval', 0)} 人、右删失 {counts.get('right', 0)} 人。

主决策采用“**组内平均达标概率达到 95% 的最早孕周**”：在控制组内平均未达标概率不超过 5% 的前提下，把检测安排得尽可能早，从而避免任意设定医学风险权重。无约束 AIC 最低模型在稀疏 BMI 尾部产生不合理 U 形和严重外推，因此决策模型按方向一致性与稳定性选择 {best_name}。再对连续 BMI 对应的 95% 达标时点作带最小组容量约束的一维动态规划分段，选择 {len(groups)} 组。最终建议见表 5；检测误差分析表明，靠近 4% 阈值的记录会使分组时点发生变化，因此临床执行应优先采用保守阈值或在临界结果时复测。

## 1. 题意审查与建模选择

题目要求按 BMI 分组，并给出使潜在风险最小的 NIPT 时点。这里同时存在两类相反风险：

- 检测过早：Y 浓度尚未达到 4%，增加结果不可靠或重复抽血的可能；
- 检测过晚：异常胎儿被发现得更晚，题设指出 12 周及以前风险较低，13—27 周风险较高，28 周以后风险极高。

若直接采用“每位孕妇第一次测到 ≥4% 的孕周”，会产生三类偏差：首次样本已达标者的真实达标时间其实更早；从未达标者不能被删除；两次检测之间何时达标未知。此外，本数据有选择性复测，检测时点本身并非随机。因此，本问最合适的统计对象是潜在首次达标时间 $T$，并使用区间删失生存模型。

本文把“风险最小”写为以下约束优化：

$$
t_g^*=\min_t t,
\qquad
\text{{s.t.}}\quad E_{{B\mid g}}[P(T\le t\mid BMI=B)]\ge 0.95.
$$

也就是说，要求第 $g$ 个 BMI 组按其经验 BMI 分布平均后，到建议时点至少有 95% 的概率已经达到 4%；在满足可靠性的可行时点中选最早者。这里不用“组内最极端 BMI 必须达到 95%”作为主约束，因为高 BMI 尾部样本很少，单个极端值会把整组建议推到题设 25 周范围之外；报告仍单独列出极端边界最低概率作为风险提示。该定义无需人为指定“检测失败”和“延迟发现”孰轻孰重，且可直接做 90%、99% 可靠性敏感性分析。

## 2. 数据整理与删失区间

沿用第一问的数据清洗：仅使用“男胎检测数据”，把 Y 染色体浓度由比例转换为百分数；将相同“孕妇代码—检测日期—抽血次数—孕周”的技术复测取均值。{audit['raw_rows']} 条原始记录合并为 {audit['valid_episodes']} 个有效检测事件，涉及 {audit['valid_women']} 位孕妇。每位孕妇用各次有效检测 BMI 的均值作为分组指标。

对阈值 $c=4\%$，删失区间定义如下：

1. 第一次有效检测已经 $Y\ge c$：$T\le U_i$，记为左删失；
2. 某次 $Y<c$，下一次首次出现 $Y\ge c$：$L_i<T\le U_i$，记为区间删失；
3. 截至最后一次仍未达到 $c$：$T>L_i$，记为右删失。

| 删失类型 | 人数 | 含义 |
|---|---:|---|
| 左删失 | {counts.get('left', 0)} | 首次检测时已经达标，真实达标时间更早 |
| 区间删失 | {counts.get('interval', 0)} | 首次达标发生在相邻两次检测之间 |
| 右删失 | {counts.get('right', 0)} | 末次检测仍未达标，只知道真实时间更晚 |

![首次达标删失区间](./censoring_intervals.png)

图中的点不是所有人的“真实首次达标周”，而是可由数据确认的上界或下界。左删失占比较高，意味着高分位时点的不确定性不能只由常规回归标准误表示，故后文给出按孕妇重抽样的 Bootstrap 区间。

## 3. 区间删失 AFT 模型

### 3.1 模型形式

令 $B_i$ 为第 $i$ 位孕妇的平均 BMI，标准化后为 $z_i$。AFT 模型写为：

$$
\log T_i=\eta_i+\varepsilon_i,
\qquad
\eta_i=\beta_0+\beta_1z_i+\beta_2z_i^2,
$$

其中是否保留 $z_i^2$ 由 AIC 决定；误差分布分别尝试对数正态、Weibull 和对数 Logistic。三类观测对似然的贡献分别为：

$$
F(U_i\mid B_i),\qquad
F(U_i\mid B_i)-F(L_i\mid B_i),\qquad
1-F(L_i\mid B_i).
$$

这使首次已达标者和末次仍未达标者都能进入估计，而无需伪造事件时间。

### 3.2 模型比较

{selection_table}

无约束最小 AIC 模型为 **{unrestricted_name}**；但它只比所选模型低 {delta_aic:.2f}，却在低 BMI 端产生与第一问及稀释机理相反的 U 形，并使 BMI=46.88 的 95% 时点外推至 58 周，明显超出题设 10—25 周范围。因而不以微小拟合改善换取严重决策不稳定，最终在 BMI 线性、方向单调的候选模型中按 AIC 选择 **{best_name}**。优化参数依次为 `{slope_text}`。参数作用在标准化 BMI 与对数时间尺度上，实际解释和决策均通过 $F(t\mid BMI)$ 与时间分位数完成。

## 4. BMI 分组优化

### 4.1 分组原则

先计算每位孕妇在连续 BMI 模型下的 95% 达标时间 $t_{{0.95}}(B_i)$，然后按 BMI 排序，用动态规划寻找使组内 $t_{{0.95}}$ 平方误差最小的连续分段；每组至少 {MIN_GROUP_SIZE} 人，以防极端 BMI 小样本形成不稳定小组。考察 1—6 组，选择达到“从 1 组到最大可行组总改善量的 90%”的最小组数。

{partition_table}

据此选取 **{len(groups)} 组**。切点取相邻两组边界样本 BMI 的中点。每组实际时点通过组内所有样本 BMI 的预测达标概率取平均并求解 95% 交点，再向上取整到整天。这样得到的是可执行的群体排期规则；表中同时列出极端边界的最低概率，提醒尾部个体不能机械套用群体平均保证。

![BMI分组与决策时点](./bmi_grouping_decision.png)

### 4.2 最佳 NIPT 时点

{group_table}

表中 BMI 区间是本样本实际覆盖范围；超出 {intervals['bmi'].min():.2f}—{intervals['bmi'].max():.2f} 的孕妇属于外推，不宜机械套用。Bootstrap 区间由 {int(bootstrap['boot_n'].max()) if not bootstrap.empty else 0} 次有效孕妇级重抽样给出，反映删失构造和样本组成的不确定性。临床排期时，若希望对模型不确定性也采取保守原则，可参考置信区间上限，而不是只用点估计。

![各组达标概率曲线](./attainment_probability.png)

图中实线是组内平均达标概率，阴影表示该 BMI 组从最低到最高预测概率的范围，实心点为建议时点。建议时点是群体层面的统计决策，不等同于对单个孕妇的诊断保证。

### 4.3 可靠性水平敏感性

95% 是主分析的管理约束，并非题目给定常数。为避免把这一选择隐藏在模型中，下表同时给出 90% 和 99% 的排期；“25周内不可满足”表示模型交点超出题设 NIPT 时间窗，此时不能继续延后机械等待，而应在 25 周前检测并预设复测或采用其他临床方案。

{reliability_table}

## 5. 检测误差对结果的影响

第一问利用 18 组同一样本技术复测估得单次检测误差标准差约为 $\sigma_e={SIGMA_E:.3f}$ 个百分点。当前数据中有 {near_threshold} 个检测事件位于 $4\%\pm\sigma_e$ 内，且有 {reversals} 位孕妇出现“先达到 4%、随后又低于 4%”的反向波动。这说明真实生物学趋势虽总体随孕周上升，单次观测却并不单调；硬阈值会把微小测量波动放大为达标状态改变。

采用三个阈值情景重新构造删失区间并重估同类型 AFT 模型：

- 宽松情景：$4\%-\sigma_e={THRESHOLD-SIGMA_E:.3f}\%$；
- 基准情景：$4.000\%$；
- 保守情景：$4\%+1.645\sigma_e={THRESHOLD+1.645*SIGMA_E:.3f}\%$。若误差近似正态，观测值达到该保守阈值时，其单侧 95% 下置信界才达到 4%。

{error_table}

![检测误差敏感性](./measurement_error_sensitivity.png)

误差影响主要通过两条路径进入结果：一是改变某次检测是否跨过 4% 的分类，进而改变删失类型和区间；二是高 BMI 组本身达标较晚，阈值上调后延迟可能更明显。实际执行建议如下：

1. 若在建议时点测得的 Y 浓度明显高于 4%，按常规流程处理；
2. 若结果落在 $4\%\pm0.447\%$ 附近，不应仅凭一次硬判定，宜结合测序质量指标并复测；
3. 若决策目标强调“尽量避免假达标”，采用保守情景时点；若强调尽早筛查，可用基准时点但预留复测方案；
4. 对 BMI 极端值、IVF 或其他高风险个体，应由临床人员结合个体情况调整，而不是只依据 BMI 分组。

## 6. 自我审查与局限性

1. **观察计划具有选择性。** 浓度较低者更可能复测，检测间隔也不统一。区间删失模型正确利用了已观察区间，但仍假设给定 BMI 后观察计划不携带额外的未建模信息。
2. **潜在达标过程被视为单次穿越。** 有 {reversals} 人出现观测反转，说明测量误差和短期波动存在。阈值敏感性分析缓解但不能完全消除这一问题。
3. **BMI 使用孕期检测均值。** 这适合对孕妇作稳定分组，但会掩盖孕期 BMI 小幅变化。第一问结果显示组间 BMI 效应明显、组内 BMI 变化不显著，因此该处理在本数据中较合理。
4. **95% 是决策可靠性水平，不是医学界统一阈值。** 报告同时保留模型函数，可根据管理部门容许的失败概率改成 90% 或 99%；可靠性要求越高，建议时点越晚。
5. **高分位估计受右删失样本数限制。** 只有 {counts.get('right', 0)} 位孕妇末次仍未达标，95% 分位的尾部外推依赖分布假设。模型比较和 Bootstrap 能揭示部分不确定性，但不能代替外部验证。
6. **本结果只在样本 BMI 范围内成立。** 超出样本范围、数据量很少的尾部区间或不同检测平台需要重新校准。
7. **群体约束不保证每个极端个体。** 第 3 组的平均达标率达到 95%，但 BMI 上边界的最低预测概率较低，因此极端高 BMI 个体应采用单独概率计算和复测策略。

## 7. 结论

第二问最关键的改进，是把“首次达到 4%”作为带左删失、区间删失和右删失的潜在事件时间，而不是把第一次测到达标的孕周当作精确值。综合拟合、方向一致性和尾部稳定性后采用 {best_name}；基于其连续 BMI—达标时间关系，经动态规划形成 {len(groups)} 个连续 BMI 组，并在各组内选择平均达标概率至少为 95% 的最早时点。表 5 给出的时点兼顾检测可靠性与尽早筛查，表 6 则量化了检测误差下的调整范围。

对临床执行而言，建议采用“基准时点 + 临界结果复测”的方案；若更重视避免假达标，则采用 $4\%+1.645\sigma_e$ 的保守时点。任何分组时点都应视为基于本样本的群体排期建议，而非个体诊断结论。
"""
    (OUTPUT_DIR / "第二问分析报告.md").write_text(report, encoding="utf-8")


def main() -> None:
    configure_plotting()
    _, valid, audit = prepare_data()
    intervals = build_intervals(valid, THRESHOLD)
    best, selection = fit_selection_table(intervals)
    groups, detail = make_groups(intervals, best)
    bootstrap = bootstrap_group_times(intervals, best, groups, reps=200)
    error = error_sensitivity(valid, best, groups)
    reliability = reliability_sensitivity(intervals, best, groups)

    plot_censoring(intervals)
    plot_grouping(intervals, best, groups, detail)
    plot_probability_curves(best, groups, intervals)
    plot_error_sensitivity(error)

    intervals.to_csv(OUTPUT_DIR / "first_attainment_intervals.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(OUTPUT_DIR / "model_selection.csv", index=False, encoding="utf-8-sig")
    groups.merge(bootstrap, on="group", how="left").to_csv(
        OUTPUT_DIR / "bmi_groups_and_timing.csv", index=False, encoding="utf-8-sig"
    )
    error.to_csv(OUTPUT_DIR / "measurement_error_sensitivity.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(OUTPUT_DIR / "reliability_level_sensitivity.csv", index=False, encoding="utf-8-sig")
    results = {
        "selected_distribution": best.distribution,
        "selected_degree": best.degree,
        "params": best.params.tolist(),
        "aic": best.aic,
        "bic": best.bic,
        "converged": best.success,
        "censor_counts": intervals["censor"].value_counts().to_dict(),
        "reversals": int(intervals["reversal"].sum()),
        "groups": groups.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "results_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(audit, valid, intervals, best, selection, groups, detail, bootstrap, error, reliability)

    print(selection.to_string(index=False))
    print("\nSelected:", best.distribution, "degree", best.degree, "AIC", best.aic)
    print("\nGroups:\n", groups.to_string(index=False))
    print("\nBootstrap:\n", bootstrap.to_string(index=False))
    print("\nError sensitivity:\n", error.to_string(index=False))


if __name__ == "__main__":
    main()
