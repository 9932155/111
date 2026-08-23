from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path


BASE_DIR = Path(r"E:\111")
OUTPUT_DIR = BASE_DIR / "output" / "question1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import font_manager
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess


SOURCE_XLSX = Path(r"E:\SvpohSGacdffe718bcaa3b6e835c03ae3461cab1 (1)\C题\附件.xlsx")
SHEET = "男胎检测数据"


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    font_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [family]
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


def parse_gestational_week(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)w(?:\+(\d+))?\s*", str(value), flags=re.IGNORECASE)
    if not match:
        return np.nan
    weeks = int(match.group(1))
    days = int(match.group(2) or 0)
    if not 0 <= days <= 6:
        return np.nan
    return weeks + days / 7.0


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_excel(SOURCE_XLSX, sheet_name=SHEET)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw["gest_week"] = raw["检测孕周"].map(parse_gestational_week)

    numeric_columns = [
        "年龄",
        "身高",
        "体重",
        "孕妇BMI",
        "Y染色体浓度",
        "GC含量",
        "原始读段数",
        "在参考基因组上比对的比例",
        "重复读段的比例",
        "唯一比对的读段数",
        "被过滤掉读段数的比例",
        "怀孕次数",
        "生产次数",
    ]
    for col in numeric_columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["检测日期"] = pd.to_datetime(raw["检测日期"], errors="coerce")
    raw["y_pct"] = raw["Y染色体浓度"] * 100.0

    episode_keys = ["孕妇代码", "检测日期", "检测抽血次数", "gest_week"]
    numeric_for_episode = [
        "年龄",
        "身高",
        "体重",
        "孕妇BMI",
        "Y染色体浓度",
        "y_pct",
        "GC含量",
        "原始读段数",
        "在参考基因组上比对的比例",
        "重复读段的比例",
        "唯一比对的读段数",
        "被过滤掉读段数的比例",
        "怀孕次数",
        "生产次数",
    ]
    episode = (
        raw.groupby(episode_keys, dropna=False, as_index=False)[numeric_for_episode]
        .mean(numeric_only=True)
        .merge(
            raw.groupby(episode_keys, dropna=False, as_index=False)
            .agg(technical_replicates=("序号", "size"), IVF妊娠=("IVF妊娠", "first")),
            on=episode_keys,
            how="left",
        )
    )
    valid = episode.dropna(subset=["孕妇代码", "gest_week", "孕妇BMI", "y_pct"]).copy()
    valid["log_raw_reads"] = np.log(valid["原始读段数"].clip(lower=1))

    for col, short in [
        ("gest_week", "gest_week"),
        ("孕妇BMI", "bmi"),
        ("年龄", "age"),
        ("GC含量", "gc"),
        ("log_raw_reads", "log_reads"),
        ("在参考基因组上比对的比例", "map_ratio"),
        ("重复读段的比例", "dup_ratio"),
        ("被过滤掉读段数的比例", "filter_ratio"),
    ]:
        mean_value = valid[col].mean()
        valid[f"{short}_c"] = valid[col] - mean_value

    audit = {
        "raw_rows": int(len(raw)),
        "raw_women": int(raw["孕妇代码"].nunique()),
        "raw_missing_y": int(raw["Y染色体浓度"].isna().sum()),
        "raw_invalid_gest_week": int(raw["gest_week"].isna().sum()),
        "episodes": int(len(episode)),
        "episodes_with_technical_replicates": int((episode["technical_replicates"] > 1).sum()),
        "max_technical_replicates": int(episode["technical_replicates"].max()),
        "valid_episodes": int(len(valid)),
        "valid_women": int(valid["孕妇代码"].nunique()),
        "women_with_multiple_valid_episodes": int(
            (valid.groupby("孕妇代码").size() > 1).sum()
        ),
        "gest_week_mean": float(valid["gest_week"].mean()),
        "gest_week_sd": float(valid["gest_week"].std()),
        "gest_week_min": float(valid["gest_week"].min()),
        "gest_week_max": float(valid["gest_week"].max()),
        "bmi_mean": float(valid["孕妇BMI"].mean()),
        "bmi_sd": float(valid["孕妇BMI"].std()),
        "bmi_min": float(valid["孕妇BMI"].min()),
        "bmi_max": float(valid["孕妇BMI"].max()),
        "y_pct_mean": float(valid["y_pct"].mean()),
        "y_pct_sd": float(valid["y_pct"].std()),
        "y_pct_min": float(valid["y_pct"].min()),
        "y_pct_max": float(valid["y_pct"].max()),
        "y_pct_negative": int((valid["y_pct"] < 0).sum()),
    }
    return raw, valid, audit


def correlation_table(valid: pd.DataFrame) -> pd.DataFrame:
    variables = {
        "孕周（周）": "gest_week",
        "BMI": "孕妇BMI",
        "年龄（岁）": "年龄",
        "身高（cm）": "身高",
        "体重（kg）": "体重",
        "GC含量": "GC含量",
        "原始读段数（对数）": "log_raw_reads",
        "参考基因组比对比例": "在参考基因组上比对的比例",
        "重复读段比例": "重复读段的比例",
        "过滤读段比例": "被过滤掉读段数的比例",
        "怀孕次数": "怀孕次数",
        "生产次数": "生产次数",
    }
    rows = []
    for label, col in variables.items():
        subset = valid[[col, "y_pct"]].dropna()
        if subset[col].nunique() < 2:
            continue
        pearson_r, pearson_p = stats.pearsonr(subset[col], subset["y_pct"])
        spearman_r, spearman_p = stats.spearmanr(subset[col], subset["y_pct"])
        rows.append(
            {
                "指标": label,
                "n": len(subset),
                "Pearson_r": pearson_r,
                "Pearson_p": pearson_p,
                "Spearman_rho": spearman_r,
                "Spearman_p": spearman_p,
            }
        )
    result = pd.DataFrame(rows)
    result["abs_Pearson_r"] = result["Pearson_r"].abs()
    return result.sort_values("abs_Pearson_r", ascending=False).drop(columns="abs_Pearson_r")


def fit_mixed_models(valid: pd.DataFrame):
    formulas = {
        "M0_空模型": "y_pct ~ 1",
        "M1_线性": "y_pct ~ gest_week_c + bmi_c",
        "M2_孕周二次": "y_pct ~ gest_week_c + I(gest_week_c ** 2) + bmi_c",
        "M3_双二次": "y_pct ~ gest_week_c + I(gest_week_c ** 2) + bmi_c + I(bmi_c ** 2)",
        "M4_交互": "y_pct ~ gest_week_c + bmi_c + gest_week_c:bmi_c",
        "M5_孕周二次交互": (
            "y_pct ~ gest_week_c + I(gest_week_c ** 2) + bmi_c + gest_week_c:bmi_c"
        ),
        "M6_双二次交互": (
            "y_pct ~ gest_week_c + I(gest_week_c ** 2) + bmi_c + "
            "I(bmi_c ** 2) + gest_week_c:bmi_c"
        ),
    }
    results = {}
    comparison_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, formula in formulas.items():
            model = smf.mixedlm(formula, valid, groups=valid["孕妇代码"], re_formula="1")
            fitted = None
            for method in ["lbfgs", "powell", "cg"]:
                try:
                    candidate = model.fit(reml=False, method=method, maxiter=2000, disp=False)
                    fitted = candidate
                    if candidate.converged:
                        break
                except Exception:
                    continue
            if fitted is None:
                raise RuntimeError(f"模型 {name} 无法拟合")
            results[name] = fitted
            comparison_rows.append(
                {
                    "模型": name,
                    "公式": formula,
                    "参数数": int(len(fitted.params)),
                    "logLik": float(fitted.llf),
                    "AIC": float(fitted.aic),
                    "BIC": float(fitted.bic),
                    "收敛": bool(fitted.converged),
                }
            )
    comparison = pd.DataFrame(comparison_rows).sort_values("AIC")
    selected_name = comparison.loc[comparison["模型"] != "M0_空模型", "模型"].iloc[0]
    selected = results[selected_name]

    null = results["M0_空模型"]
    lr_stat = 2 * (selected.llf - null.llf)
    lr_df = max(1, len(selected.fe_params) - len(null.fe_params))
    lr_p = stats.chi2.sf(lr_stat, lr_df)

    fixed_prediction = np.asarray(selected.model.exog @ selected.fe_params)
    var_fixed = float(np.var(fixed_prediction, ddof=1))
    var_random = float(np.asarray(selected.cov_re)[0, 0])
    var_residual = float(selected.scale)
    total_variance = var_fixed + var_random + var_residual
    metrics = {
        "selected_name": selected_name,
        "selected_formula": formulas[selected_name],
        "lr_stat_vs_null": float(lr_stat),
        "lr_df": int(lr_df),
        "lr_p_vs_null": float(lr_p),
        "marginal_r2": var_fixed / total_variance,
        "conditional_r2": (var_fixed + var_random) / total_variance,
        "icc": var_random / (var_random + var_residual),
        "random_intercept_variance": var_random,
        "residual_variance": var_residual,
    }
    return formulas, results, comparison, selected_name, selected, metrics


def coefficient_table(result) -> pd.DataFrame:
    conf = result.conf_int()
    names = list(result.params.index)
    table = pd.DataFrame(
        {
            "项": names,
            "估计值": result.params.values,
            "标准误": result.bse.reindex(names).values,
            "z值": result.tvalues.reindex(names).values,
            "p值": result.pvalues.reindex(names).values,
            "95%CI下限": conf.reindex(names)[0].values,
            "95%CI上限": conf.reindex(names)[1].values,
        }
    )
    return table


def fit_extended_model(valid: pd.DataFrame, base_formula: str):
    needed = [
        "年龄",
        "GC含量",
        "log_raw_reads",
        "在参考基因组上比对的比例",
        "重复读段的比例",
        "被过滤掉读段数的比例",
        "IVF妊娠",
    ]
    extended_data = valid.dropna(subset=needed).copy()
    extended_formula = (
        base_formula
        + " + age_c + gc_c + log_reads_c + map_ratio_c + dup_ratio_c + filter_ratio_c"
        + " + C(IVF妊娠, Treatment(reference='自然受孕'))"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            extended_formula,
            extended_data,
            groups=extended_data["孕妇代码"],
            re_formula="1",
        )
        result = None
        for method in ["lbfgs", "powell", "cg"]:
            try:
                candidate = model.fit(reml=False, method=method, maxiter=2500, disp=False)
                result = candidate
                if candidate.converged:
                    break
            except Exception:
                continue
    if result is None:
        raise RuntimeError("扩展模型无法拟合")
    return extended_formula, extended_data, result


def prediction_grid(valid: pd.DataFrame, result) -> pd.DataFrame:
    week_values = np.linspace(valid["gest_week"].quantile(0.01), valid["gest_week"].quantile(0.99), 100)
    bmi_values = np.linspace(valid["孕妇BMI"].quantile(0.01), valid["孕妇BMI"].quantile(0.99), 90)
    week_grid, bmi_grid = np.meshgrid(week_values, bmi_values)
    grid = pd.DataFrame(
        {
            "gest_week": week_grid.ravel(),
            "孕妇BMI": bmi_grid.ravel(),
        }
    )
    grid["gest_week_c"] = grid["gest_week"] - valid["gest_week"].mean()
    grid["bmi_c"] = grid["孕妇BMI"] - valid["孕妇BMI"].mean()
    grid["pred_y_pct"] = result.predict(grid)
    return grid


def plot_relationships(valid: pd.DataFrame, result, selected_name: str) -> None:
    color = "#176B87"
    accent = "#D95F59"
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

    ax = axes[0]
    points = ax.scatter(
        valid["gest_week"],
        valid["y_pct"],
        c=valid["孕妇BMI"],
        cmap="viridis",
        alpha=0.42,
        s=22,
        linewidths=0,
    )
    smooth = lowess(valid["y_pct"], valid["gest_week"], frac=0.32, return_sorted=True)
    ax.plot(smooth[:, 0], smooth[:, 1], color=accent, lw=2.5, label="LOWESS趋势")
    ax.axhline(4, color="#555555", ls="--", lw=1.3, label="4%达标线")
    ax.set(title="A  孕周与Y染色体浓度", xlabel="检测孕周（周）", ylabel="Y染色体浓度（%）")
    ax.legend(frameon=False, loc="upper left")
    cbar = fig.colorbar(points, ax=ax, pad=0.01)
    cbar.set_label("BMI")

    ax = axes[1]
    points2 = ax.scatter(
        valid["孕妇BMI"],
        valid["y_pct"],
        c=valid["gest_week"],
        cmap="plasma",
        alpha=0.42,
        s=22,
        linewidths=0,
    )
    smooth2 = lowess(valid["y_pct"], valid["孕妇BMI"], frac=0.34, return_sorted=True)
    ax.plot(smooth2[:, 0], smooth2[:, 1], color=color, lw=2.5, label="LOWESS趋势")
    ax.axhline(4, color="#555555", ls="--", lw=1.3, label="4%达标线")
    ax.set(title="B  BMI与Y染色体浓度", xlabel="孕妇BMI", ylabel="Y染色体浓度（%）")
    ax.legend(frameon=False, loc="upper right")
    cbar2 = fig.colorbar(points2, ax=ax, pad=0.01)
    cbar2.set_label("孕周（周）")

    ax = axes[2]
    grid = prediction_grid(valid, result)
    pivot = grid.pivot(index="孕妇BMI", columns="gest_week", values="pred_y_pct")
    contour = ax.contourf(
        pivot.columns.values,
        pivot.index.values,
        pivot.values,
        levels=16,
        cmap="YlGnBu",
    )
    model_label = selected_name.split("_", 1)[-1]
    ax.set(title=f"C  {model_label}模型的固定效应预测面", xlabel="检测孕周（周）", ylabel="孕妇BMI")
    cbar3 = fig.colorbar(contour, ax=ax, pad=0.01)
    cbar3.set_label("预测Y染色体浓度（%）")

    fig.suptitle("男胎Y染色体浓度与孕周、BMI的统计关系", fontsize=16, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "question1_relationships.png", bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(valid: pd.DataFrame, result) -> dict:
    fitted = np.asarray(result.fittedvalues)
    resid = np.asarray(result.resid)
    standardized = resid / np.std(resid, ddof=1)
    influence_count = int((np.abs(standardized) > 3).sum())
    shapiro_sample = resid if len(resid) <= 5000 else resid[:5000]
    shapiro = stats.shapiro(shapiro_sample)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    axes[0].scatter(fitted, standardized, alpha=0.45, s=24, color="#247BA0", linewidths=0)
    smooth = lowess(standardized, fitted, frac=0.35, return_sorted=True)
    axes[0].plot(smooth[:, 0], smooth[:, 1], color="#D95F59", lw=2.2)
    axes[0].axhline(0, color="#555555", ls="--", lw=1.2)
    axes[0].axhline(3, color="#999999", ls=":", lw=1)
    axes[0].axhline(-3, color="#999999", ls=":", lw=1)
    axes[0].set(
        title="A  标准化残差与拟合值",
        xlabel="拟合Y染色体浓度（%）",
        ylabel="标准化残差",
    )

    sm.qqplot(resid, line="45", ax=axes[1], markerfacecolor="#247BA0", markeredgecolor="none", alpha=0.55)
    axes[1].set(title="B  残差正态Q-Q图", xlabel="理论分位数", ylabel="样本分位数")
    fig.suptitle("关系模型诊断", fontsize=15, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "question1_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "std_residual_abs_gt_3": influence_count,
        "shapiro_w": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
    }


def plot_correlation_heatmap(valid: pd.DataFrame) -> None:
    rename = {
        "y_pct": "Y浓度",
        "gest_week": "孕周",
        "孕妇BMI": "BMI",
        "年龄": "年龄",
        "身高": "身高",
        "体重": "体重",
        "GC含量": "GC含量",
        "log_raw_reads": "log读段数",
        "在参考基因组上比对的比例": "比对比例",
        "重复读段的比例": "重复比例",
        "被过滤掉读段数的比例": "过滤比例",
    }
    corr = valid[list(rename)].rename(columns=rename).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9.5, 8), constrained_layout=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman相关系数"},
        ax=ax,
    )
    ax.set_title("主要指标的Spearman相关矩阵", pad=12)
    fig.savefig(OUTPUT_DIR / "question1_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def sensitivity_analysis(valid: pd.DataFrame, formula: str) -> dict:
    lower, upper = valid["y_pct"].quantile([0.01, 0.99])
    trimmed = valid[valid["y_pct"].between(lower, upper)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trimmed_result = smf.mixedlm(
            formula, trimmed, groups=trimmed["孕妇代码"], re_formula="1"
        ).fit(reml=False, method="lbfgs", maxiter=2000, disp=False)
    cluster_ols = smf.ols(formula, valid).fit(
        cov_type="cluster", cov_kwds={"groups": valid["孕妇代码"]}
    )
    return {
        "trimmed_n": int(len(trimmed)),
        "trimmed_lower_y_pct": float(lower),
        "trimmed_upper_y_pct": float(upper),
        "trimmed_coefficients": {k: float(v) for k, v in trimmed_result.fe_params.items()},
        "trimmed_pvalues": {k: float(v) for k, v in trimmed_result.pvalues.items() if k != "Group Var"},
        "cluster_ols_coefficients": {k: float(v) for k, v in cluster_ols.params.items()},
        "cluster_ols_pvalues": {k: float(v) for k, v in cluster_ols.pvalues.items()},
    }


def main() -> None:
    configure_plotting()
    raw, valid, audit = prepare_data()
    corr = correlation_table(valid)
    formulas, results, comparison, selected_name, selected, metrics = fit_mixed_models(valid)
    coefficients = coefficient_table(selected)
    extended_formula, extended_data, extended = fit_extended_model(valid, formulas[selected_name])
    extended_coefficients = coefficient_table(extended)
    diagnostics = plot_diagnostics(valid, selected)
    sensitivity = sensitivity_analysis(valid, formulas[selected_name])
    plot_relationships(valid, selected, selected_name)
    plot_correlation_heatmap(valid)

    centers = {
        "gest_week_mean": float(valid["gest_week"].mean()),
        "bmi_mean": float(valid["孕妇BMI"].mean()),
    }
    model_summary = {
        "audit": audit,
        "centers": centers,
        "model_metrics": metrics,
        "diagnostics": diagnostics,
        "sensitivity": sensitivity,
        "selected_fixed_effects": {k: float(v) for k, v in selected.fe_params.items()},
        "selected_fixed_pvalues": {
            k: float(v) for k, v in selected.pvalues.items() if k in selected.fe_params.index
        },
        "extended_formula": extended_formula,
        "extended_n": int(len(extended_data)),
        "extended_aic": float(extended.aic),
    }

    valid.to_csv(OUTPUT_DIR / "question1_clean_episode_data.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUTPUT_DIR / "question1_correlations.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_DIR / "question1_model_comparison.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(OUTPUT_DIR / "question1_coefficients.csv", index=False, encoding="utf-8-sig")
    extended_coefficients.to_csv(
        OUTPUT_DIR / "question1_extended_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "question1_summary.json").write_text(
        json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(model_summary, ensure_ascii=False, indent=2))
    print("\nMODEL COMPARISON\n", comparison.to_string(index=False))
    print("\nSELECTED COEFFICIENTS\n", coefficients.to_string(index=False))
    print("\nCORRELATIONS\n", corr.to_string(index=False))
    print("\nEXTENDED COEFFICIENTS\n", extended_coefficients.to_string(index=False))


if __name__ == "__main__":
    main()
