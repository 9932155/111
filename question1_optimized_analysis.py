from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path


BASE_DIR = Path(r"E:\111")
OUTPUT_DIR = BASE_DIR / "output" / "question1_optimized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))
sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import font_manager
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.nonparametric.smoothers_lowess import lowess

from question1_analysis import prepare_data


WEEK_SCALE = 4.0
BMI_SCALE = 3.0


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    for font_path in [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]:
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


def enrich_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw, data, audit = prepare_data()
    keys = ["孕妇代码", "检测日期", "检测抽血次数", "gest_week"]
    labels = raw.groupby(keys, dropna=False, as_index=False).agg(
        健康=("胎儿是否健康", "first"),
        非整倍体=("染色体的非整倍体", "first"),
    )
    data = data.merge(labels, on=keys, how="left")

    data["g_mean"] = data.groupby("孕妇代码")["gest_week"].transform("mean")
    data["g_within"] = data["gest_week"] - data["g_mean"]
    data["g_between"] = data["g_mean"] - data["gest_week"].mean()
    data["bmi_mean"] = data.groupby("孕妇代码")["孕妇BMI"].transform("mean")
    data["bmi_within"] = data["孕妇BMI"] - data["bmi_mean"]
    data["bmi_between"] = data["bmi_mean"] - data["孕妇BMI"].mean()
    data["y_within"] = data["y_pct"] - data.groupby("孕妇代码")["y_pct"].transform("mean")

    data["g_global_s"] = (data["gest_week"] - data["gest_week"].mean()) / WEEK_SCALE
    data["bmi_global_s"] = (data["孕妇BMI"] - data["孕妇BMI"].mean()) / BMI_SCALE
    data["g_within_s"] = data["g_within"] / WEEK_SCALE
    data["g_between_s"] = data["g_between"] / WEEK_SCALE
    data["bmi_within_s"] = data["bmi_within"] / BMI_SCALE
    data["bmi_between_s"] = data["bmi_between"] / BMI_SCALE
    return raw, data, audit


def fit_mixed(formula: str, data: pd.DataFrame, re_formula: str, reml: bool, method: str = "powell"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data, groups=data["孕妇代码"], re_formula=re_formula)
        result = model.fit(reml=reml, method=method, maxiter=6000, disp=False)
    return result


def fit_models(data: pd.DataFrame):
    formula_global = "y_pct ~ g_global_s + I(g_global_s ** 2) + bmi_global_s"
    formula_wb = (
        "y_pct ~ g_within_s + I(g_within_s ** 2) + g_between_s "
        "+ bmi_within_s + bmi_between_s"
    )
    definitions = {
        "M1 全局效应+随机截距": (formula_global, "1"),
        "M2 全局效应+随机斜率": (formula_global, "1 + g_global_s"),
        "M3 组内组间+随机截距": (formula_wb, "1"),
        "M4 组内组间+随机斜率": (formula_wb, "1 + g_within_s"),
    }
    results = {}
    rows = []
    for name, (formula, re_formula) in definitions.items():
        result = fit_mixed(formula, data, re_formula, reml=False)
        results[name] = result
        rows.append(
            {
                "模型": name,
                "固定效应": formula,
                "随机效应": re_formula,
                "logLik": result.llf,
                "AIC": result.aic,
                "BIC": result.bic,
                "收敛": bool(result.converged),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("AIC")
    final_ml = results["M4 组内组间+随机斜率"]
    final_reml = fit_mixed(formula_wb, data, "1 + g_within_s", reml=True)
    return definitions, results, comparison, final_ml, final_reml


def likelihood_ratio_tests(data: pd.DataFrame, full_ml) -> dict:
    reduced = {
        "总体固定效应": ("y_pct ~ 1", 5),
        "组内孕周一次与二次项": (
            "y_pct ~ g_between_s + bmi_within_s + bmi_between_s", 2
        ),
        "组间平均检测孕周": (
            "y_pct ~ g_within_s + I(g_within_s ** 2) + bmi_within_s + bmi_between_s", 1
        ),
        "BMI组内与组间项": (
            "y_pct ~ g_within_s + I(g_within_s ** 2) + g_between_s", 2
        ),
    }
    tests = {}
    for label, (formula, df) in reduced.items():
        reduced_fit = fit_mixed(formula, data, "1 + g_within_s", reml=False)
        lr = 2 * (full_ml.llf - reduced_fit.llf)
        tests[label] = {
            "lr_stat": float(lr),
            "df": int(df),
            "pvalue": float(stats.chi2.sf(lr, df)),
            "reduced_loglik": float(reduced_fit.llf),
        }
    return tests


def converted_coefficient_table(result) -> pd.DataFrame:
    scales = {
        "Intercept": 1.0,
        "g_within_s": WEEK_SCALE,
        "I(g_within_s ** 2)": WEEK_SCALE**2,
        "g_between_s": WEEK_SCALE,
        "bmi_within_s": BMI_SCALE,
        "bmi_between_s": BMI_SCALE,
    }
    labels = {
        "Intercept": "截距",
        "g_within_s": "组内孕周差",
        "I(g_within_s ** 2)": "组内孕周差二次项",
        "g_between_s": "组间平均检测孕周",
        "bmi_within_s": "组内BMI差",
        "bmi_between_s": "组间平均BMI",
    }
    conf = result.conf_int()
    rows = []
    for term in result.fe_params.index:
        divisor = scales[term]
        rows.append(
            {
                "变量": labels[term],
                "项": term,
                "系数": result.fe_params[term] / divisor,
                "标准误": result.bse_fe[term] / divisor,
                "95%CI下限": conf.loc[term, 0] / divisor,
                "95%CI上限": conf.loc[term, 1] / divisor,
                "p值": result.pvalues[term],
                "单位": "百分点/原始单位" if term != "Intercept" else "百分点",
            }
        )
    return pd.DataFrame(rows)


def robust_checks(raw: pd.DataFrame, data: pd.DataFrame, final_formula: str) -> dict:
    gee_formula = "y_pct ~ g_global_s + I(g_global_s ** 2) + bmi_global_s"
    gee = smf.gee(
        gee_formula,
        groups="孕妇代码",
        data=data,
        cov_struct=Exchangeable(),
        family=sm.families.Gaussian(),
    ).fit()

    first = data.sort_values(["孕妇代码", "gest_week"]).groupby("孕妇代码", as_index=False).first()
    first["g_first_c"] = first["gest_week"] - first["gest_week"].mean()
    first["bmi_first_c"] = first["孕妇BMI"] - first["孕妇BMI"].mean()
    first_ols = smf.ols(
        "y_pct ~ g_first_c + I(g_first_c ** 2) + bmi_first_c", first
    ).fit(cov_type="HC3")

    normal = data[(data["健康"] == "是") & data["非整倍体"].isna()].copy()
    normal_model = fit_mixed(final_formula, normal, "1 + g_within_s", reml=True)
    normal_coef = converted_coefficient_table(normal_model)

    extended_formula = (
        final_formula
        + " + age_c + gc_c + log_reads_c + map_ratio_c + dup_ratio_c + filter_ratio_c"
        + " + C(IVF妊娠, Treatment(reference='自然受孕'))"
    )
    extended_model = fit_mixed(extended_formula, data, "1 + g_within_s", reml=True)

    keys = ["孕妇代码", "检测日期", "检测抽血次数", "gest_week"]
    pair_lists = (
        raw.groupby(keys, dropna=False)
        .filter(lambda frame: len(frame) == 2)
        .groupby(keys, dropna=False)["y_pct"]
        .agg(list)
    )
    pairs = pd.DataFrame(
        {
            "mean_y_pct": [(values[0] + values[1]) / 2 for values in pair_lists],
            "diff_y_pct": [values[0] - values[1] for values in pair_lists],
        }
    )
    diff_mean = float(pairs["diff_y_pct"].mean())
    diff_sd = float(pairs["diff_y_pct"].std(ddof=1))
    technical = {
        "pairs": int(len(pairs)),
        "mean_difference_pp": diff_mean,
        "mean_absolute_difference_pp": float(pairs["diff_y_pct"].abs().mean()),
        "repeat_measurement_sd_pp": diff_sd / np.sqrt(2),
        "max_absolute_difference_pp": float(pairs["diff_y_pct"].abs().max()),
        "loa_lower_pp": diff_mean - 1.96 * diff_sd,
        "loa_upper_pp": diff_mean + 1.96 * diff_sd,
    }

    return {
        "gee": gee,
        "first": first,
        "first_ols": first_ols,
        "normal": normal,
        "normal_model": normal_model,
        "normal_coef": normal_coef,
        "extended_formula": extended_formula,
        "extended_model": extended_model,
        "pairs": pairs,
        "technical": technical,
    }


def model_metrics(result, data: pd.DataFrame) -> dict:
    fitted_fixed = np.asarray(result.model.exog @ result.fe_params)
    var_fixed = float(np.var(fitted_fixed, ddof=1))
    cov_re = np.asarray(result.cov_re)
    z = np.asarray(result.model.exog_re)
    random_variance_by_row = np.einsum("ij,jk,ik->i", z, cov_re, z)
    var_random_average = float(np.mean(random_variance_by_row))
    var_residual = float(result.scale)
    total = var_fixed + var_random_average + var_residual
    resid = np.asarray(result.resid)
    shapiro = stats.shapiro(resid)
    return {
        "marginal_r2": var_fixed / total,
        "conditional_r2": (var_fixed + var_random_average) / total,
        "fixed_variance": var_fixed,
        "average_random_variance": var_random_average,
        "residual_variance": var_residual,
        "shapiro_w": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "abs_standardized_residual_gt_3": int(
            (np.abs(resid / np.std(resid, ddof=1)) > 3).sum()
        ),
        "random_effect_covariance_scaled": cov_re.tolist(),
        "n": int(len(data)),
        "women": int(data["孕妇代码"].nunique()),
    }


def fixed_effect_curve(result, x_weeks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = list(result.fe_params.index)
    cov = result.cov_params().loc[names, names].to_numpy()
    design = np.zeros((len(x_weeks), len(names)))
    design[:, names.index("g_within_s")] = x_weeks / WEEK_SCALE
    design[:, names.index("I(g_within_s ** 2)")] = (x_weeks / WEEK_SCALE) ** 2
    effect = design @ result.fe_params.to_numpy()
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, cov, design))
    return effect, effect - 1.96 * se, effect + 1.96 * se


def plot_relationships(data: pd.DataFrame, result) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=True)
    accent = "#D95F59"
    blue = "#176B87"

    ax = axes[0, 0]
    scatter = ax.scatter(
        data["gest_week"], data["y_pct"], c=data["孕妇BMI"], cmap="viridis",
        alpha=0.38, s=21, linewidths=0
    )
    smooth = lowess(data["y_pct"], data["gest_week"], frac=0.32, return_sorted=True)
    ax.plot(smooth[:, 0], smooth[:, 1], color=accent, lw=2.4, label="总体LOWESS")
    ax.axhline(4, color="#555555", ls="--", lw=1.2, label="4%达标线")
    ax.set(title="A  未校正的总体关系", xlabel="检测孕周（周）", ylabel="Y染色体浓度（%）")
    ax.legend(frameon=False)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("BMI")

    ax = axes[0, 1]
    ax.scatter(data["g_within"], data["y_within"], alpha=0.25, s=18, color="#4C8DAE", linewidths=0)
    smooth_w = lowess(data["y_within"], data["g_within"], frac=0.30, return_sorted=True)
    ax.plot(smooth_w[:, 0], smooth_w[:, 1], color=blue, lw=2.2, label="组内LOWESS")
    grid = np.linspace(data["g_within"].quantile(0.01), data["g_within"].quantile(0.99), 180)
    effect, lower, upper = fixed_effect_curve(result, grid)
    ax.plot(grid, effect, color=accent, lw=2.5, label="混合模型组内效应")
    ax.fill_between(grid, lower, upper, color=accent, alpha=0.16, linewidth=0)
    ax.axhline(0, color="#666666", ls="--", lw=1)
    ax.axvline(0, color="#999999", ls=":", lw=1)
    ax.set(title="B  同一孕妇内的孕周变化", xlabel="相对个人平均孕周（周）", ylabel="相对个人平均Y浓度（百分点）")
    ax.legend(frameon=False)

    person = data.groupby("孕妇代码", as_index=False).agg(
        平均BMI=("孕妇BMI", "mean"), 平均Y浓度=("y_pct", "mean"), 平均孕周=("gest_week", "mean"), 检测次数=("y_pct", "size")
    )
    ax = axes[1, 0]
    dots = ax.scatter(
        person["平均BMI"], person["平均Y浓度"], c=person["平均孕周"],
        s=18 + 8 * person["检测次数"], cmap="plasma", alpha=0.65, linewidths=0
    )
    smooth_b = lowess(person["平均Y浓度"], person["平均BMI"], frac=0.42, return_sorted=True)
    ax.plot(smooth_b[:, 0], smooth_b[:, 1], color=blue, lw=2.2, label="孕妇层面LOWESS")
    bmi_grid = np.linspace(person["平均BMI"].quantile(0.02), person["平均BMI"].quantile(0.98), 120)
    bcoef = result.fe_params["bmi_between_s"] / BMI_SCALE
    model_y = result.fe_params["Intercept"] + bcoef * (bmi_grid - data["孕妇BMI"].mean())
    ax.plot(bmi_grid, model_y, color=accent, lw=2.4, label="校正后的组间BMI效应")
    ax.set(title="C  孕妇间BMI差异", xlabel="孕妇平均BMI", ylabel="孕妇平均Y浓度（%）")
    ax.legend(frameon=False)
    cbar2 = fig.colorbar(dots, ax=ax, pad=0.01)
    cbar2.set_label("平均检测孕周（周）")

    ax = axes[1, 1]
    rng = np.random.default_rng(20250830)
    eligible = person.loc[person["检测次数"] >= 3, "孕妇代码"].to_numpy()
    chosen = rng.choice(eligible, size=min(28, len(eligible)), replace=False)
    for code in chosen:
        sub = data[data["孕妇代码"] == code].sort_values("gest_week")
        ax.plot(sub["gest_week"], sub["y_pct"], color="#5B8FA8", alpha=0.24, lw=1.2)
        ax.scatter(sub["gest_week"], sub["y_pct"], color="#5B8FA8", alpha=0.30, s=12, linewidths=0)
    overall = lowess(data["y_pct"], data["gest_week"], frac=0.32, return_sorted=True)
    ax.plot(overall[:, 0], overall[:, 1], color=accent, lw=2.8, label="总体LOWESS")
    ax.axhline(4, color="#555555", ls="--", lw=1.1)
    ax.set(title="D  个体轨迹示例", xlabel="检测孕周（周）", ylabel="Y染色体浓度（%）")
    ax.legend(frameon=False)

    fig.suptitle("第一问优化分析：总体、组内与组间关系", fontsize=17, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "optimized_relationships.png", bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(result, checks: dict) -> None:
    fitted = np.asarray(result.fittedvalues)
    resid = np.asarray(result.resid)
    std_resid = resid / np.std(resid, ddof=1)
    pairs = checks["pairs"]
    tech = checks["technical"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    axes[0].scatter(fitted, std_resid, alpha=0.38, s=20, color="#247BA0", linewidths=0)
    smooth = lowess(std_resid, fitted, frac=0.34, return_sorted=True)
    axes[0].plot(smooth[:, 0], smooth[:, 1], color="#D95F59", lw=2.2)
    axes[0].axhline(0, color="#555555", ls="--", lw=1)
    axes[0].axhline(3, color="#999999", ls=":", lw=1)
    axes[0].axhline(-3, color="#999999", ls=":", lw=1)
    axes[0].set(title="A  标准化残差与拟合值", xlabel="拟合Y浓度（%）", ylabel="标准化残差")

    sm.qqplot(resid, line="45", ax=axes[1], markerfacecolor="#247BA0", markeredgecolor="none", alpha=0.52)
    axes[1].set(title="B  残差Q-Q图", xlabel="理论分位数", ylabel="样本分位数")

    axes[2].scatter(pairs["mean_y_pct"], pairs["diff_y_pct"], color="#247BA0", s=42, alpha=0.78)
    axes[2].axhline(tech["mean_difference_pp"], color="#222222", lw=1.5, label="平均差")
    axes[2].axhline(tech["loa_lower_pp"], color="#D95F59", ls="--", lw=1.5, label="95%一致性界限")
    axes[2].axhline(tech["loa_upper_pp"], color="#D95F59", ls="--", lw=1.5)
    axes[2].set(title="C  技术复测Bland-Altman图", xlabel="两次复测平均浓度（%）", ylabel="两次复测差值（百分点）")
    axes[2].legend(frameon=False)

    fig.suptitle("优化模型诊断与技术复测误差", fontsize=16, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "optimized_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def serialize_series(series: pd.Series) -> dict:
    return {str(key): float(value) for key, value in series.items()}


def main() -> None:
    configure_plotting()
    raw, data, audit = enrich_data()
    definitions, results, comparison, final_ml, final_reml = fit_models(data)
    lr_tests = likelihood_ratio_tests(data, final_ml)
    coefficients = converted_coefficient_table(final_reml)
    formula_wb = definitions["M4 组内组间+随机斜率"][0]
    checks = robust_checks(raw, data, formula_wb)
    metrics = model_metrics(final_reml, data)
    plot_relationships(data, final_reml)
    plot_diagnostics(final_reml, checks)

    gee = checks["gee"]
    first_ols = checks["first_ols"]
    summary = {
        "audit": audit,
        "main_model": {
            "formula_scaled": formula_wb,
            "ml_aic": float(final_ml.aic),
            "ml_bic": float(final_ml.bic),
            "ml_loglik": float(final_ml.llf),
            "ml_converged": bool(final_ml.converged),
            "reml_converged": bool(final_reml.converged),
            "metrics": metrics,
            "likelihood_ratio_tests": lr_tests,
        },
        "technical_replicates": checks["technical"],
        "gee": {
            "dependence": float(gee.cov_struct.dep_params),
            "coefficients_scaled": serialize_series(gee.params),
            "robust_se_scaled": serialize_series(gee.bse),
            "pvalues": serialize_series(gee.pvalues),
        },
        "first_measurement": {
            "n": int(len(checks["first"])),
            "week_min": float(checks["first"]["gest_week"].min()),
            "week_max": float(checks["first"]["gest_week"].max()),
            "coefficients": serialize_series(first_ols.params),
            "hc3_se": serialize_series(first_ols.bse),
            "pvalues": serialize_series(first_ols.pvalues),
        },
        "normal_subset": {
            "n": int(len(checks["normal"])),
            "women": int(checks["normal"]["孕妇代码"].nunique()),
            "reml_converged": bool(checks["normal_model"].converged),
        },
        "extended_model": {
            "reml_converged": bool(checks["extended_model"].converged),
            "coefficients": serialize_series(checks["extended_model"].fe_params),
            "pvalues": serialize_series(
                checks["extended_model"].pvalues[checks["extended_model"].fe_params.index]
            ),
        },
    }

    comparison.to_csv(OUTPUT_DIR / "optimized_model_comparison.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(OUTPUT_DIR / "optimized_main_coefficients.csv", index=False, encoding="utf-8-sig")
    checks["normal_coef"].to_csv(OUTPUT_DIR / "optimized_normal_subset_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "项": checks["extended_model"].fe_params.index,
            "系数": checks["extended_model"].fe_params.values,
            "标准误": checks["extended_model"].bse_fe.values,
            "p值": checks["extended_model"].pvalues[checks["extended_model"].fe_params.index].values,
        }
    ).to_csv(OUTPUT_DIR / "optimized_extended_coefficients.csv", index=False, encoding="utf-8-sig")
    checks["pairs"].to_csv(OUTPUT_DIR / "optimized_technical_replicates.csv", index=False, encoding="utf-8-sig")
    data.to_csv(OUTPUT_DIR / "optimized_analysis_data.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "optimized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nMODEL COMPARISON\n", comparison.to_string(index=False))
    print("\nMAIN COEFFICIENTS (ORIGINAL UNITS)\n", coefficients.to_string(index=False))
    print("\nNORMAL SUBSET COEFFICIENTS\n", checks["normal_coef"].to_string(index=False))


if __name__ == "__main__":
    main()
