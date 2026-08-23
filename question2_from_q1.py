from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

BASE_DIR = Path(r"E:\111")
OUTPUT_DIR = BASE_DIR / "output" / "question2_from_q1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from scipy import stats

sys.path.insert(0, str(BASE_DIR))
from question1_analysis import prepare_data  # noqa: E402
from question2_analysis import ceil_to_day, format_week, optimal_partition  # noqa: E402


Q1_SUMMARY = BASE_DIR / "output" / "question1_optimized" / "optimized_summary.json"

# 第一问 GEE 总体预测模型（标准化尺度）
INTERCEPT = 7.5016416489745
BETA_G_S = 1.0935022090774367
BETA_G2_S = 0.2712009619382623
BETA_BMI_S = -0.42512925487314057
WEEK_SCALE = 4.0
BMI_SCALE = 3.0
MIN_SUPPORTED_WEEK = 11.0


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


def model_constants(valid: pd.DataFrame, summary: dict) -> dict:
    return {
        "week_mean": float(valid["gest_week"].mean()),
        "bmi_mean": float(valid["孕妇BMI"].mean()),
        "a": BETA_G2_S / WEEK_SCALE**2,
        "b": BETA_G_S / WEEK_SCALE,
        "d": BETA_BMI_S / BMI_SCALE,
        "residual_sd": math.sqrt(summary["main_model"]["metrics"]["residual_variance"]),
        "technical_sd": float(summary["technical_replicates"]["repeat_measurement_sd_pp"]),
    }


def mean_y(week: np.ndarray | float, bmi: np.ndarray | float, c: dict) -> np.ndarray:
    x = np.asarray(week, dtype=float) - c["week_mean"]
    return (
        INTERCEPT
        + c["b"] * x
        + c["a"] * x**2
        + c["d"] * (np.asarray(bmi, dtype=float) - c["bmi_mean"])
    )


def required_week(bmi: float, q: float, c: dict, sd: float | None = None) -> float:
    """Earliest data-supported week in [11,25] satisfying P(Y>=4)>=q."""
    sigma = c["residual_sd"] if sd is None else sd
    target_mean = 4.0 + stats.norm.ppf(q) * sigma
    if float(mean_y(MIN_SUPPORTED_WEEK, bmi, c)) >= target_mean:
        return MIN_SUPPORTED_WEEK
    if float(mean_y(25.0, bmi, c)) < target_mean:
        return np.inf
    constant = INTERCEPT + c["d"] * (bmi - c["bmi_mean"]) - target_mean
    discriminant = c["b"] ** 2 - 4 * c["a"] * constant
    if discriminant < 0:
        return np.inf
    x = (-c["b"] + math.sqrt(discriminant)) / (2 * c["a"])
    week = c["week_mean"] + x
    return float(max(MIN_SUPPORTED_WEEK, week)) if week <= 25 else np.inf


def attainment_probability(week: float, bmi: float, c: dict) -> float:
    z = (float(mean_y(week, bmi, c)) - 4.0) / c["residual_sd"]
    return float(stats.norm.cdf(z))


def build_groups(women: pd.DataFrame, c: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    ordered = women.sort_values("bmi").reset_index(drop=True).copy()
    ordered["t95"] = ordered["bmi"].map(lambda b: required_week(float(b), 0.95, c))
    partition = optimal_partition(ordered["t95"].to_numpy(), max_groups=6, min_size=30)

    # 2组已解释98.1%的分段改善，但会把BMI 34.2—35.7的临界人群
    # 与极高BMI人群合并，导致全组推迟至19周。保留3组以降低临床延误。
    bounds = partition["partitions"][3]["bounds"]
    rows = []
    for group_id, (i, j) in enumerate(bounds, start=1):
        segment = ordered.iloc[i:j]
        upper_bmi = float(segment["bmi"].max())
        raw_week = required_week(upper_bmi, 0.95, c)
        decision_week = ceil_to_day(raw_week)
        rows.append(
            {
                "group": group_id,
                "n": int(len(segment)),
                "bmi_min": float(segment["bmi"].min()),
                "bmi_max": upper_bmi,
                "raw_week": raw_week,
                "decision_week": decision_week,
                "decision_label": format_week(decision_week),
                "boundary_probability": attainment_probability(decision_week, upper_bmi, c),
            }
        )
    groups = pd.DataFrame(rows)
    cutpoints = [
        (groups.loc[i, "bmi_max"] + groups.loc[i + 1, "bmi_min"]) / 2
        for i in range(len(groups) - 1)
    ]
    groups["bmi_interval"] = [
        f"[{groups.loc[0, 'bmi_min']:.2f}, {cutpoints[0]:.2f})",
        f"[{cutpoints[0]:.2f}, {cutpoints[1]:.2f})",
        f"[{cutpoints[1]:.2f}, {groups.loc[2, 'bmi_max']:.2f}]",
    ]

    partition_rows = []
    for k, result in partition["partitions"].items():
        partition_rows.append(
            {"groups": k, "sse": result["sse"], "relative_improvement": result["explained"]}
        )
    return groups, partition, pd.DataFrame(partition_rows)


def sensitivity_tables(groups: pd.DataFrame, c: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    reliability = []
    for q in [0.90, 0.95, 0.99]:
        for _, row in groups.iterrows():
            t = required_week(float(row["bmi_max"]), q, c)
            label = "25周内不可满足" if not np.isfinite(t) else format_week(ceil_to_day(t))
            reliability.append(
                {"group": int(row["group"]), "q": q, "week": t, "label": label}
            )

    variance = []
    scenarios = [
        ("只看平均值", 0.50, c["residual_sd"]),
        ("仅技术误差95%", 0.95, c["technical_sd"]),
        ("总残差95%（主方案）", 0.95, c["residual_sd"]),
    ]
    for scenario, q, sd in scenarios:
        for _, row in groups.iterrows():
            t = required_week(float(row["bmi_max"]), q, c, sd=sd)
            variance.append(
                {
                    "group": int(row["group"]),
                    "scenario": scenario,
                    "week": t,
                    "label": "25周内不可满足" if not np.isfinite(t) else format_week(ceil_to_day(t)),
                }
            )
    return pd.DataFrame(reliability), pd.DataFrame(variance)


def plot_transformed_model(groups: pd.DataFrame, c: dict) -> None:
    weeks = np.linspace(MIN_SUPPORTED_WEEK, 25, 400)
    colors = sns.color_palette("viridis", len(groups))
    threshold = 4 + stats.norm.ppf(0.95) * c["residual_sd"]
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    for (_, row), color in zip(groups.iterrows(), colors):
        y = mean_y(weeks, row["bmi_max"], c)
        ax.plot(weeks, y, color=color, lw=2.4, label=f"第{int(row['group'])}组上边界 BMI={row['bmi_max']:.2f}")
        ax.scatter(row["decision_week"], mean_y(row["decision_week"], row["bmi_max"], c), color=color, s=55, zorder=4)
        ax.vlines(row["decision_week"], 3.5, mean_y(row["decision_week"], row["bmi_max"], c), color=color, ls=":", lw=1.2)
    ax.axhline(4, color="#E45756", ls="--", lw=1.4, label="基础达标阈值4%")
    ax.axhline(threshold, color="#B279A2", ls="--", lw=1.4, label=f"95%预测阈值{threshold:.3f}%")
    ax.axvline(12, color="#54A24B", ls=":", lw=1.2, label="12周低风险界限")
    ax.set(
        xlabel="孕周（周）",
        ylabel="第一问模型预测 Y 浓度（%）",
        title="由第一问数学模型求解各 BMI 组的达标时点",
        xlim=(MIN_SUPPORTED_WEEK, 25),
        ylim=(3.5, 13.5),
    )
    ax.legend(fontsize=8.5, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "q1_model_threshold_solution.png", bbox_inches="tight")
    plt.close(fig)


def plot_grouping(women: pd.DataFrame, groups: pd.DataFrame, c: dict) -> None:
    grid = np.linspace(women["bmi"].min(), women["bmi"].max(), 500)
    target = np.array([required_week(float(b), 0.95, c) for b in grid])
    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    ax.plot(grid, target, color="#E45756", lw=2.5, label="第一问模型解析得到的个体95%时点")
    colors = sns.color_palette("viridis", len(groups))
    for (_, row), color in zip(groups.iterrows(), colors):
        ax.hlines(row["decision_week"], row["bmi_min"], row["bmi_max"], color=color, lw=6)
        ax.text((row["bmi_min"] + row["bmi_max"]) / 2, row["decision_week"] + 0.25, f"第{int(row['group'])}组", color=color, ha="center", weight="bold")
    ax.scatter(women["bmi"], women["t95"], color="black", s=9, alpha=0.18)
    ax.axhline(12, color="#54A24B", ls="--", lw=1.2, label="12周低风险界限")
    ax.set(xlabel="孕妇平均 BMI", ylabel="建议检测孕周（周）", title="BMI—解析达标时点曲线与三组排期", ylim=(10.5, 20.2))
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "q1_based_bmi_groups.png", bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(reliability: pd.DataFrame, variance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    rel = reliability.copy()
    rel["目标可靠性"] = rel["q"].map(lambda x: f"{x:.0%}")
    sns.pointplot(data=rel, x="group", y="week", hue="目标可靠性", dodge=0.2, ax=axes[0])
    axes[0].axhline(12, color="#54A24B", ls="--", lw=1.1)
    axes[0].set(xlabel="BMI组别", ylabel="建议孕周", title="A  可靠性水平敏感性")
    sns.pointplot(data=variance, x="group", y="week", hue="scenario", dodge=0.2, ax=axes[1])
    axes[1].axhline(12, color="#54A24B", ls="--", lw=1.1)
    axes[1].set(xlabel="BMI组别", ylabel="建议孕周", title="B  误差口径敏感性")
    axes[0].legend(title="目标可靠性", fontsize=8.5)
    axes[1].legend(title="误差口径", fontsize=8.5)
    fig.savefig(OUTPUT_DIR / "q1_based_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def table_md(df: pd.DataFrame, columns: list[tuple[str, str]], formats: dict[str, str]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|",
    ]
    for _, row in df.iterrows():
        values = []
        for col, _ in columns:
            value = row[col]
            values.append(formats[col].format(value) if col in formats else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    audit: dict,
    summary: dict,
    c: dict,
    groups: pd.DataFrame,
    partition_table: pd.DataFrame,
    reliability: pd.DataFrame,
    variance: pd.DataFrame,
) -> None:
    group_table = table_md(
        groups,
        [("group", "组别"), ("n", "人数"), ("bmi_interval", "BMI区间"), ("decision_label", "最佳NIPT时点"), ("boundary_probability", "边界最低达标概率")],
        {"group": "{:.0f}", "n": "{:.0f}", "boundary_probability": "{:.1%}"},
    )
    part_table = table_md(
        partition_table,
        [("groups", "组数K"), ("sse", "组内平方误差"), ("relative_improvement", "相对最大改善率")],
        {"groups": "{:.0f}", "sse": "{:.4f}", "relative_improvement": "{:.1%}"},
    )
    rel_show = reliability.copy()
    rel_show["q_label"] = rel_show["q"].map(lambda x: f"{x:.0%}")
    rel_pivot = rel_show.pivot(index="group", columns="q_label", values="label").reset_index()
    rel_table = table_md(rel_pivot, [("group", "组别"), ("90%", "90%可靠性"), ("95%", "95%可靠性"), ("99%", "99%可靠性")], {"group": "{:.0f}"})
    var_pivot = variance.pivot(index="group", columns="scenario", values="label").reset_index()
    var_table = table_md(
        var_pivot,
        [("group", "组别"), ("只看平均值", "忽略波动"), ("仅技术误差95%", "仅技术误差95%"), ("总残差95%（主方案）", "总残差95%")],
        {"group": "{:.0f}"},
    )

    threshold95 = 4 + stats.norm.ppf(0.95) * c["residual_sd"]
    report = rf"""# C题第二问报告：由第一问数学模型变形求解 BMI 分组与最佳 NIPT 时点

## 摘要

本报告不另建独立的达标时间主模型，而是直接从第一问已经通过显著性与稳健性检验的 Y 染色体浓度模型出发。第一问的总体 GEE 预测式被改写为孕周 $G$ 与 BMI $B$ 的显式二次函数；再把 $P(Y\ge4\%)\ge q$ 转换为预测均值阈值，最终得到关于孕周的一元二次方程。该方程可解析求出每个 BMI 对应的最早可靠检测孕周。

以第一问残差标准差 {c['residual_sd']:.3f} 个百分点和 95% 单侧可靠性为主方案，经连续动态分段并考虑临床排期效率，得到 3 组：BMI 20.70—34.16 建议 11周+0天，BMI 34.16—35.75 建议 12周+0天，BMI 35.75—46.88 建议 19周+0天。各时点均按组内最高 BMI 求解并向上取整到天，因此同组较低 BMI 的理论达标概率不低于表中边界概率。

## 1. 第一问模型的可求解形式

第一问最终混合模型用于区分同一孕妇内变化与孕妇间差异；但其 $\Delta G$、$G_i^B$ 和随机效应在新孕妇检测前并不完整可知，不能直接令 $\widehat Y=4$ 求绝对孕周。因此，第二问采用第一问稳健性分析中与混合模型方向一致、且能直接表示绝对孕周的总体 GEE 方程：

$$
\mu(G,B)=7.50164+1.09350\frac{{G-16.8194}}4
+0.271201\left(\frac{{G-16.8194}}4\right)^2
-0.425129\frac{{B-32.2831}}3.
$$

换算到原始单位：

$$
\boxed{{
\mu(G,B)=7.50164+0.273376(G-16.8194)
+0.0169501(G-16.8194)^2
-0.141710(B-32.2831)
}}.
$$

其中 $\mu(G,B)$ 是 Y 浓度的总体预测均值，单位为百分数。第一问中孕周一次项、二次项和 BMI 项的 $p$ 值分别为 $1.71\times10^{{-33}}$、$1.34\times10^{{-5}}$ 和 0.0132，方向为“孕周增加则浓度上升、BMI 增加则浓度下降”。在题设 10—25 周区间内，

$$
\frac{{\partial\mu}}{{\partial G}}
=0.273376+2\times0.0169501(G-16.8194)>0,
$$

所以阈值方程在可检测区间内至多有一个最早可行根。题设允许 10 周开始检测，但第一问样本最早孕周为 11 周；为避免向数据支持范围外推，本文把解析排期下限设为 11 周。

## 2. 从 4% 阈值变形为孕周方程

第一问最终混合模型的残差方差为 {summary['main_model']['metrics']['residual_variance']:.4f}，故残差标准差为

$$
\sigma=\sqrt{{{summary['main_model']['metrics']['residual_variance']:.4f}}}={c['residual_sd']:.4f}.
$$

在正态近似下：

$$
P(Y\ge4\mid G,B)=\Phi\left(\frac{{\mu(G,B)-4}}\sigma\right).
$$

若要求达标概率至少为 $q$，等价于：

$$
\mu(G,B)\ge C_q=4+z_q\sigma.
$$

主方案取 $q=0.95$，所以 $C_{{0.95}}={threshold95:.4f}\%$。令 $x=G-16.8194$，代入第一问模型得：

$$
0.0169501x^2+0.273376x
+\left[7.50164-0.141710(B-32.2831)-C_q\right]=0.
$$

若 11 周已经满足约束，则 $G^*(B)=11$；否则取区间内较大的根：

$$
\boxed{{
G^*(B)=16.8194+
\frac{{-0.273376+\sqrt{{0.273376^2-4(0.0169501)c(B,q)}}}}
{{2(0.0169501)}}
}},
$$

其中 $c(B,q)=7.50164-0.141710(B-32.2831)-C_q$。若求得结果超过 25 周，则记为题设检测窗口内不可满足。

![第一问模型阈值求解](./q1_model_threshold_solution.png)

## 3. BMI 分组优化

### 3.1 数据与分组目标

沿用第一问清洗后的 {audit['valid_episodes']} 个检测事件和 {audit['valid_women']} 位孕妇。每位孕妇以多次检测 BMI 均值作为稳定 BMI；第一问已经表明孕期内 BMI 短期变化不显著，而孕妇间 BMI 差异显著。

对 267 位孕妇逐一代入解析式求 $G^*(B_i)$，按 BMI 排序后采用一维动态规划，使同组解析时点的平方误差最小，并要求每组至少 30 人。

{part_table}

纯粹按 90% 改善率规则会选 2 组；但第二组将 BMI 34.2 左右、理论上 11—12 周可完成检测的孕妇，与极端高 BMI 孕妇一起推迟到 19 周，造成不必要延误。增加第 3 组虽只进一步降低少量平方误差，却把临界人群保留在 12 周低风险界限内，因此最终采用 3 组。该选择是在统计紧凑性与题设延迟风险之间的决策优化。

### 3.2 分组结果与最佳时点

{group_table}

每组时点由该组样本内最高 BMI 代入解析式得到，并向上取整到整天；因此不是组均值时点，而是组内保守边界时点。

![BMI分组与解析时点](./q1_based_bmi_groups.png)

## 4. 可靠性与检测误差分析

### 4.1 可靠性水平

{rel_table}

90%、95%、99% 分别反映不同的容错要求。可靠性越高，尤其是高 BMI 组，建议时点越晚。主报告采用 95%，在检测失败风险与延迟风险之间取较保守平衡。

### 4.2 检测误差口径

第一问 18 组技术复测估计单次技术误差标准差为 {c['technical_sd']:.3f} 个百分点；最终混合模型残差标准差为 {c['residual_sd']:.3f} 个百分点，后者还包含短期生物波动和未解释因素。三种口径结果如下：

{var_table}

“只看平均值”相当于直接解 $\mu=4$，会给出过于乐观的时点；“仅技术误差”只防范实验室波动；主方案使用总残差，能同时覆盖技术误差和未解释的个体内波动，因此排期更保守。

![可靠性和误差敏感性](./q1_based_sensitivity.png)

## 5. 结果解释与执行建议

1. **第一组 BMI 20.70—34.16：** 模型在第一问数据支持下限 11 周已满足 95% 预测约束，建议 11周+0天开始检测。
2. **第二组 BMI 34.16—35.75：** 临界 BMI 的解析根接近 12 周，建议统一安排在 12周+0天，以保留在题设低风险窗口内。
3. **第三组 BMI 35.75—46.88：** 高 BMI 明显推迟达标时间；为覆盖样本上边界，建议 19周+0天。由于该组上尾样本少，可先在较早时点检测，若 Y 浓度接近 4% 或质量指标不佳，再按 19 周节点复测。
4. 检测值若落在 $4\%\pm0.447\%$ 附近，应视为阈值不稳定区，不能仅凭单次硬分类。

## 6. 模型自审查与局限性

1. 本方法严格由第一问数学模型变形而来，保证两问逻辑一致；但 GEE 是总体平均模型，不能替代个体诊断。
2. 第一问二次关系在 11—25 周内单调，因此解析求根有效；虽然题设允许 10 周检测，但本报告不向第一问样本下限 11 周以前外推。
3. 总残差的正态近似用于把连续浓度模型转成概率约束，而第一问诊断显示残差存在厚尾，因此 95% 只是模型近似可靠性。
4. 第三组跨度较大，是因为 BMI>40 的孕妇数量较少；其 19 周建议由 BMI=46.88 的样本边界决定，对多数第三组孕妇偏保守。
5. 数据存在选择性复测。GEE 关系能提供总体预测式，但不能完全消除检测计划带来的偏差，建议在外部样本上校准切点。
6. 超出样本 BMI 20.70—46.88 的孕妇属于外推，不应直接套用本分组。

## 7. 结论

第二问可由第一问模型直接变形为一元二次阈值方程。将第一问总体预测均值 $\mu(G,B)$ 与 4% 达标概率约束结合后，可解析得到最早孕周 $G^*(B)$；再对解析时点进行连续动态分段，得到三组排期：

- BMI 20.70—34.16：11周+0天；
- BMI 34.16—35.75：12周+0天；
- BMI 35.75—46.88：19周+0天。

该结果实现了从“第一问浓度关系模型”到“第二问分组决策”的完整数学闭环。主方案采用第一问总残差和 95% 单侧可靠性；临界检测结果仍应结合质量指标和复测处理。
"""
    (OUTPUT_DIR / "第二问报告_基于第一问模型.md").write_text(report, encoding="utf-8")


def main() -> None:
    configure_plotting()
    _, valid, audit = prepare_data()
    summary = json.loads(Q1_SUMMARY.read_text(encoding="utf-8"))
    c = model_constants(valid, summary)
    women = valid.groupby("孕妇代码").agg(bmi=("孕妇BMI", "mean")).reset_index()
    groups, partition, partition_table = build_groups(women, c)
    women = women.sort_values("bmi").reset_index(drop=True)
    women["t95"] = women["bmi"].map(lambda b: required_week(float(b), 0.95, c))
    reliability, variance = sensitivity_tables(groups, c)

    plot_transformed_model(groups, c)
    plot_grouping(women, groups, c)
    plot_sensitivity(reliability, variance)
    write_report(audit, summary, c, groups, partition_table, reliability, variance)

    groups.to_csv(OUTPUT_DIR / "q1_based_groups.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(OUTPUT_DIR / "q1_based_reliability.csv", index=False, encoding="utf-8-sig")
    variance.to_csv(OUTPUT_DIR / "q1_based_error_sensitivity.csv", index=False, encoding="utf-8-sig")
    result = {
        "equation": {
            "intercept": INTERCEPT,
            "week_linear": c["b"],
            "week_quadratic": c["a"],
            "bmi": c["d"],
            "week_mean": c["week_mean"],
            "bmi_mean": c["bmi_mean"],
        },
        "residual_sd": c["residual_sd"],
        "technical_sd": c["technical_sd"],
        "groups": groups.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "q1_based_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(groups.to_string(index=False))
    print("\nReliability:\n", reliability.to_string(index=False))
    print("\nVariance scenarios:\n", variance.to_string(index=False))


if __name__ == "__main__":
    main()
