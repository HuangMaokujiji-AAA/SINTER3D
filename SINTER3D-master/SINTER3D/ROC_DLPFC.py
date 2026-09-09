import re

import matplotlib.pyplot as plt
import numpy as np


class SmartLayerAdapter:
    """Detect common cortical-layer labels and normalize them."""

    def __init__(self):
        self.layer_patterns = [
            r"Layer_(\d+)",
            r"Layer(\d+)",
            r"L(\d+)",
            r"layer_(\d+)",
            r"layer(\d+)",
            r"(\d+)",
        ]

    def detect_layer_format(self, layer_names):
        layer_info = {}
        other_regions = []

        for name in layer_names:
            for pattern in self.layer_patterns:
                match = re.fullmatch(pattern, str(name), re.IGNORECASE)
                if match:
                    layer_info[name] = int(match.group(1))
                    break
            else:
                other_regions.append(name)

        return layer_info, other_regions

    def create_standardized_mapping(self, layer_names):
        layer_info, other_regions = self.detect_layer_format(layer_names)
        mapping = {name: f"Layer{number}" for name, number in layer_info.items()}
        mapping.update({region: region for region in other_regions})
        return mapping, layer_info, other_regions

    @staticmethod
    def extract_layer_numbers_from_celltype(celltype_name):
        for pattern in (r"L(\d+)_(\d+)", r"L(\d+)", r"Layer(\d+)_(\d+)", r"Layer(\d+)"):
            match = re.search(pattern, celltype_name)
            if match:
                return [int(number) for number in match.groups() if number is not None]
        return []

    def auto_generate_cell_layer_mapping(self, celltype_names, available_layers):
        mapping = {}
        for celltype in celltype_names:
            if not celltype.startswith("Ex_"):
                continue

            layer_numbers = self.extract_layer_numbers_from_celltype(celltype)
            if len(layer_numbers) == 2:
                layer_numbers = range(min(layer_numbers), max(layer_numbers) + 1)

            mapping[celltype] = [
                f"Layer{number}"
                for number in layer_numbers
                if f"Layer{number}" in available_layers
            ]
        return mapping


def smart_roc_analysis(adata, cluster_column="cluster"):
    """Evaluate layer specificity of excitatory-neuron proportions."""
    from sklearn.metrics import auc, roc_curve

    if cluster_column not in adata.obs:
        raise KeyError(f"adata.obs does not contain {cluster_column!r}")

    print("🔍 开始智能数据格式检测...")
    adapter = SmartLayerAdapter()
    layer_mapping, layer_info, _ = adapter.create_standardized_mapping(
        adata.obs[cluster_column].dropna().unique()
    )

    ex_celltypes = [column for column in adata.obs.columns if column.startswith("Ex_")]
    available_layers = [f"Layer{i}" for i in sorted(set(layer_info.values()))]
    cell_layer_mapping = adapter.auto_generate_cell_layer_mapping(ex_celltypes, available_layers)
    standardized_layers = adata.obs[cluster_column].map(layer_mapping)
    true_layers = standardized_layers.to_numpy()

    print("\n🚀 开始执行ROC分析...")
    auc_results = {}
    roc_curves = {}

    for celltype, target_layers in cell_layer_mapping.items():
        if not target_layers:
            print(f"⚠️  {celltype}: 跳过（无匹配层）")
            continue

        scores = adata.obs[celltype].to_numpy(dtype=float)
        valid = np.isfinite(scores) & standardized_layers.notna().to_numpy()
        y_true = np.isin(true_layers[valid], target_layers).astype(int)
        y_scores = scores[valid]
        n_positive = int(y_true.sum())
        n_total = len(y_true)

        if n_positive == 0:
            print(f"⚠️  {celltype}: 跳过（目标层无样本）")
            continue
        if n_positive == n_total:
            print(f"⚠️  {celltype}: 跳过（所有样本都在目标层）")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
        auc_results[celltype] = {
            "auc": auc_score,
            "target_layers": target_layers,
            "n_positive": n_positive,
            "n_negative": n_total - n_positive,
            "mean_prop_in_target": np.mean(y_scores[y_true == 1]),
            "mean_prop_in_other": np.mean(y_scores[y_true == 0]),
        }
        roc_curves[celltype] = {"fpr": fpr, "tpr": tpr, "auc": auc_score}
        print(
            f"✅ {celltype:15} | AUC: {auc_score:.3f} | "
            f"目标层: {target_layers} | 正样本: {n_positive:4d}"
        )

    if not auc_results:
        print("❌ 未找到可分析的细胞类型！")
        return None, None

    auc_scores = [result["auc"] for result in auc_results.values()]
    best = max(auc_results, key=lambda name: auc_results[name]["auc"])
    worst = min(auc_results, key=lambda name: auc_results[name]["auc"])
    print("\n📈 分析完成!")
    print(f"   成功分析的细胞类型: {len(auc_results)}")
    print(f"   平均AUC: {np.mean(auc_scores):.3f}")
    print(f"   AUC范围: {min(auc_scores):.3f} - {max(auc_scores):.3f}")
    print(f"   最佳表现: {best} (AUC: {auc_results[best]['auc']:.3f})")
    print(f"   最差表现: {worst} (AUC: {auc_results[worst]['auc']:.3f})")
    return auc_results, roc_curves


def plot_roc_curves(auc_results, roc_curves, figsize=(15, 6)):
    """Plot ROC curves and an AUC ranking."""
    if not auc_results or not roc_curves:
        raise ValueError("No ROC results to plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_curves)))

    for i, (celltype, roc_data) in enumerate(roc_curves.items()):
        fpr, tpr, auc_score = roc_data["fpr"], roc_data["tpr"], roc_data["auc"]
        ax1.plot(
            fpr,
            tpr,
            color=colors[i],
            linewidth=2.5,
            label=f"{celltype} (AUC: {auc_score:.3f})",
            alpha=0.8,
        )
        if i < 3:
            ax1.fill_between(fpr, tpr, alpha=0.1, color=colors[i])

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=2, label="Random Classifier (AUC: 0.5)")
    ax1.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False Positive Rate (FPR)", ylabel="True Positive Rate (TPR)")
    ax1.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    sorted_results = sorted(
        ((celltype, result["auc"]) for celltype, result in auc_results.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    celltypes, auc_scores = zip(*sorted_results)
    normed = (np.asarray(auc_scores) - min(auc_scores)) / (max(auc_scores) - min(auc_scores) + 1e-8)
    bars = ax2.barh(range(len(celltypes)), auc_scores, color=plt.cm.RdYlGn(normed), alpha=0.8, edgecolor="black", linewidth=0.5)

    for i, (bar, score) in enumerate(zip(bars, auc_scores)):
        ax2.text(score + 0.01, i, f"{score:.3f}", va="center", ha="left", fontweight="bold", fontsize=10)

    ax2.set_yticks(range(len(celltypes)), labels=celltypes, fontsize=10)
    ax2.set(xlim=(0, 1), xlabel="AUC Score")
    ax2.set_title("Cell Type AUC Score Ranking", fontsize=14, fontweight="bold")
    ax2.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Random Level")
    ax2.axvline(np.mean(auc_scores), color="blue", linestyle="--", alpha=0.7, label="Average AUC")
    ax2.legend()
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Excitatory Neuron Cell Type Layer-Specific Analysis", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig
