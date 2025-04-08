import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Config:
    BASE_DIR = "E:/pythonCode/MR2_baseline"
    matlab_root = os.path.join(BASE_DIR, "matlab")
    # 原始数据 JSON 路径
    dataset_xlsx = os.path.join(matlab_root, "output_matlab.xlsx")

def plot_confusion_matrices(excel_file: str, save_fig: bool = True):
    """
    读取指定的 xlsx 文件，筛选出用户指定的三种模型，
    逐一绘制其 3x3 混淆矩阵，横向排布，并在每幅图下方显示表标题和三类的召回率公式及数值。
    
    :param excel_file: 包含数据的 xlsx 文件名，如 "output.xlsx"
    :param save_fig: 是否保存图片到 matlab_root 文件夹
    """
    # 1. 读取数据
    df = pd.read_excel(excel_file)

    # 2. 筛选指定的模型（若名称不同，请自行修改）
    # target_models = [
    #     "xlm-roberta-large",
    #     "xlm-roberta-base",
    #     "bert-base-multilingual-cased"
    # ]

    # target_models = [
    #     "xlm-roberta-large+clip_early",
    #     "xlm-roberta-large+clip_late",
    #     "xlm-roberta-large+clip_mid"
    # ]

    target_models = [
        "xlm-roberta-large+vit_early",
        "xlm-roberta-large+vit_late",
        "xlm-roberta-large+vit_mid"
    ]

    # target_models = [
    #     "clip_early",
    #     "clip_mid",
    #     "clip_late"
    # ]

    df_filtered = df[df["model_name"].isin(target_models)]
    
    # 如果找不到这三个模型，可以检查一下表格里写的名称是否对得上
    if len(df_filtered) < 3:
        print("警告：可能找不到完整的三个模型，请检查表格中 model_name 是否匹配。")
    
    # 3. 创建画布，一排三个子图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 调整边距，为子图下方预留足够空间放文字
    plt.subplots_adjust(bottom=0.35, wspace=0.4)

    # 4. 循环绘制混淆矩阵
    for i, (idx, row) in enumerate(df_filtered.iterrows()):
        # 取出对应模型的 3×3 混淆矩阵
        cm = np.array([
            [row["cm00"], row["cm01"], row["cm02"]],
            [row["cm10"], row["cm11"], row["cm12"]],
            [row["cm20"], row["cm21"], row["cm22"]]
        ])
        
        # 计算三类的召回率
        sum0 = cm[0, :].sum()
        sum1 = cm[1, :].sum()
        sum2 = cm[2, :].sum()
        recall0 = cm[0, 0] / sum0 if sum0 != 0 else 0.0
        recall1 = cm[1, 1] / sum1 if sum1 != 0 else 0.0
        recall2 = cm[2, 2] / sum2 if sum2 != 0 else 0.0

        # 用 seaborn 画热力图
        sns.heatmap(
            cm,
            annot=True,      # 在方格中显示数值
            cmap="Blues",    # 颜色
            fmt="d",         # 整数格式
            ax=axes[i],
            annot_kws={"size": 14}
        )
        
        # 设置坐标轴标签
        axes[i].set_xlabel("Predicted", fontsize=13)
        axes[i].set_ylabel("Actual", fontsize=13)
        axes[i].tick_params(labelsize=13)
        
        # 生成子图标签，动态赋值为 (a)、(b)、(c)
        label = chr(ord('d') + i)
        # 在子图下方写表标题（模型名称）及召回率公式和结果
        bottom_text = (
            f"\nRecall(label=0) = {recall0:.3f}\n"
            f"Recall(label=1) = {recall1:.3f}\n"
            f"Recall(label=2) = {recall2:.3f}\n\n"
            f"({label}) Model: {row['model_name']}"
        )
        axes[i].text(
            0.5, -0.3, bottom_text,
            transform=axes[i].transAxes,
            ha='center', va='center', fontsize=13
        )
    
    # 5. 保存并显示图片
    if save_fig:
        save_path = os.path.join(Config.matlab_root, "matrices_multi_d-f.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 运行时会读取指定路径下的 "output.xlsx"
    plot_confusion_matrices(Config.dataset_xlsx, save_fig=True)
