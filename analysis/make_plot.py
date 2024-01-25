import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path, stage, metrics):
    """
    加载数据。
    Args:
        path: 数据路径。
        stagel: 数据标签。如训练集、验证集、测试集。
        metrics: 数据标题。如准确率、损失。
    """
    df = pd.read_csv(path)
    return {"data": df, "stage": stage, "metrics": metrics}


def make_plot(df, file_name, title=None, is_plot=False):
    """
    绘制图片，如果需要在一张图中绘制多条曲线，请输入df列表。
    """
    sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)  # 设置主题风格

    # 绘制图像
    if isinstance(df, list):
        for _df in df:
            fig = sns.lineplot(data=_df["data"], x="Step", y="Value", label=f"{_df['stage']}_{_df['metrics']}")
    else:
        fig = sns.lineplot(data=_df["data"], x="Step", y="Value", label=f"{_df['stage']}_{_df['metrics']}")

    fig.set(xlabel="epoch", ylabel=f"{_df['metrics']}")
    fig.set_title(title, loc="center") if title else None

    # 是否显示图片
    if is_plot:
        fig.plot()
        plt.show()

    # 保存图片
    fig.figure.savefig(f"./analysis//fig/{file_name}")


if __name__ == "__main__":
    df1 = load_data("./analysis/metrics/test_data1.csv", "val", "test")
    df2 = load_data("./analysis/metrics/test_data2.csv", "val", "test")

    make_plot([df1, df2], "test.svg", "test", is_plot=True)
