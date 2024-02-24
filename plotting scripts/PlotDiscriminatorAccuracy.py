from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns


def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    
    sns.lineplot(data=df.loc[1:, ["discriminator apples fake accuracy"]], ax=axes[0], legend=None, palette=['green'], label="apples")
    sns.lineplot(data=df.loc[1:, ["discriminator oranges fake accuracy"]], ax=axes[0], legend=None, palette=['orange'], label="oranges")
    axes[0].set_title("Discriminator fake accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")

    sns.lineplot(data=df.loc[1:, ["discriminator apples real accuracy"]], ax=axes[1], legend=None, palette=['green'], label="apples")
    sns.lineplot(data=df.loc[1:, ["discriminator oranges real accuracy"]], ax=axes[1], legend=None, palette=['orange'], label="oranges")
    axes[1].set_title("Discriminator real accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right")

    # grid
    for ax in axes.flatten():
        ax.grid()

    plt.tight_layout()
    plt.savefig("../plots/DiscriminatorAccuracy.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")