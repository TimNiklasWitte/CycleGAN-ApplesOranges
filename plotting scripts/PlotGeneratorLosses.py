from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    fig, axes = plt.subplots(1, 4, figsize=(15,5))
    
    sns.lineplot(data=df.loc[1:, ["generator apples loss"]], ax=axes[0], legend=None, palette=['green'], label="apples")
    sns.lineplot(data=df.loc[1:, ["generator oranges loss"]], ax=axes[0], legend=None, palette=['orange'], label="oranges")
    axes[0].set_title("Generator total loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    sns.lineplot(data=df.loc[1:, ["generator apples classic loss"]], ax=axes[1], legend=None, palette=['green'], label="apples")
    sns.lineplot(data=df.loc[1:, ["generator oranges classic loss"]], ax=axes[1], legend=None, palette=['orange'], label="oranges")
    axes[1].set_title("Generator classic loss")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    sns.lineplot(data=df.loc[1:, ["generator apples identity loss"]], ax=axes[2], legend=None, palette=['green'], label="apples")
    sns.lineplot(data=df.loc[1:, ["generator oranges identity loss"]], ax=axes[2], legend=None, palette=['orange'], label="oranges")
    axes[2].set_title("Generator identity loss")
    axes[2].set_ylabel("Loss")
    axes[2].legend()

    sns.lineplot(data=df.loc[1:, ["generator apples l1 loss"]], ax=axes[3], legend=None, palette=['green'], label="apples")
    sns.lineplot(data=df.loc[1:, ["generator oranges l1 loss"]], ax=axes[3], legend=None, palette=['orange'], label="oranges")
    axes[3].set_title("Generator cycle loss")
    axes[3].set_ylabel("Loss")
    axes[3].legend()

    # grid
    for ax in axes.flatten():
        ax.grid()

    plt.tight_layout()
    plt.savefig("../plots/GeneratorLosses.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")