from LoadDataframe import *
from matplotlib import pyplot as plt



def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    for row_idx, epoch in enumerate(range(0, 200 + 10, 10)):

        fig, axes = plt.subplots(nrows=1, ncols=2)

        apples_to_oranges = df.loc[epoch, "apples to oranges"][13]

        img_1 = apples_to_oranges[:256, :, :]
        img_2 = apples_to_oranges[256:, :, :]

        axes[0].imshow(img_1)
        axes[0].axis("off")
        axes[0].set_title("Real apples")

        axes[1].imshow(img_2)
        axes[1].axis("off")
        axes[1].set_title("Fake oranges")

        fig.suptitle(f"Epoch: {epoch}")
        plt.tight_layout()
        plt.savefig(f"../plots/generated images while training/apples to oranges/epoch_{epoch}.png", bbox_inches='tight')
        plt.close()

    for row_idx, epoch in enumerate(range(0, 200 + 10, 10)):

        fig, axes = plt.subplots(nrows=1, ncols=2)

        apples_to_oranges = df.loc[epoch, "oranges to apples"][0]

        img_1 = apples_to_oranges[:256, :, :]
        img_2 = apples_to_oranges[256:, :, :]

        axes[0].imshow(img_1)
        axes[0].axis("off")
        axes[0].set_title("Real oranges")

        axes[1].imshow(img_2)
        axes[1].axis("off")
        axes[1].set_title("Fake apples")

        fig.suptitle(f"Epoch: {epoch}")
        plt.tight_layout()
        plt.savefig(f"../plots/generated images while training/oranges to apples/epoch_{epoch}.png", bbox_inches='tight')
        plt.close()
   

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")