import sys
sys.path.append("../")

import tensorflow as tf
import tensorflow_datasets as tfds

from LoadDataframe import *
from matplotlib import pyplot as plt


from CycleGAN import *


def main():

    cycle_gan = CycleGAN()

    # Build 
    cycle_gan.generator_apples.build(input_shape=(1, 256, 256, 3))
    cycle_gan.generator_oranges.build(input_shape=(1, 256, 256, 3))
    
    cycle_gan.generator_apples.load_weights(f"../saved_models/generator_apples/trained_weights_200").expect_partial()
    cycle_gan.generator_oranges.load_weights(f"../saved_models/generator_oranges/trained_weights_150").expect_partial()
    
    apples_test_ds = tfds.load("cycle_gan", split="testA", as_supervised=True)
    apples_test_ds = apples_test_ds.apply(prepare_data)

    oranges_test_ds = tfds.load("cycle_gan", split="testB", as_supervised=True)
    oranges_test_ds = oranges_test_ds.apply(prepare_data)


    for idx, img_real_apples in enumerate(apples_test_ds):

        img_fake_oranges = cycle_gan.generator_oranges(img_real_apples)
        img_fake_oranges = img_fake_oranges[0]
        img_fake_oranges = (img_fake_oranges + 1)/2


        img_real_apples = img_real_apples[0]
        img_real_apples = (img_real_apples + 1)/2

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))

        axes[0].imshow(img_real_apples)
        axes[0].axis("off")
        axes[0].set_title("Real apples")

        axes[1].imshow(img_fake_oranges)
        axes[1].axis("off")
        axes[1].set_title("Fake oranges")

        plt.tight_layout()
        plt.savefig(f"../plots/results/apples to oranges/{idx}.png", bbox_inches='tight')
        plt.close()
         


    for idx, img_real_oranges in enumerate(oranges_test_ds):

        img_fake_apples = cycle_gan.generator_apples(img_real_oranges)
        img_fake_apples = img_fake_apples[0]
        img_fake_apples = (img_fake_apples + 1)/2


        img_real_oranges = img_real_oranges[0]
        img_real_oranges = (img_real_oranges + 1)/2

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))

        axes[0].imshow(img_real_oranges)
        axes[0].axis("off")
        axes[0].set_title("Real oranges")

        axes[1].imshow(img_fake_apples)
        axes[1].axis("off")
        axes[1].set_title("Fake apples")

        plt.tight_layout()
        plt.savefig(f"../plots/results/oranges to apples/{idx}.png", bbox_inches='tight')
        plt.close()

def prepare_data(dataset):

    # Remove label
    dataset = dataset.map(lambda img, label: img)

    # dataset = dataset.map(lambda img: tf.image.random_flip_left_right(img))

    # dataset = dataset.map(lambda img: tf.image.random_crop(img, size=[256, 256, 3]))
    
    dataset = dataset.map(lambda img: tf.image.resize(img, size=[256, 256]))

    # Convert data from uint8 to float32
    dataset = dataset.map(lambda img: tf.cast(img, tf.float32) )

    #Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img: (img/128.)-1. )

    # Cache
    #dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(1, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")