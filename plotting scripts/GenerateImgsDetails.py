import sys
sys.path.append("../")

import tensorflow as tf
import tensorflow_datasets as tfds

from LoadDataframe import *
from matplotlib import pyplot as plt


from CycleGAN import *


def remove_batchDim_scale(img):
    img = (img + 1)/2
    img = img[0]
    return img


def main():

    cycle_gan = CycleGAN()

    # Build 
    cycle_gan.generator_apples.build(input_shape=(1, 256, 256, 3))
    cycle_gan.generator_oranges.build(input_shape=(1, 256, 256, 3))
    
    cycle_gan.discriminator_apples.build(input_shape=(1, 256, 256, 3))
    cycle_gan.discriminator_oranges.build(input_shape=(1, 256, 256, 3))

    cycle_gan.generator_apples.load_weights(f"../saved_models/generator_apples/trained_weights_200").expect_partial()
    cycle_gan.generator_oranges.load_weights(f"../saved_models/generator_oranges/trained_weights_150").expect_partial()
    
    cycle_gan.discriminator_apples.load_weights(f"../saved_models/discriminator_apples/trained_weights_200").expect_partial()
    cycle_gan.discriminator_oranges.load_weights(f"../saved_models/discriminator_oranges/trained_weights_150").expect_partial()
    
    cycle_gan.generator_apples.summary()
    cycle_gan.discriminator_apples.summary()

    apples_test_ds = tfds.load("cycle_gan", split="testA", as_supervised=True)
    apples_test_ds = apples_test_ds.apply(prepare_data)

    oranges_test_ds = tfds.load("cycle_gan", split="testB", as_supervised=True)
    oranges_test_ds = oranges_test_ds.apply(prepare_data)

    #
    # Apples -> oranges
    #

    for idx, img_real_apples in enumerate(apples_test_ds):

        img_fake_oranges = cycle_gan.generator_oranges(img_real_apples)
        img_cycle_apples = cycle_gan.generator_apples(img_fake_oranges)


        pred_apples_real_apples = cycle_gan.discriminator_apples(img_real_apples)
        pred_oranges_real_apples = cycle_gan.discriminator_oranges(img_real_apples)

        pred_apples_fake_oranges = cycle_gan.discriminator_apples(img_fake_oranges)
        pred_oranges_fake_oranges = cycle_gan.discriminator_oranges(img_fake_oranges)

        pred_apples_cycle_apples = cycle_gan.discriminator_apples(img_cycle_apples)
        pred_oranges_cycle_apples = cycle_gan.discriminator_oranges(img_cycle_apples)


        img_fake_oranges = remove_batchDim_scale(img_fake_oranges)
        img_real_apples = remove_batchDim_scale(img_real_apples)
        img_cycle_apples = remove_batchDim_scale(img_cycle_apples)


        pred_apples_real_apples = remove_batchDim_scale(pred_apples_real_apples)
        pred_oranges_real_apples = remove_batchDim_scale(pred_oranges_real_apples)

        pred_apples_fake_oranges = remove_batchDim_scale(pred_apples_fake_oranges)
        pred_oranges_fake_oranges = remove_batchDim_scale(pred_oranges_fake_oranges)

        pred_apples_cycle_apples = remove_batchDim_scale(pred_apples_cycle_apples)
        pred_oranges_cycle_apples = remove_batchDim_scale(pred_oranges_cycle_apples)


        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

        #
        # 1st column
        #
        axes[0][0].imshow(img_real_apples)
        axes[0][0].set_title("Real apples")
        axes[0][0].axis("off")

        axes[1][0].imshow(pred_apples_real_apples)
        pred = tf.math.reduce_mean(pred_apples_real_apples)
        axes[1][0].set_title(f"Discriminator apples patches\n(avg: {pred:1.3f})")
        axes[1][0].axis("off")

        axes[2][0].imshow(pred_oranges_real_apples)
        pred = tf.math.reduce_mean(pred_oranges_real_apples)
        axes[2][0].set_title(f"Discriminator oranges patches\n(avg: {pred:1.3f})")
        axes[2][0].axis("off")
        axes[2][0].axis("off")

        #
        # 2nd column
        #

        axes[0][1].imshow(img_fake_oranges)
        axes[0][1].set_title("Fake oranges")
        axes[0][1].axis("off")

        axes[1][1].imshow(pred_apples_fake_oranges)
        pred = tf.math.reduce_mean(pred_apples_fake_oranges)
        axes[1][1].set_title(f"Discriminator apples patches\n(avg: {pred:1.3f})")
        axes[1][1].axis("off")

        axes[2][1].imshow(pred_oranges_fake_oranges)
        pred = tf.math.reduce_mean(pred_oranges_fake_oranges)
        axes[2][1].set_title(f"Discriminator oranges patches\n(avg: {pred:1.3f})")
        axes[2][1].axis("off")
   

        #
        # 3rd column
        #

        axes[0][2].imshow(img_cycle_apples)
        axes[0][2].set_title("Cycle apples")
        axes[0][2].axis("off")

        axes[1][2].imshow(pred_apples_cycle_apples)
        pred = tf.math.reduce_mean(pred_apples_cycle_apples)
        axes[1][2].set_title(f"Discriminator apples patches\n(avg: {pred:1.3f})")
        axes[1][2].axis("off")

        axes[2][2].imshow(pred_oranges_cycle_apples)
        pred = tf.math.reduce_mean(pred_oranges_cycle_apples)
        axes[2][2].set_title(f"Discriminator oranges patches\n(avg: {pred:1.3f})")
        axes[2][2].axis("off")

        plt.tight_layout()
        plt.savefig(f"../plots/results details/apples to oranges/{idx}.png", bbox_inches='tight')
        plt.close()


    #
    # Oranges -> apples
    #

    for idx, img_real_oranges in enumerate(oranges_test_ds):

        img_fake_apples = cycle_gan.generator_apples(img_real_oranges)
        img_cycle_oranges = cycle_gan.generator_oranges(img_fake_apples)

        pred_apples_fake_apples = cycle_gan.discriminator_apples(img_fake_apples)
        pred_oranges_fake_apples = cycle_gan.discriminator_oranges(img_fake_apples)

        pred_apples_real_oranges = cycle_gan.discriminator_apples(img_real_oranges)
        pred_oranges_real_oranges = cycle_gan.discriminator_oranges(img_real_oranges)

        pred_apples_cycle_oranges = cycle_gan.discriminator_apples(img_cycle_oranges)
        pred_oranges_cycle_oranges = cycle_gan.discriminator_oranges(img_cycle_oranges)


        img_fake_apples = remove_batchDim_scale(img_fake_apples)
        img_real_oranges = remove_batchDim_scale(img_real_oranges)
        img_cycle_oranges = remove_batchDim_scale(img_cycle_oranges)


        pred_apples_real_oranges = remove_batchDim_scale(pred_apples_real_oranges)
        pred_oranges_real_oranges = remove_batchDim_scale(pred_oranges_real_oranges)

        pred_apples_fake_apples = remove_batchDim_scale(pred_apples_fake_apples)
        pred_oranges_fake_apples = remove_batchDim_scale(pred_oranges_fake_apples)

        pred_apples_cycle_oranges = remove_batchDim_scale(pred_apples_cycle_oranges)
        pred_oranges_cycle_oranges = remove_batchDim_scale(pred_oranges_cycle_oranges)


        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

        #
        # 1st column
        #
        axes[0][0].imshow(img_real_oranges)
        axes[0][0].set_title("Real oranges")
        axes[0][0].axis("off")

        axes[1][0].imshow(pred_apples_real_oranges)
        pred = tf.math.reduce_mean(pred_apples_real_oranges)
        axes[1][0].set_title(f"Discriminator apples patches\n(avg: {pred:1.3f})")
        axes[1][0].axis("off")

        axes[2][0].imshow(pred_oranges_real_oranges)
        pred = tf.math.reduce_mean(pred_oranges_real_oranges)
        axes[2][0].set_title(f"Discriminator oranges patches\n(avg: {pred:1.3f})")
        axes[2][0].axis("off")
        axes[2][0].axis("off")

        #
        # 2nd column
        #

        axes[0][1].imshow(img_fake_apples)
        axes[0][1].set_title("Fake apples")
        axes[0][1].axis("off")

        axes[1][1].imshow(pred_apples_fake_apples)
        pred = tf.math.reduce_mean(pred_apples_fake_apples)
        axes[1][1].set_title(f"Discriminator apples patches\n(avg: {pred:1.3f})")
        axes[1][1].axis("off")

        axes[2][1].imshow(pred_oranges_fake_apples)
        pred = tf.math.reduce_mean(pred_oranges_fake_apples)
        axes[2][1].set_title(f"Discriminator oranges patches\n(avg: {pred:1.3f})")
        axes[2][1].axis("off")
   

        #
        # 3rd column
        #

        axes[0][2].imshow(img_cycle_oranges)
        axes[0][2].set_title("Cycle oranges")
        axes[0][2].axis("off")

        axes[1][2].imshow(pred_apples_cycle_oranges)
        pred = tf.math.reduce_mean(pred_apples_cycle_oranges)
        axes[1][2].set_title(f"Discriminator apples patches\n(avg: {pred:1.3f})")
        axes[1][2].axis("off")

        axes[2][2].imshow(pred_oranges_cycle_oranges)
        pred = tf.math.reduce_mean(pred_oranges_cycle_oranges)
        axes[2][2].set_title(f"Discriminator oranges patches\n(avg: {pred:1.3f})")
        axes[2][2].axis("off")

        plt.tight_layout()
        plt.savefig(f"../plots/results details/oranges to apples/{idx}.png", bbox_inches='tight')
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