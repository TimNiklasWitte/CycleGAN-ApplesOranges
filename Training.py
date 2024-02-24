import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import datetime

from CycleGAN import *

from matplotlib import pyplot as plt

NUM_EPOCHS = 200
BATCH_SIZE = 1

num_imgs_tensorboard = 16
interval_log_imgs_tensorboard = 5


def main():

    #
    # Load dataset
    #   

    apples_train_ds = tfds.load("cycle_gan", split="trainA", as_supervised=True)
    apples_train_ds = apples_train_ds.apply(prepare_data)

    apples_test_ds = tfds.load("cycle_gan", split="testA", as_supervised=True)
    apples_test_ds = apples_test_ds.apply(prepare_data)

    oranges_train_ds = tfds.load("cycle_gan", split="trainB", as_supervised=True)
    oranges_train_ds = oranges_train_ds.apply(prepare_data)

    oranges_test_ds = tfds.load("cycle_gan", split="testB", as_supervised=True)
    oranges_test_ds = oranges_test_ds.apply(prepare_data)


    #
    # Model
    #

    cycle_gan = CycleGAN()
    cycle_gan.generator_apples.build(input_shape=(None, 256, 256, 3))
    cycle_gan.generator_oranges.build(input_shape=(None, 256, 256, 3))

    cycle_gan.discriminator_apples.build(input_shape=(None, 256, 256, 3))
    cycle_gan.discriminator_oranges.build(input_shape=(None, 256, 256, 3))

    cycle_gan.generator_apples.summary()
    cycle_gan.discriminator_apples.summary()

    #
    # Logging
    #

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    #
    # Images shown in tensorboard
    #

    img_real_apples_list = []
    for img_real_apples in apples_test_ds.take(num_imgs_tensorboard):
        img_real_apples_list.append(img_real_apples) 
    
    img_real_apples_tensorboard = tf.concat(img_real_apples_list, axis=0)[:num_imgs_tensorboard]

    img_real_oranges_list = []
    for img_real_oranges  in oranges_test_ds.take(num_imgs_tensorboard):
        img_real_oranges_list.append(img_real_oranges) 

    img_real_oranges_tensorboard = tf.concat(img_real_oranges_list, axis=0)[:num_imgs_tensorboard]
    
    log(train_summary_writer, cycle_gan, img_real_apples_tensorboard , img_real_oranges_tensorboard, epoch=0)
     
    #
    # Initialize model.
    #

    cycle_gan = CycleGAN()

    # Build 
    cycle_gan.discriminator_apples.build(input_shape=(1, 256, 256, 3))
    cycle_gan.discriminator_oranges.build(input_shape=(1, 256, 256, 3))
    
    cycle_gan.generator_apples.build(input_shape=(1, 256, 256, 3))
    cycle_gan.generator_oranges.build(input_shape=(1, 256, 256, 3))

    # Get overview of number of parameters
    cycle_gan.discriminator_apples.summary()
    cycle_gan.generator_oranges.summary()
    
    #
    # Train loop
    #
    for epoch in range(1, NUM_EPOCHS + 1):
            
        print(f"Epoch {epoch}")

        for img_real_apples, img_real_oranges in tqdm.tqdm(zip(apples_train_ds, oranges_test_ds), position=0, leave=True):
            cycle_gan.train_step(img_real_apples, img_real_oranges)

        log(train_summary_writer, cycle_gan, img_real_apples_tensorboard , img_real_oranges_tensorboard, epoch)

        if epoch % 25 == 0:
            # Save model (its parameters)
            cycle_gan.discriminator_apples.save_weights(f"./saved_models/discriminator_apples/trained_weights_{epoch}", save_format="tf")
            cycle_gan.discriminator_oranges.save_weights(f"./saved_models/discriminator_oranges/trained_weights_{epoch}", save_format="tf")
            
            cycle_gan.generator_apples.save_weights(f"./saved_models/generator_apples/trained_weights_{epoch}", save_format="tf")
            cycle_gan.generator_oranges.save_weights(f"./saved_models/generator_oranges/trained_weights_{epoch}", save_format="tf")

def log(train_summary_writer, cycle_gan, img_real_apples, img_real_oranges, epoch):

    #
    # Generate images
    #

    if epoch % interval_log_imgs_tensorboard == 0:

        img_fake_apples = cycle_gan.generator_apples(img_real_oranges, training=False)
        img_fake_oranges = cycle_gan.generator_oranges(img_real_apples, training=False)

        num_generated_imgs = img_real_oranges.shape[0]

        apples = tf.concat([img_real_oranges, img_fake_apples], axis=1)
        apples = (apples + 1)/2
        oranges = tf.concat([img_real_apples, img_fake_oranges], axis=1)
        oranges = (oranges + 1)/2
    #
    # Write to TensorBoard
    #
    

    with train_summary_writer.as_default():
        
        for metric in cycle_gan.generator_apples.metrics:
            tf.summary.scalar(f"generator_apples_{metric.name}", metric.result(), step=epoch)
            print(f"generator_apples_{metric.name}: {metric.result()}")
            metric.reset_state()

        for metric in cycle_gan.generator_oranges.metrics:
            tf.summary.scalar(f"generator_oranges_{metric.name}", metric.result(), step=epoch)
            print(f"generator_oranges_{metric.name}: {metric.result()}")
            metric.reset_state()

        
        for metric in cycle_gan.discriminator_apples.metrics:
            tf.summary.scalar(f"discriminator_apples_{metric.name}", metric.result(), step=epoch)
            print(f"discriminator_apples_{metric.name}: {metric.result()}")
            metric.reset_state()

        
        for metric in cycle_gan.discriminator_oranges.metrics:
            tf.summary.scalar(f"discriminator_oranges_{metric.name}", metric.result(), step=epoch)
            print(f"discriminator_oranges_{metric.name}: {metric.result()}")
            metric.reset_state()

        if epoch % interval_log_imgs_tensorboard == 0:
            tf.summary.image(name="oranges to apples",data = apples, step=epoch, max_outputs=num_generated_imgs)
            tf.summary.image(name="apples to oranges",data = oranges, step=epoch, max_outputs=num_generated_imgs)
        



def prepare_data(dataset):

    # Remove label
    dataset = dataset.map(lambda img, label: img)

    dataset = dataset.map(lambda img: tf.image.random_flip_left_right(img))

    dataset = dataset.map(lambda img: tf.image.random_crop(img, size=[256, 256, 3]))
  
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
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")