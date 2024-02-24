import tensorflow as tf

from Generator import *
from Discriminator import *

class CycleGAN(tf.keras.Model):

    def __init__(self):
        super(CycleGAN, self).__init__()

        self.generator_apples = Generator()
        self.generator_oranges = Generator()

        self.discriminator_apples = Discriminator()
        self.discriminator_oranges = Discriminator()

        self.bce_loss = tf.keras.losses.BinaryCrossentropy()
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()

    @tf.function
    def train_step(self, img_apples, img_oranges):

        with tf.GradientTape(persistent=True) as tape:
            
            # 
            # Classic loss
            #

            img_fake_apples = self.generator_apples(img_oranges, training=True)
            img_fake_oranges = self.generator_oranges(img_apples, training=True)
            
            rating_fake_apples = self.discriminator_apples(img_fake_apples)
            rating_fake_oranges = self.discriminator_oranges(img_fake_oranges)
            
            rating_real_apples = self.discriminator_apples(img_apples)
            rating_real_oranges = self.discriminator_oranges(img_oranges)

            generator_apples_classic_loss = self.bce_loss(tf.ones_like(rating_fake_apples), rating_fake_apples)
            generator_oranges_classic_loss = self.bce_loss(tf.ones_like(rating_fake_oranges), rating_fake_oranges)

            discriminator_apples_fake_loss = self.bce_loss(tf.zeros_like(rating_fake_apples), rating_fake_apples)
            discriminator_apples_real_loss = self.bce_loss(tf.ones_like(rating_real_apples), rating_real_apples)
            
            discriminator_oranges_fake_loss = self.bce_loss(tf.zeros_like(rating_fake_oranges), rating_fake_oranges)
            discriminator_oranges_real_loss = self.bce_loss(tf.ones_like(rating_real_oranges), rating_real_oranges)

            #
            # Cycle loss
            #

            img_cycle_apples = self.generator_apples(img_fake_oranges, training=True)
            img_cycle_oranges = self.generator_oranges(img_fake_apples, training=True)

            cycle_loss_apples = self.mae_loss(img_apples, img_cycle_apples)
            cycle_loss_oranges = self.mae_loss(img_oranges, img_cycle_oranges)

            cycle_loss = cycle_loss_apples + cycle_loss_oranges

            #
            # Identity loss
            #

            img_identity_apples = self.generator_apples(img_apples, training=True)
            img_identity_oranges = self.generator_oranges(img_oranges, training=True)

            identity_loss_apples = self.mae_loss(img_apples, img_identity_apples)
            identity_loss_oranges = self.mae_loss(img_oranges, img_identity_oranges)
            
            #
            # Total loss
            #

            generator_apples_loss = generator_apples_classic_loss + \
                                    10*cycle_loss + \
                                    5*identity_loss_apples 

            generator_oranges_loss = generator_oranges_classic_loss + \
                                    10*cycle_loss + \
                                    5*identity_loss_oranges
            
            discriminator_apples_loss = discriminator_apples_fake_loss + \
                                        discriminator_apples_real_loss

            discriminator_oranges_loss = discriminator_oranges_fake_loss + \
                                         discriminator_oranges_real_loss
            
        #
        # Update generators
        #
        gradients = tape.gradient(generator_apples_loss, self.generator_apples.trainable_variables)
        self.generator_apples.optimizer.apply_gradients(zip(gradients, self.generator_apples.trainable_variables))

        gradients = tape.gradient(generator_oranges_loss, self.generator_oranges.trainable_variables)
        self.generator_oranges.optimizer.apply_gradients(zip(gradients, self.generator_oranges.trainable_variables))

        #
        # Update discriminator
        #

        gradients = tape.gradient(discriminator_apples_loss, self.discriminator_apples.trainable_variables)
        self.discriminator_apples.optimizer.apply_gradients(zip(gradients, self.discriminator_apples.trainable_variables))

        gradients = tape.gradient(discriminator_oranges_loss, self.discriminator_oranges.trainable_variables)
        self.discriminator_oranges.optimizer.apply_gradients(zip(gradients, self.discriminator_oranges.trainable_variables))

        #
        # Update metrices
        #

        # Generator
        self.generator_apples.metric_loss.update_state(generator_apples_loss)
        self.generator_apples.metric_classic_loss.update_state(generator_apples_classic_loss)
        self.generator_apples.metric_cycle_loss.update_state(cycle_loss_apples)
        self.generator_apples.metric_identity_loss.update_state(identity_loss_apples)

        self.generator_oranges.metric_loss.update_state(generator_oranges_loss)
        self.generator_oranges.metric_classic_loss.update_state(generator_oranges_classic_loss)
        self.generator_oranges.metric_cycle_loss.update_state(cycle_loss_oranges)
        self.generator_oranges.metric_identity_loss.update_state(identity_loss_oranges)

        # Discriminator

        # Loss
        self.discriminator_apples.metric_loss.update_state(discriminator_apples_loss)
        self.discriminator_apples.metric_fake_loss.update_state(discriminator_apples_fake_loss)
        self.discriminator_apples.metric_real_loss.update_state(discriminator_apples_real_loss)


        self.discriminator_oranges.metric_loss.update_state(discriminator_oranges_loss)
        self.discriminator_oranges.metric_fake_loss.update_state(discriminator_oranges_fake_loss)
        self.discriminator_oranges.metric_real_loss.update_state(discriminator_oranges_real_loss)

       

        # Accuracy
        classified_fake_apples = tf.math.round(rating_fake_apples)
        classified_real_applies = tf.math.round(rating_real_apples)

        zeros = tf.zeros_like(classified_fake_apples)
        ones = tf.ones_like(classified_real_applies)
        self.discriminator_apples.metric_fake_accuracy.update_state(zeros, classified_fake_apples)
        self.discriminator_apples.metric_real_accuracy.update_state(ones, classified_real_applies)

        classified_fake_oranges = tf.math.round(rating_fake_oranges)
        classified_real_oranges = tf.math.round(rating_real_oranges)

        self.discriminator_oranges.metric_fake_accuracy.update_state(zeros, classified_fake_oranges)
        self.discriminator_oranges.metric_real_accuracy.update_state(ones, classified_real_oranges)
