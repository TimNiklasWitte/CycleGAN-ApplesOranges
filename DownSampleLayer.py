import tensorflow as tf
import tensorflow_addons as tfa

class DownSampleLayer(tf.keras.layers.Layer):

    def __init__(self, filters, size, apply_instance_norm=True):
        super(DownSampleLayer, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.layer_list = [
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False)
        ]

        if apply_instance_norm:
            self.layer_list.append(tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform"))

        self.layer_list.append(tf.keras.layers.LeakyReLU())

    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        
        return x