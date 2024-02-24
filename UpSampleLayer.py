import tensorflow as tf
import tensorflow_addons as tfa

class UpSampleLayer(tf.keras.layers.Layer):

    def __init__(self, filters, size, apply_dropout=False):
        super(UpSampleLayer, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.layer_list = [
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False),
            tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")

        ]

        if apply_dropout:
            self.layer_list.append(tf.keras.layers.Dropout(0.5))

        self.layer_list.append(tf.keras.layers.ReLU())

    def call(self, x, training):

        for layer in self.layer_list:
            
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training)
            else:
                x = layer(x)
        
        return x