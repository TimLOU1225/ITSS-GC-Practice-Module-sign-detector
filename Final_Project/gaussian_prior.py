from __future__ import division
import tensorflow as tf
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import numpy as np

class LearningPrior(Layer):
    def __init__(self, nb_gaussian, init='normal', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, **kwargs):
        super(LearningPrior, self).__init__(**kwargs)
        self.nb_gaussian = nb_gaussian
        self.init = initializers.get(init)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.W_constraint = constraints.get(W_constraint)

        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights

        
    def build(self, input_shape):
        self.W_shape = (self.nb_gaussian * 4,)
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=self.W_shape,
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 trainable=True,
                                 dtype='float32')

        super(LearningPrior, self).build(input_shape)

        
    def call(self, x, mask=None):
        mu_x = self.W[:self.nb_gaussian]
        mu_y = self.W[self.nb_gaussian:self.nb_gaussian*2]
        sigma_x = self.W[self.nb_gaussian*2:self.nb_gaussian*3]
        sigma_y = self.W[self.nb_gaussian*3:]

        b_s, height, width = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        e = tf.cast(height, tf.float32) / tf.cast(width, tf.float32)
        e1 = (1 - e) / 2
        e2 = e1 + e

        mu_x = tf.clip_by_value(mu_x, 0.25, 0.75)
        mu_y = tf.clip_by_value(mu_y, 0.35, 0.65)
        sigma_x = tf.clip_by_value(sigma_x, 0.1, 0.9)
        sigma_y = tf.clip_by_value(sigma_y, 0.2, 0.8)

        linspace_x = tf.linspace(0.0, 1.0, width)
        linspace_y = tf.linspace(e1, e2, height)

        x_t = tf.matmul(tf.ones((height, 1)), tf.reshape(linspace_x, (1, -1)))
        y_t = tf.matmul(tf.reshape(linspace_y, (-1, 1)), tf.ones((1, width)))

        x_t = tf.tile(tf.expand_dims(x_t, -1), [1, 1, self.nb_gaussian])
        y_t = tf.tile(tf.expand_dims(y_t, -1), [1, 1, self.nb_gaussian])

        gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + tf.keras.backend.epsilon()) * \
                   tf.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + tf.keras.backend.epsilon()) +
                            (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + tf.keras.backend.epsilon())))

        max_gauss = tf.reduce_max(gaussian, axis=(0, 1), keepdims=True)
        gaussian_normalized = gaussian / max_gauss

        output = tf.tile(tf.expand_dims(gaussian_normalized, 0), [b_s, 1, 1, 1])

        return output

    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], self.nb_gaussian

    
    def get_config(self):
        config = super(LearningPrior, self).get_config()
        config.update({
            'nb_gaussian': self.nb_gaussian,
            'init': initializers.serialize(self.init),
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
        })
        return config