import numpy as np
from keras import Input
from keras.layers import Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, Layer
from keras.models import Model
from keras.metrics import binary_crossentropy
import keras.backend as K

class CustomVariationalLayer(Layer):
    def set_z_mean(self, z_mean):
        self._z_mean = z_mean

    def set_z_log_var(self, z_log_var):
        self._z_log_var = z_log_var

    def _vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        reconstruction_loss = binary_crossentropy(x, z_decoded)
        regularization_parameter = -5e-4 * self._compute_KL_divergence(self._z_mean, self._z_log_var)
        return K.mean(reconstruction_loss + regularization_parameter)

    def _compute_KL_divergence(self, z_mean, z_log_var):
        return K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self._vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

class VAE(object):
    def __init__(self, image_shape, latent_dim):
        self._latent_dim = latent_dim

        # Encoding
        input_img = Input(shape=image_shape)
        x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
        x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)

        # Sampling
        z = Lambda(self._sampling)([z_mean, z_log_var])

        # Decoding
        decoder_input = Input(K.int_shape(z)[1:])
        x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
        x = Reshape(shape_before_flattening[1:])(x)
        x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(1, 3, padding='same', activation='sigmoid')(x)
        self._decoder = Model(inputs=decoder_input, outputs=x)
        z_decoded = self._decoder(z)
        l = CustomVariationalLayer()
        l.set_z_mean(z_mean)
        l.set_z_log_var(z_log_var)
        y = l([input_img, z_decoded])

        self._vae = Model(input_img, y)

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self._latent_dim), mean=0.0, stddev=1.0)
        return z_mean + K.exp(z_log_var)*epsilon

    def get_model(self):
        return self._vae

    def get_decoder(self):
        return self._decoder
        
