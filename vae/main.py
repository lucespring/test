import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.optimizers import RMSprop
from keras.datasets import mnist
from vae import VAE

img_shape = (28, 28, 1)
batch_size = 32
latent_dim = 2

(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32')/255.0
x_test = x_test.reshape(x_test.shape + (1,))

vae = VAE(img_shape, latent_dim)
decoder = vae.get_decoder()
vae = vae.get_model()
vae.compile(optimizer=RMSprop(), loss=None)
history = vae.fit(x=x_train, y=None, shuffle=True, epochs=10, batch_size=batch_size)
with open('loss.txt', 'a') as f:
    for loss in history.history['loss']:
        f.write(str(loss) + '\r')

n = 15
digit_size = 28
figure = np.zeros((digit_size*n, digit_size*n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i*digit_size:(i+1)*digit_size,
               j*digit_size:(j+1)*digit_size] = digit
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
