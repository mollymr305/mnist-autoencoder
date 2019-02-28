"""Code-layer visualization (3D)."""
import cPickle as pickle
import gzip
import lasagne as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

from mpl_toolkits.mplot3d import Axes3D

from helpers import load_mnist_data
from auto_encoder import auto_encoder
from auto_encoder import functions


plt.switch_backend('Agg')

# Get 'well-distributed' list of colours for visualization
cm = plt.get_cmap(name="nipy_spectral")
C = [cm(1. - 1. * (i/10.)) for i in range(10)]

# Load data; show test input
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
plt.figure(figsize=(12,3))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i].reshape(28,28))
    plt.axis('off')

plt.tight_layout()
plt.savefig('./output/3D_in.eps', format='eps')
plt.savefig('./output/3D_in.jpg', format='jpg')
plt.close()

# Retrieve model configuration
encoder, network = auto_encoder(d=3, encode_image=False)
F = functions(encoder, network)
training_function, test_function, code_function, decode_function = F

# Before training: show test output
decoded = test_function(X_test, X_test)[1]
plt.figure(figsize=(12,3))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(decoded[i][0].reshape(28,28))
    plt.axis('off')

plt.tight_layout()
plt.savefig('./output/3D_out_1.eps', format='eps')
plt.savefig('./output/3D_out_1.jpg', format='jpg')
plt.close()

# Before training: show 3D code layer
df = pd.DataFrame(code_function(X_test).reshape(-1, 3), y_test)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(10):
    ax.scatter(df[y_test == i][0], 
               df[y_test == i][1],
               df[y_test == i][2],
               c=C[i])

plt.tight_layout()
plt.savefig('./output/3D_code_1.eps', format='eps')
plt.savefig('./output/3D_code_1.jpg', format='jpg')
plt.close()

# Before training: show 2D view of code layer
coordinates = [(0, 1), (1, 2), (0, 2)]
titles = ["1st & 2nd", "2nd & 3rd", "1st & 3rd"]
fig, ax = plt.subplots(1, 3)
fig.set_figheight(5)
fig.set_figwidth(15)

for i in range(3):
    x, y = coordinates[i]
    for j in range(10):
        ax[i].scatter(df[y_test == j][x], 
                      df[y_test == j][y], 
                      c=C[j])
        ax[i].set_title(" ".join([titles[i], "Coordinates"]), size=18)

plt.legend([str(i) for i in range(10)],bbox_to_anchor=(1.3,1.0),fontsize=18)
plt.savefig('./output/3D_code_xy_1.eps', format='eps')
plt.savefig('./output/3D_code_xy_1.jpg', format='jpg')
plt.close()

# After training: show test output
f = gzip.open('./output/auto_encoder_3D_weights.pkl.gz', 'rb')
weights = pickle.load(f)
f.close()

nn.layers.set_all_param_values(network, weights)
decoded = test_function(X_test, X_test)[1]
plt.figure(figsize=(12,3))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(decoded[i][0].reshape(28,28))
    plt.axis('off')

plt.tight_layout()
plt.savefig('./output/3D_out_2.eps', format='eps')
plt.savefig('./output/3D_out_2.jpg', format='jpg')

# After training: show 3d code layer
df = pd.DataFrame(code_function(X_test).reshape(-1, 3), y_test)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(10):
    ax.scatter(df[y_test == i][0], 
               df[y_test == i][1],
               df[y_test == i][2],
               c=C[i])

plt.tight_layout()
plt.savefig('./output/3D_code_2.eps', format='eps')
plt.savefig('./output/3D_code_2.jpg', format='jpg')
plt.close()

# After training: show 2D view of code layer
coordinates = [(0, 1), (1, 2), (0, 2)]
titles = ["1st & 2nd", "2nd & 3rd", "1st & 3rd"]
fig, ax = plt.subplots(1, 3)
fig.set_figheight(5)
fig.set_figwidth(15)

for i in range(3):
    x, y = coordinates[i]
    for j in range(10):
        ax[i].scatter(df[y_test == j][x], 
                      df[y_test == j][y], 
                      c=C[j])
        ax[i].set_title(" ".join([titles[i], "Coordinates"]))

plt.legend([str(i) for i in range(10)],bbox_to_anchor=(1.3,1.0),fontsize=18)
plt.savefig('./output/3D_code_xy_2.eps', format='eps')
plt.savefig('./output/3D_code_xy_2.jpg', format='jpg')
plt.close()