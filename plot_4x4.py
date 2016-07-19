import sys
import gzip
import cPickle as pickle
import lasagne as nn
import seaborn as sns
import matplotlib.pyplot as plt
from __helpers__ import load_mnist_data
from auto_encoder import auto_encoder
from auto_encoder import functions
plt.switch_backend('Agg')
# load data; show test input
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i].reshape(28,28))
    plt.axis('off')
plt.tight_layout()
plt.savefig('./output/4x4_in.eps', format='eps')
plt.savefig('./output/4x4_in.jpg', format='jpg')
plt.close()
# retrieve model configuration
encoder, network = auto_encoder(d=4, encode_image=True)
F = functions(encoder, network)
training_function, test_function, code_function, decode_function = F
# before training: show test output
decoded = test_function(X_test, X_test)[1]
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(decoded[i][0].reshape(28,28))
    plt.axis('off')
plt.tight_layout()
plt.savefig('./output/4x4_out_1.eps', format='eps')
plt.savefig('./output/4x4_out_1.jpg', format='jpg')
plt.close()
# before training: show 4x4 code layer
encoded = code_function(X_test)
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(encoded[i][0])
    plt.axis('off')
plt.tight_layout()
plt.savefig('./output/4x4_code_1.eps', format='eps')
plt.savefig('./output/4x4_code_1.jpg', format='jpg')
plt.close()
# after training: show test output
f = gzip.open('./output/auto_encoder_4x4_weights.pkl.gz', 'rb')
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
plt.savefig('./output/4x4_out_2.eps', format='eps')
plt.savefig('./output/4x4_out_2.jpg', format='jpg')
# after training: show 4x4 code layer
encoded = code_function(X_test)
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(encoded[i][0])
    plt.axis('off')
plt.tight_layout()
plt.savefig('./output/4x4_code_2.eps', format='eps')
plt.savefig('./output/4x4_code_2.jpg', format='jpg')
plt.close()
