import cPickle as pickle
import datetime as dt
import gzip
import lasagne as nn
import numpy as np
import os
import sys
import time
import theano
import theano.tensor as T

# Helper functions
from helpers import report, load_mnist_data, generate_batches


def auto_encoder(d=3, encode_image=False):
    # d: dimensions of code layer, encode_image: code into image
    encoder = nn.layers.InputLayer(shape=(None, 1, 28, 28))
    encoder = nn.layers.Conv2DLayer(incoming=encoder, num_filters=32, 
        filter_size=(5,5), nonlinearity=nn.nonlinearities.rectify, b=None)
    encoder = nn.layers.DenseLayer(incoming=encoder, num_units=1000,
        nonlinearity=nn.nonlinearities.rectify, b=None)
    encoder = nn.layers.DenseLayer(incoming=encoder, num_units=500,
        nonlinearity=nn.nonlinearities.rectify, b=None)
    encoder = nn.layers.DenseLayer(incoming=encoder, num_units=250,
        nonlinearity=nn.nonlinearities.rectify, b=None)
    encoder = nn.layers.DenseLayer(incoming=encoder, num_units=125,
        nonlinearity=nn.nonlinearities.rectify, b=None)

    if encode_image:
        encoder = nn.layers.DenseLayer(incoming=encoder,
            num_units=1*d*d, nonlinearity=nn.nonlinearities.sigmoid, b=None)
        encoder = nn.layers.ReshapeLayer(incoming=encoder, 
            shape=(-1, 1, d, d)) # d x d image matrix
    else:        
        encoder = nn.layers.DenseLayer(incoming=encoder, num_units=d,
            nonlinearity=None, b=None)
        encoder = nn.layers.ReshapeLayer(incoming=encoder, 
            shape=(-1, 1, 1, d)) # d-dimensional space

    network = nn.layers.DenseLayer(incoming=encoder, num_units=125,
        nonlinearity=nn.nonlinearities.rectify, b=None)
    network = nn.layers.DenseLayer(incoming=network, num_units=250,
        nonlinearity=nn.nonlinearities.rectify, b=None)
    network = nn.layers.DenseLayer(incoming=network, num_units=500,
        nonlinearity=nn.nonlinearities.rectify, b=None)    
    network = nn.layers.DenseLayer(incoming=network, num_units=1000,
        nonlinearity=nn.nonlinearities.rectify, b=None)
    network = nn.layers.DenseLayer(incoming=network, num_units=2000,
        nonlinearity=nn.nonlinearities.rectify, b=None)
    network = nn.layers.DenseLayer(incoming=network, num_units=784,
        nonlinearity=nn.nonlinearities.sigmoid, b=None)
    network = nn.layers.ReshapeLayer(incoming=network, shape=(-1, 1, 28, 28))

    return encoder, network


def functions(encoder, network, l_rate=1.):
    # For network
    X = T.tensor4()
    Y = T.tensor4()  # X = Y
    parameters = nn.layers.get_all_params(layer=network, trainable=True)
    output = nn.layers.get_output(layer_or_layers=network, inputs=X)
    all_layers = nn.layers.get_all_layers(network)
    loss = T.mean(nn.objectives.squared_error(output, Y))
    updates = nn.updates.sgd(
        loss_or_grads=loss, params=parameters, learning_rate=l_rate)
    training_function = theano.function(
        inputs=[X, Y], outputs=loss, updates=updates)
    test_function = theano.function(
        inputs=[X, Y], outputs=[loss, output])

    # For encoder
    code_output = nn.layers.get_output(layer_or_layers=encoder, inputs=X)
    code_function = theano.function(inputs=[X], outputs=code_output)

    # For decoder
    Z = T.tensor4()
    decode_output = nn.layers.get_output(
        layer_or_layers=network, inputs={encoder: Z})
    decode_function = theano.function(inputs=[Z], outputs=decode_output)

    return training_function, test_function, code_function, decode_function


def train(network, training_function, test_function, output_file):
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    report('Data OK.', output_file)

    # Start training
    epochs = 500
    batch_size = 500
    TL, VL = [], []
    report('Training over {} Epochs...'.format(epochs), output_file)

    header = ['Epoch', 'TL', 'VL', 'Time']
    report('{:<10}{:<20}{:<20}{:<20}'.format(*header), output_file)

    for e in xrange(epochs):
        start_time = time.time()
        tl, vl = 0., 0.
        t_batches, v_batches = 1, 1

        # Training round
        for batch in generate_batches(X_train, X_train, batch_size):
            data, targets = batch
            l = training_function(data, targets)
            tl += l; t_batches += 1
        tl /= t_batches
        TL.append(tl)

        # Validation round
        for batch in generate_batches(X_val, X_val, batch_size):
            data, targets = batch
            l = test_function(data, targets)[0]
            vl += l; v_batches += 1

        vl /= v_batches
        VL.append(vl)
        row = [e + 1, tl, vl, time.time() - start_time]
        report('{:<10}{:<20}{:<20}{:<20}'.format(*row),output_file)

    report('Finished training.', output_file)

    # Model's loss on test data (MSE)
    mse = test_function(X_test, X_test)[0]
    report('MSE on test data: {}.'.format(mse), output_file)

    return TL, VL


if __name__ == '__main__': 
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()

    # Train a model which encodes data into 3-d vector
    # Output settings: record to file instead of printing statements
    title = 'auto_encoder_3D'
    output_file = './output/{}.txt'.format(title)
    report('\n\nStarted: {}.'.format(dt.datetime.now()), output_file)
    report('>>> {}'.format(output_file), output_file)
    # Get network with: d=3, encode_image=False
    encoder, network = auto_encoder(d=3, encode_image=False)
    report('Autoencoder OK.', output_file)
    report('{} parameters'.format(nn.layers.count_params(network)),output_file)
    # Get functions
    F = functions(encoder, network)
    training_function, test_function, code_function, decode_function = F
    report('Functions OK.', output_file)
    TL, VL = train(network, training_function, test_function, output_file)
    # Save training information
    f = gzip.open('./output/{}_info.pkl.gz'.format(title), 'wb')
    info = {'training loss':TL, 'validation loss':VL}
    pickle.dump(info, f)
    f.close()
    report('Saved training info.', output_file)
    # Save weights
    weights = nn.layers.get_all_params(network)
    weights = [np.array(w.get_value()) for w in weights]
    f = gzip.open('./output/{}_weights.pkl.gz'.format(title), 'wb')
    pickle.dump(weights, f)
    f.close()
    report('Saved weights.', output_file)
    # Done
    report('Completed: {}.'.format(dt.datetime.now()), output_file)


    # Train a model which encodes data into 4x4 image
    # Output settings: record to file instead of printing statements
    title = 'auto_encoder_4x4'
    output_file = './output/{}.txt'.format(title)
    report('\n\nStarted: {}.'.format(dt.datetime.now()), output_file)
    report('>>> {}'.format(output_file), output_file)
    # Get network with: d=10, encode_image=True
    encoder, network = auto_encoder(d=4, encode_image=True)
    report('Autoencoder OK.', output_file)
    report('{} parameters'.format(nn.layers.count_params(network)),output_file)
    # Get functions
    F = functions(encoder, network)
    training_function, test_function, code_function, decode_function = F
    report('Functions OK.', output_file)
    TL, VL = train(network, training_function, test_function, output_file)
    # Save training information
    f = gzip.open('./output/{}_info.pkl.gz'.format(title), 'wb')
    info = {'training loss':TL, 'validation loss':VL}
    pickle.dump(info, f)
    f.close()
    report('Saved training info.', output_file)
    # Save weights
    weights = nn.layers.get_all_params(network)
    weights = [np.array(w.get_value()) for w in weights]
    f = gzip.open('./output/{}_weights.pkl.gz'.format(title), 'wb')
    pickle.dump(weights, f)
    f.close()
    report('Saved weights.', output_file)
    # Done
    report('Completed: {}.'.format(dt.datetime.now()), output_file)