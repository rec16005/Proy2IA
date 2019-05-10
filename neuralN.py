import numpy as np
import random

def predict(images, W1, W2, B1, B2): 
    predictions = []
    for im in images: 
        a = f(im[0], W1, W2, B1, B2)
        predictions.append(np.argmax(a))
    return predictions

def sigmoid(s):
    return 1/(1+np.exp(-s))

def sigmoid_prime(s):
    return sigmoid(s) * (1 - sigmoid(s))

def f(x, W1, W2, B1, B2):
    Z1 = np.dot(W1, x) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)
    return A2

def vectorize_mini_batch(mini_batch):
    mini_batch_x = []
    mini_batch_y = []
    for k in range(0, len(mini_batch)): 
        mini_batch_x.append(mini_batch[k][0])
        mini_batch_y.append(mini_batch[k][1])

    X = np.hstack(mini_batch_x)
    Y = np.hstack(mini_batch_y)
    return X, Y

def SGD(training_data, epochs, mini_batch_size, eta, test_data): 
    n = len(training_data)
    n_test = len(test_data)

    W1 = np.random.randn(30, 28)
    W2 = np.random.randn(28, 30)
    B1 = np.random.randn(30, 1)
    B2 = np.random.randn(28, 1)

    for j in range(epochs):
        random.shuffle(training_data)
        for k in range(0, n, mini_batch_size): 
            mini_batch = training_data[k: k + mini_batch_size]
            X, Y = vectorize_mini_batch(mini_batch)

            #feed forward
            Z1 = np.dot(W1, X) + B1
            A1 = sigmoid(Z1)
            Z2 = np.dot(W2, A1) + B2
            A2 = sigmoid(Z2)

            #backpropagate
            dZ2 = 1/mini_batch_size*(A2-Y)*sigmoid_prime(Z2)
            dW2 = np.dot(dZ2, A1.T)
            dB2 = 1 / mini_batch_size * np.sum(dZ2, axis = 1, keepdims = True)

            dZ1 = 1/ mini_batch_size * np.dot(W2.T, dZ2) * sigmoid_prime(Z1)
            dW1 = np.dot(dZ1, X.T)
            dB1 = 1 / mini_batch_size * np.sum(dZ1, axis = 1, keepdims = True)

            W2 = W2 - eta*dW2
            W1 = W1 - eta*dW1
            B2 = B2 - eta*dB2
            B1 = B1 - eta*dB1       

        
        test_results = [(np.argmax(f(x, W1, W2, B1, B2)), y) for (x, y) in test_data]
        num_correct = sum(int(x == y) for (x, y) in test_results)
        print("Epoch {} : {} / {}".format(j, num_correct, n_test));

    return W1, B1, W2, B2