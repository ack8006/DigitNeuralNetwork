from __future__ import division
import numpy as np
import scipy
import scipy.io
from matplotlib import pyplot, cm
import copy



class NeuralNetwork(object):
    def __init__(self, sizes, X, y, test_x, test_y):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.X.shape == (5000,400)
        #self.y.shape == (5000,10)
        self.X = X
        self.y = self.convert_y(y, sizes[self.num_layers-1])
        self.test_x, self.test_y = test_x, test_y

        self.thetas = []
        #self.test_thetas()

        #self.thetas[0].shape == (25,401) self.thetas[1].shape == (10x26)
        for x in xrange(len(sizes)-1):
            self.thetas.append(self.rand_initial_thetas(sizes[x], sizes[x+1]))

#Good
    def test_thetas(self):
        test_theta = scipy.io.loadmat('ex4weights.mat')
        self.thetas.append(test_theta['Theta1'])
        self.thetas.append(test_theta['Theta2'])

#Good
    def compute_cost(self):
        m = len(self.y)
        _, activations = self.feed_forward(self.X)
        a = activations[-1]
        inner = (-self.y * np.log(a)) - (1-self.y)*np.log(1-a)
        J = (1/m) * sum(sum(inner))

        return J
        #return J + self.regularize_cost(1, m)

#Good
    def regularize_cost(self, lamb, m):
        reg_sum = sum(map(lambda x:sum(sum(np.square(x[:,1:]))), self.thetas))
        return (lamb/(2*m))*reg_sum

    def feed_forward(self, a):
        #as = [3500x401, 3500x26, 3500x10]  zs = [3500x25, 3500x10]
        zs, activations= [], []
        for l in xrange(self.num_layers-1):
            a = np.hstack([np.ones((len(a),1)), a])
            activations.append(a)
            z = a.dot(self.thetas[l].T)
            zs.append(z)
            a = sigmoid(z)
        activations.append(a)
        return zs, activations

    def gradient_descent(self, iterations, eta, num_batches):
        #print 'nabla thetas' [0] == 25x401, [1]==10x26
        for it in xrange(1, iterations+1):
            print 'it: ' + str(it) + ' Cost: ' + str(self.compute_cost())
            if it%2 == 0:
                print 'test: ' + str(self.evaluate())

            self.X, self.y = shuffle_inputs(self.X, self.y)
            for bn in xrange(num_batches):
                k = len(self.X) // num_batches
                batch_x, batch_y = self.X[k*bn:k*(bn+1)], self.y[k*bn:k*(bn+1)]

                nabla_thetas = [np.zeros_like(theta) for theta in self.thetas]
                delta_nablas = self.backprop(batch_x, batch_y)
                nabla_thetas = [nb+dnb for nb, dnb in zip(nabla_thetas, delta_nablas)]
                self.thetas = [t - (eta/len(batch_y))*nt
                            for t, nt in zip(self.thetas, nabla_thetas)]

    #24%, 46%, 61%, 69%, 75%
    #24, 33, 57, 68
    def backprop(self, batch_x, batch_y):
        #as = [Mx401, Mx26, Mx10]  zs = [Mx25, Mx10]
        zs, activations = self.feed_forward(batch_x)
        deltas = [np.zeros_like(z) for z in zs]
        delta_nablas = [np.zeros_like(theta) for theta in self.thetas]

        deltas[-1] = activations[-1] - batch_y
        for i in xrange(self.num_layers-2, 0, -1):
            deltas[i-1] = (deltas[i].dot(self.thetas[i]))[:,1:] * sigmoid_gradient(zs[i-1])

        for i in xrange(len(deltas)-1, -1, -1):
            delta_nablas[i] = delta_nablas[i] + deltas[i].T.dot(activations[i])

        return delta_nablas

    def convert_y(self, y, output_size):
        # 1 is [1,0,0,0,0,0,0,0,0,0]
        # 0 is [0,0,0,0,0,0,0,0,0,1]
        m = np.shape(y)[0]
        output = np.zeros((m, output_size))
        for a in xrange(m):
            output[a, y[a]-1] = 1
        return output

    def rand_initial_thetas(self, num_input_layers, num_output_layers):
        #epsilon
        e = 0.12
        return np.random.randn(num_output_layers, num_input_layers+1)*2*e-e

    def evaluate(self, test_x=None, test_y=None):
        if test_x is None or test_y is None:
            test_x, test_y = self.test_x, self.test_y
        correct = 0
        for x, y in zip(test_x, test_y):
            predict = self.predict(x)
            if predict == y:
                correct += 1
        return correct/len(test_x)

    def predict(self, x):
        _, activations = self.feed_forward(x.reshape((1,len(x))))
        return np.argmax(activations[-1])+1


def sigmoid(z):
    #return np.divide(1.0, (1.0 + np.exp(-z)))
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_gradient(z):
    #return np.muliply(sigmoid(z),(1-sigmoid(z)))
    return sigmoid(z)*(1-sigmoid(z))

def shuffle_inputs(X,y):
    shuffle_order = np.random.permutation(len(X))
    return X[shuffle_order], y[shuffle_order]

def run():
    data = scipy.io.loadmat('ex4data1.mat')
    X, y = shuffle_inputs(data['X'], data['y'])
    training_size = len(X)*0.7
    X, test_x = X[:training_size], X[training_size:]
    y, test_y = y[:training_size], y[training_size:]

    input_later = X.shape[1]
    hidden_layer = 100
    output_layer = 10
    sizes = [input_later, hidden_layer, output_layer]

    nn = NeuralNetwork(sizes,X,y, test_x, test_y)
    nn.gradient_descent(200, .9, 1750)

#BENCHMARKS
    #93%   it=200, eta=.9, minibatch=7, hiddenlayer=100
    #95.4% it=70,  eta=.9, minibatch=1750, hiddenlayer=100
    #94.3% it=40, eta=.1, minibatch=1750, hiddenlayer=100


if __name__ == '__main__':
    run()








