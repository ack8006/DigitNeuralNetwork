from __future__ import division
import numpy as np
import scipy
import scipy.io
from matplotlib import pyplot, cm
import copy
import cPickle, gzip



class NeuralNetwork(object):
    def __init__(self, sizes, X, y, test_x, test_y):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.X.shape == (5000,400)
        #self.y.shape == (5000,10)
        self.X = X
        self.train_y = y
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

        #return J
        return J + self.regularize_cost(1, m)

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

    def gradient_descent(self, iterations, eta, lamb, num_batches):
        #print 'nabla thetas' [0] == 25x401, [1]==10x26
        for it in xrange(1, iterations+1):
            print 'it: ' + str(it) + ' Cost: ' + str(self.compute_cost())
            if it%3 == 0:
                print 'training: ' + str(self.evaluate(self.X, self.train_y))
                print 'test: ' + str(self.evaluate())

            if it%18 == 0:
                eta = self.update_learning_rate(eta, .9)
                lamb = self.update_learning_rate(lamb, .95)
                print eta, lamb

            self.X, self.y, self.train_y = shuffle_inputs(self.X, self.y, self.train_y)
            for bn in xrange(num_batches):
                k = len(self.X) // num_batches
                batch_x, batch_y = self.X[k*bn:k*(bn+1)], self.y[k*bn:k*(bn+1)]

                nabla_thetas = [np.zeros_like(theta) for theta in self.thetas]
                delta_nablas = self.backprop(batch_x, batch_y)
                nabla_thetas = [nb+dnb for nb, dnb in zip(nabla_thetas, delta_nablas)]

                self.l2_regularize(eta, lamb)
                self.thetas = [t - (eta/len(batch_y))*nt
                            for t, nt in zip(self.thetas, nabla_thetas)]

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

    def l2_regularize(self, eta, lamb):
        reg_term = 1-((eta*lamb)/len(self.y))
        self.thetas = [np.hstack([x[:,:1], x[:,1:]*reg_term]) for x in self.thetas]

    def update_learning_rate(self, rate, scalar):
        return rate*scalar

    def convert_y(self, y, output_size):
        # 1 is [0,1,0,0,0,0,0,0,0,0]
        # 0 is [1,0,0,0,0,0,0,0,0,0]
        m = np.shape(y)[0]
        output = np.zeros((m, output_size))
#Mod 10 handles differences in two different datasets
        for a in xrange(m):
            output[a, y[a]%10] = 1
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
#Mod 10 handles differences in two different datasets
            if predict == (y%10):
                correct += 1
        return correct/len(test_x)

    def predict(self, x):
        _, activations = self.feed_forward(x.reshape((1,len(x))))
        return np.argmax(activations[-1])


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def shuffle_inputs(*args):
    shuffle_order = np.random.permutation(len(args[0]))
    return (x[shuffle_order] for x in args)

def load_mnist_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def load_ex4_data():
    data = scipy.io.loadmat('ex4data1.mat')
    X, y = shuffle_inputs(data['X'], data['y'])
    training_size = len(X)*0.8
    X, test_x = X[:training_size], X[training_size:]
    y, test_y = y[:training_size], y[training_size:]
    return (X,y), None, (test_x, test_y)

def ex4():
    pass

def run():
    train_set, valid_set, test_set = load_mnist_data()
    X, y = shuffle_inputs(train_set[0][:10000], train_set[1][:10000])
    test_x, test_y = shuffle_inputs(test_set[0][:1000], test_set[1][:1000])

    #train_set, valid_set, test_set = load_ex4_data()
    #X, y = shuffle_inputs(train_set[0], train_set[1])
    #test_x, test_y = shuffle_inputs(test_set[0], test_set[1])

    input_later = X.shape[1]
    hidden_layer = 50
    output_layer = 10
    sizes = [input_later, hidden_layer, output_layer]

    nn = NeuralNetwork(sizes,X,y, test_x, test_y)
    nn.gradient_descent(400, .5, 4.5, 500)
    #nn.gradient_descent(200, .2, 10, 500)

#Bench ex4
    #94.2, 96.2, it100, eta.5, lamb5, mini200, hidden100
  #*#95.9, 98.8 it400, eta.5, lamb4.5, mini200, hidden50 (18, .9, .95)
    #95.7 96.7, it200, eta.5, lamb4, mini200, hidden40
    #93.7, 97, it200, eta.3, lamb4, mini200, hidden40
    #92.8, 96.2 it200, eta.2, lamb5, mini200, hidden40
    #94.3, 98 it 400, eta.5, lamb4.5, mini200, hidden25 (18, .9, .95)

#Bench mnist
    #85, 87, it200, eta.2, lamb10, mini500, hidden100


#BENCHMARKS
    #92.4  it=200, eta=.5, lamb=1, mini = 7, hidden=100
    #92.3  it=200, eta=.9, lamb=1, mini=7, hidden=100

    #90  it=400, eta=.1, lamb=1, mini=10, hidden=100
    #96.1    it=100, eta=.1, lamb=1, mini=1750, hidden=100
    #94, 99.5 it 100, eta=.03, lamb=1, mini=3500, hidden=100
    #92.5 95.3 it=400, eta=.5, lamb=5, mini=10, hidden=100
    #92.6 95.6 it=400, eta=.5, lamb=5, mini=10, hidden=35

    #93%   it=200, eta=.9, minibatch=7, hiddenlayer=100
    #95.4% it=70,  eta=.9, minibatch=1750, hiddenlayer=100
    #94.3% it=40, eta=.1, minibatch=1750, hiddenlayer=100


if __name__ == '__main__':
    run()








