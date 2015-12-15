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
        #self.y.shape == (5000,10)
        self.y = self.convert_y(y, sizes[self.num_layers-1])
        #self.y.shape == (5000,400)
        self.X = X
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
        a = self.X
        for theta in self.thetas:
            z = self.feed_forward(a, theta)
            a = sigmoid(z)
        inner = (-self.y * np.log(a)) - (1-self.y)*np.log(1-a)
        J = (1/m) * sum(sum(inner))

        #return J
        return J + self.regularize_cost(1, m)

#Good
    def regularize_cost(self, lamb, m):
        reg_sum = sum(map(lambda x:sum(sum(np.square(x[:,1:]))), self.thetas))
        return (lamb/(2*m))*reg_sum

    #takes a(n) and returns a(n+1)
    #returns Z not A
    def feed_forward(self, a, theta):
        a = np.hstack([np.ones((len(a),1)), a])
        return a.dot(theta.T)

    def gradient_descent(self, iterations, eta):
        #print 'nabla thetas' [0] == 25x401, [1]==10x26
        for it in xrange(iterations):
            nabla_thetas = [np.zeros_like(theta) for theta in self.thetas]
            print 'it: ' + str(it) + ' Cost: ' + str(self.compute_cost())
            if it%10 == 0:
                print self.evaluate()

            delta_nablas = self.backprop()
            nabla_thetas = [nb+dnb for nb, dnb in zip(nabla_thetas, delta_nablas)]
            self.thetas = [t - (eta/len(self.y))*nt
                        for t, nt in zip(self.thetas, nabla_thetas)]

    def backprop(self):
        #as = [401x1, 26x1, 10x1]
        a1 = np.hstack([np.ones((len(self.X),1)), self.X])
        z2 = a1.dot(self.thetas[0].T)
        a2 = sigmoid(z2)
        a2 = np.hstack([np.ones((len(a2),1)), a2])
        z3 = a2.dot(self.thetas[1].T)
        a3 = sigmoid(z3)

        delta3 = a3-self.y
        delta2 = (delta3.dot(self.thetas[1]))[:,1:] * sigmoid_gradient(z2) #5000x25

        #25x401, 10x26
        delta_nablas = [np.zeros_like(theta) for theta in self.thetas]

        m = len(self.y)

        reg2 = np.hstack([np.zeros((self.thetas[1].shape[0],1)), self.thetas[1][:,1:]])
        delta_nablas[1] = (delta_nablas[1] + delta3.T.dot(a2)) + reg2

#*** This reg might not be the right formula. missing lambda
        reg1 = np.hstack([np.zeros((self.thetas[0].shape[0],1)), self.thetas[0][:,1:]])
        delta_nablas[0] = (delta_nablas[0] + delta2.T.dot(a1)) + reg1

        return delta_nablas

    def convert_y(self, y, output_size):
        m = np.shape(y)[0]
        output = np.zeros((m, output_size))
        for a in xrange(m):
            output[a, y[a]-1] = 1
        #for a in xrange(m):
        #    if y[a] == 10:
        #        output[a, 0] = 1
        #    else:
        #        output[a, y[a]] = 1
        return output

    def rand_initial_thetas(self, num_input_layers, num_output_layers):
        #epsilon
        e = 0.12
        return np.random.randn(num_output_layers, num_input_layers+1)*2*e-e

    def evaluate(self):
        correct = 0
        for x, y in zip(self.test_x, self.test_y):
            predict = self.predict(x)
            if predict == y:
                correct += 1
        return correct/len(self.test_x)

    def predict(self, x):
        x = x.reshape((len(x),1))
        a1 = np.vstack([1,x])
        z2 = self.thetas[0].dot(a1)
        a2 = sigmoid(z2)
        a2 = np.vstack([1,a2])
        z3 = self.thetas[1].dot(a2)
        a3 = sigmoid(z3)
        return np.argmax(a3)+1



def sigmoid(z):
    #return np.divide(1.0, (1.0 + np.exp(-z)))
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_gradient(z):
    #return np.muliply(sigmoid(z),(1-sigmoid(z)))
    return sigmoid(z)*(1-sigmoid(z))

def run():
    data = scipy.io.loadmat('ex4data1.mat')

    xys = zip(data['X'], data['y'])
    np.random.shuffle(xys)
    training, test = xys[:3500], xys[3500:]
    X,y = zip(*training)
    test_x, test_y = zip(*test)

    input_later = 400
    hidden_layer = 15
    output_layer = 10
    sizes = [input_later, hidden_layer, output_layer]

    nn = NeuralNetwork(sizes,X,y, test_x, test_y)
    nn.gradient_descent(400, 2)

    #800, 1 == 83.92
    #1200, 2 == 91.07


if __name__ == '__main__':
    run()








