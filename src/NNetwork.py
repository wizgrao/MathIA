import numpy as np
import random
class NNetwork(object):

    def __init__(self,shape):
        self.n_layers = shape.size()
        self.biases = [np.random.randn(y, 1) for y in shape[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]

    def get_result(self,x):
        result = x
        for w, b in zip(self.weights, self.biases):
            result = sigmoid(np.dot(w, result)+b)
        return result
    def stochasticDescent(self, ):

    def back_prop(self, sample_in, sample_out):
        nabla_w = [np.zeroes(w.shape) for w in self.weights]
        nabla_b = [np.zeroes(b.shape) for b in self.biases]


        activations = [sample_in]
        zs = []
        for w, b in zip(self.weights, self.biases):
            zs.append(np.dot(w, activations[-1]) + b)
            activations.append(sigmoid(zs[-1]))


        nabla_c = activations[-1]-sample_out
        delta = nabla_c * sigPrime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in xrange(2,self.n_layers):
            nabla_b[-i] = np.dot(self.weights[-i + 1].transpose(), delta) * sigPrime(zs[-i])
            nabla_w[-i] = np.dot(nabla_b[-i], activations[-i - 1].transpose())

        return (nabla_b, nabla_w)

    def train(self, sample_in, sample_out, step):
        nabla_w = [np.zeroes(w.shape) for w in self.weights]
        nabla_b = [np.zeroes(b.shape) for b in self.biases]

        for i,o in zip(sample_in,sample_out):
            dnb, dnw = self.back_prop(i,o)

            nabla_b = [nb + db for nb, db in zip(nabla_b, dnb)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, dnw)]

        self.weights = [w - (step/sample_in.size)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (step / sample_in.size) * nb for b, nb in zip(self.biases, nabla_b)]




def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigPrime(z):
    return np.exp(-z)/np.power(np.exp(-z)+1,2)
