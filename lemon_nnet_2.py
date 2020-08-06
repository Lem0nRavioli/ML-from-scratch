import numpy as np
import random


class Lemon_nnet:
    def __init__(self, regul=0, alpha=0.01, max_iter=20,threshold=10**-10, bias=.5):
        self.layers = []
        self.features = None
        self.labels = None
        self.regul = regul
        self.alpha = alpha
        self.max_iter = max_iter
        self.threshold = threshold
        self.bias = bias

    def fit(self, layout, features, labels):
        features = np.array(features)
        print(len(features[0]))
        layout.insert(0, (len(features[0]), None))
        for i,lay in enumerate(layout[:-1]):
            self.layers.append(Layer(lay[0], layout[i+1][0], layout[i+1][1]))

        self.features = np.insert(np.array(features), 0, 1, axis=1)
        self.labels = np.array(labels)
        cost_previous = float('inf')
        cost_actual = self.cost_function(self.features, self.labels)
        i = self.max_iter
        stop = False
        while i >= 0 and not stop:
            i -= 1
            nudges = self.backward_prop()
            for j in range(len(self.layers)):
                self.layers[j].weights -= self.alpha * nudges[j]
            cost_previous = cost_actual
            cost_actual = self.cost_function(self.features, self.labels)
            if cost_previous - cost_actual <= self.threshold:
                print("minimum reached")
                print(i)
                stop = True
            print(cost_actual)


    def forward_prop(self, features):
        input = features.T
        output = None
        reg = 0
        activations = []
        activations.append(input)
        for layer in self.layers:
            reg += np.sum(layer.weights[:,1:])
            vfunc = np.vectorize(layer.activation)
            output = np.dot(layer.weights, input)
            output = vfunc(output)
            input = np.insert(output, 0, 1, axis=0)
            activations.append(input)
        return output, reg, activations

    # if things fuck up look into the sign of first delta calculation
    def backward_prop(self):
        deltas = []
        nudges = []
        activations = self.forward_prop(self.features)[2]
        # second indices is to get rid of bias value
        deltas.append(np.mean((activations[-1][1:] - self.labels), axis=1, keepdims=True))
        # putting hidden bias to ease the for loop
        deltas[0] = np.insert(deltas[0], 0, 1, axis=0)
        for i,a in enumerate(activations[1:-1]):
            theta = self.layers[-i-1].weights
            # ignoring first last activation
            delta = deltas[0][1:]
            deltas.insert(0, self.calculate_delta(theta, delta))
        for i in range(len(self.layers)):
            ain = activations[i]
            aout = activations[i+1][1:]
            delta = deltas[i][1:]
            nudges.append(self.cost_derivative(delta, aout, ain))
        return nudges

    def cost_function(self, features, labels):
        output, reg, _ = self.forward_prop(features)
        output = (output - labels)**2
        return (np.sum(output) + (self.regul/2) * reg) / len(labels[0])

    # make this separate than the delta calculation loop
    # fucked up by iteration restriction of deltas
    def cost_derivative(self, delta, aout, ain):
        aout = np.mean(aout, axis=1, keepdims=True)
        ain = np.mean(ain, axis=1, keepdims=True)
        return (delta * aout * (1 - aout)) * ain.T

    def calculate_delta(self, theta, delta):
        return np.mean(np.dot(theta.T,delta), axis=1, keepdims=True)


    def gradient_descent(self):
        output, reg, _ = self.forward_prop(self.features)
        # reg not included yet
        error = (output - self.labels)**2
        total_error = np.sum(error) / (2*len(output[0]))

    def score(self):
        pass
    def predict(self, features):
        features = np.insert(np.array(features), 0, 1, axis=1)
        return self.forward_prop(np.array(features))[0]

    def weights_(self):
        print("row => a1, a2, a3, ... an, col => w0, w1, w2,...,wn")
        for i,layer in enumerate(self.layers):
            print("layer " + str(i+1) + " :")
            print(layer.weights)


class Layer:
    def __init__(self, input, output, activation):
        self.weights = np.random.rand(output, input+1)
        if activation == "sig": self.activation = lambda x: 1/(1+np.e**-x)
        elif activation == "relu": self.activation = lambda x: max(0, x)
        elif activation == "tanh": self.activation = lambda x: (np.e**-x - np.e**-x)/(np.e**x + np.e**-x)






'''layout = [(3, "sig"), (2, "sig"), (3, "sig"), (3, "sig"), (4, "sig"), (3, "sig")]
features = [[1,2,3,5],[3,2,1,4],[4,4,1,5],[7,9,1,3]]
labels = [[0,1,0,0],[0,1,0,0],[0,0,0,1]]
model = Lemon_nnet()
model.fit(layout,features,labels)'''


'''model.weights_()
#print(model.features)
print(model.forward_prop(model.features)[0])
#print(model.labels)
#print(model.cost_function(model.features, model.labels))
for arr in model.backward_prop():
    print(arr)'''
'''
a = model.forward_prop(model.features)[0]
c = a - labels
print(c)
b = np.array([np.mean(c, axis=1)]).T
d = np.mean(c, axis=1, keepdims=True)
print(b)
print(d)'''
'''print("sep")
for acti in model.forward_prop(model.features)[2][1:-1]:
    print(acti)
'''
'''print("sep")
for delta in model.backward_prop()[0]:
    print(delta)
for nudge in model.backward_prop():
    print(nudge)'''
'''print(model.predict([[2,5,4,1]]))'''



