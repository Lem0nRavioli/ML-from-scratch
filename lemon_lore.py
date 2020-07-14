import random
import numpy as np


class LemonRegression:
    def __init__(self, alpha=.1, max_iter=2000, threshold=10**-10, init_random=False, bias=.5, regularization=0):
        self.alpha = alpha
        self.regul = regularization
        self.weights = None
        self.labels = None
        self.features = None
        self.max_iter = max_iter
        self.threshold = threshold
        self.bias = bias
        self.init_random = init_random

    def fit(self, features, labels, random_state=None):
        random.seed(random_state)
        features = np.array(features)
        # add 1 for the intercept multiplier
        features = np.insert(features, 0, 1, axis=1)
        self.features = features
        self.weights = np.zeros((len(features[0]), 1))
        if self.init_random: self.weights = np.random.rand(len(features[0]), 1)
        self.labels = np.array(labels)
        self.gradient_descent()

    def predict(self, features, percent=False):
        features = np.array(features)
        # add 1 for the intercept multiplier
        features = np.insert(features, 0, 1, axis=1)
        predict_coef = np.dot(features,self.weights)
        vfunc = np.vectorize(lambda x: 1 if x >= self.bias else 0)
        predict = vfunc(predict_coef)
        if percent: return predict_coef # np.insert(predict_coef,[1],predict,axis=1)
        return predict

    def score(self, features, labels):
        predictions = self.predict(features).T[0]
        labels = np.array(labels)
        score = 0
        for i,x in enumerate(predictions):
            if labels[i] == x: score+=1
        return score/len(labels)

    # do the features processing then return cost function
    def score_(self, features, labels):
        features = np.array(features)
        # add 1 for the intercept multiplier
        features = np.insert(features, 0, 1, axis=1)
        return self.cost_function(features,labels)

    def weights_(self):
        return self.weights

    def gradient_descent(self):
        previous_score = self.cost_function(self.features, self.labels)
        running = True
        iter = 0
        while running:
            iter +=1
            if iter >= self.max_iter: running = False
            regularization = self.regul * self.weights[1:] / len(self.features)
            regularization = np.insert(regularization, 0, [0], axis=0)
            self.weights -= self.alpha * (self.cost_function_derivative() + regularization)
            current_score = self.cost_function(self.features, self.labels)
            if abs(previous_score - current_score) < self.threshold:
                running = False
                # uncomment this to observe the effect of regularization
                # print(iter)
            previous_score = current_score
            # uncomment those to follow gradient descent
            #print("iter : " + str(iter) + " current cost:" + str(previous_score))
            #print(self.weights)
        if iter == self.max_iter:
            print("Model was unable to reach global minimum")
            print("Please try to modify threshold, alpha or max_iter")

    # update this
    def cost_function_derivative(self):
        delta = np.dot((np.dot(self.weights.T, self.features.T) - self.labels), self.features).T
        delta /= len(self.features)
        return delta

    def cost_function(self, features, labels):
        hypo = np.dot(self.weights.T, features.T) # θTx (more like θTxT)
        vfunc = np.vectorize(lambda x: 1/(1+np.exp(-x)))
        sig = vfunc(hypo) # 1/(1+e^-θTx) (θTxT because of the format)
        total_cost = [-y * np.log(x) - (1 - y) * np.log(1 - x) for y,x in zip(labels, sig[0])]
        regurlarization = self.regul*(self.weights.T[0, 1:]**2) # λ*θT**2 with j = 1:n
        return (sum(total_cost) + (sum(regurlarization)/2)) / len(features)



# don't mind that, just testing code
######################################################################################################################

'''a = np.array([[1,2,3],[3,3,2],[1,4,1],[3,3,1]], dtype=float)
b = np.ones((4,3))
c = np.random.randint(-5,5,(4,3))
theta = np.array([[1],[3],[4]])
htheta = np.dot(a, theta)
labels = np.array([[0,1,1,0]])
delta = 1/len(a) * (np.sum(0))'''
'''print(htheta)
print((htheta - labels.T) * a.T[1])
print(len(a))
print(np.e**2) # this one goes further
print(np.exp(2))
a = np.array(a)
print(a)'''

'''model = LemonRegression()
model.fit(a, labels[0])'''
'''print(model.labels)
print(np.dot(model.weights.T, model.features.T))
print(np.log(1))

hypo = np.dot(model.weights.T, model.features.T)
vfunc = np.vectorize(lambda x: 1/(1+np.exp(-x)))
sig = vfunc(hypo)
print(sig[0])
print(model.features)
print(model.labels)
print(model.score(model.features, model.labels))
print(np.log(.5))

f = np.ones((1,4))
g = np.ones(4)
g /= 2
print(f,g)
print((f - g).T)'''


'''print(model.weights)
print(model.predict(a))
print(model.score(a, labels[0]))'''
'''h = np.ones((4,1))
i = np.ones((4,1))
j = np.insert(h,[1],i,axis=1)
print(i)
print(j)'''
