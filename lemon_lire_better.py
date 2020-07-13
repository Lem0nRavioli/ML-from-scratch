import random
import numpy as np


class LemonRegression:
    def __init__(self, alpha=.1, max_iter=20000, threshold=10**-10, init_random=False, bias=.5,regularization=0):
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

    #update this
    def predict(self, features):
        features = np.array(features)
        # add 1 for the intercept multiplier
        features = np.insert(features, 0, 1, axis=1)
        return np.dot(features,self.weights)

    # update this
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
            regularization = self.regul*self.weights[1:] / len(self.features)
            regularization = np.insert(regularization, 0, [0], axis=0)
            self.weights -= self.alpha * (self.cost_function_derivative() + regularization)
            current_score = self.cost_function(self.features,self.labels)
            if abs(previous_score-current_score) < self.threshold:
                running = False
                # uncomment this to observe the effect of regularization
                #print(iter)
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
        cost = (hypo[0] - labels)**2
        regularization = self.regul*(self.weights.T[0, 1:]**2)
        return (sum(cost) + sum(regularization)) / len(features)


# Testing code, don't mind this
#######################################################################################################################
'''a = np.array([1,2,3])
a = a**2
print(a)
features = np.array([[1,2,1,4],[2,2,4,4],[4,1,16,1],[3,2,9,4]])
labels = np.array([5,8,13,11])
model = LemonRegression(regularization=1000,alpha=.001,max_iter=2000000)
model.fit(features,labels)
print(model.weights_())
print(model.predict([[2,4,4,16],[3,3,9,9]]))'''

'''features = np.array([[1,2,1,4],[2,2,4,4],[4,1,16,1],[3,2,9,4]])
features = np.insert(features,0,[1,1,54,123], axis=0)
print(features)'''
