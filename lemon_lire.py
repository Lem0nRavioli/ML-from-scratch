import random


class LemonRegression:
    def __init__(self, alpha=.000001, max_iter=200, threshold=0.0001, init_random=False):
        self.alpha = alpha
        self.weights = []
        self.labels = []
        self.features = []
        self.max_iter = max_iter
        self.threshold = threshold
        self.init_random = init_random

    def fit(self, features, labels, random_state=None):
        random.seed(random_state)
        # add 1 for the interecept multiplier
        [feature.insert(0,1) for feature in features]
        self.features = features
        self.weights = [0 for i in range(len(features[0]))] # random.random()
        if self.init_random: self.weights = [random.random() for i in range(len(features[0]))]
        self.labels = labels
        self.gradient_descent()

    def predict(self, features):
        [feature.insert(0, 1) for feature in features]
        return [sum([w*x for w, x in zip(self.weights, feature)]) for feature in features]

    def score(self, features, labels):
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
            temp_new_weights = []
            for i in range(len(self.weights)):
                temp_new_weights.append(self.weights[i] - self.alpha * self.cost_function_derivative(i))
            self.weights = temp_new_weights
            current_score = self.cost_function(self.features,self.labels)
            if abs(previous_score-current_score) <= self.threshold : running = False
            previous_score = current_score
            # uncomment those to follow gradient descent
            print("iter : " + str(iter) + " current cost:" + str(previous_score))
            print(self.weights)

    def cost_function_derivative(self, i):
        cost = sum([(sum([w*x for w,x in zip(self.weights, feature)]) - label) * feature[i]
                    for feature,label in zip(self.features, self.labels)])
        #cost = sum((sum(w*x for w,x in zip(self.weights, feature)) - y) * feature[i] for feature, y in zip(self.features, self.labels))
        return cost / len(self.features)

    def cost_function(self, features, labels):
        cost = sum([(sum([w * x for w, x in zip(self.weights, feature)]) - label) ** 2
                    for feature, label in zip(features, labels)])
        #cost = sum((sum(w * x for w, x in zip(self.weights, feature)) - y) ** 2 for feature, y in zip(features, labels))
        return cost / (2 * len(self.features))





# testing code
'''ez_feat = [[4,1],[8,1],[2,1],[10,1]]
sample_feat = [[1,2,3],[3,3,2],[1,4,1],[3,3,1]]
sample_labels = [2,4,1,5]
model = LemonRegression()
model.fit(sample_feat, sample_labels, random_state=1)
print(model.predict([[3,2,1]]))
print(model.score([[1,2,3],[3,3,2]], [2,4]))

'''