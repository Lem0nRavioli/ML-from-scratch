import lemon_lire
import lemon_lore
import lemon_nnet_2
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pandas as pd


pd.options.display.width = None
#pd.set_option('display.max_rows', 200)

train = pd.read_csv("train_numbers.csv")
features = train.columns[1:]
features_train = train[features]
labels = train["label"]
labels = [labels]
model = lemon_nnet_2.Lemon_nnet(alpha=40)
layout = [(16, "sig"), (16, "sig"), (16, "sig"), (10, "sig")]
model.fit(layout, features_train, labels)
test_feat = [np.array(features_train)[1]]
print(model.predict(test_feat))
print(labels[0][1])