import lemon_lire
import lemon_lore
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pandas as pd


pd.options.display.width = None
#pd.set_option('display.max_rows', 200)

'''train = pd.read_csv('../HousePricing/train.csv')
test = pd.read_csv('../HousePricing/test.csv')
train_feature = train[['LotArea', 'LotFrontage']].copy().dropna()
print(train_feature.count())
labels = train.SalePrice.copy().tolist()
print(train_feature.head())
scaler = MinMaxScaler()
train_feature_scaled = scaler.fit_transform(train_feature).tolist()

print(train_feature_scaled[:5])

my_model = lemon_lire.LemonRegression(max_iter=200)
my_model.fit(train_feature_scaled, labels)
print(my_model.score(train_feature_scaled,labels))'''

'''features = [[4],
            [6],
            [12],
            [2]]
labels = [9,13,25,5]

model = lemon_lire.LemonRegression(max_iter=20000, alpha=.01, threshold=0)
model.fit(features,labels)
print(model.score(features,labels))'''

train = pd.read_csv('../Titanic/train.csv')
test = pd.read_csv('../Titanic/test.csv')
combine = [train, test]
for df in combine:
    df.loc[(df.Sex == 'male') & df.Age.isnull(), 'Age'] = df[df.Sex == 'male'].Age.mean()
    df.loc[(df.Sex == 'female') & df.Age.isnull(), 'Age'] = df[df.Sex == 'female'].Age.mean()
    df.dropna(subset=['Embarked'], inplace=True)
    df.Fare.fillna(df.Fare.mean(), inplace=True)
    df['num_family'] = df.SibSp + df.Parch
    df['AgeClass'] = df.Age * df.Pclass
    df.Sex = df.Sex.map({'male': 1, 'female': 2})
    df.Embarked = df.Embarked.map({'S': 1, 'C': 2, 'Q': 3})


# FINAL SEPARATION
###############
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'num_family', 'Fare', 'Embarked', 'AgeClass']
features_train = train[features]
features_test = test[features]
labels_train = train.Survived
scaler = preprocessing.MinMaxScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

X_train,X_test,y_train,y_test = train_test_split(features_train_scaled, labels_train, test_size=.1)


model = lemon_lore.LemonRegression(max_iter=20000)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
