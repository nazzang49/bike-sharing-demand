import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

# evaluation function
def rmsle(predicted_values, actual_values):
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)

    diff = log_predict - log_actual
    diff = np.square(diff)

    mean_diff = diff.mean()
    final_score = np.sqrt(mean_diff)
    print(final_score)
    return final_score

# global setting
mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

# call dataset
train = pd.read_csv("C:/bike-sharing-demand-dataset/train.csv", parse_dates=["datetime"])
test = pd.read_csv("C:/bike-sharing-demand-dataset/test.csv", parse_dates=["datetime"])

print(train.head())
print(test.head())
print(train.shape)
print(test.shape)

train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second
train["dayofweek"] = train["datetime"].dt.dayofweek

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour
test["minute"] = test["datetime"].dt.minute
test["second"] = test["datetime"].dt.second
test["dayofweek"] = test["datetime"].dt.dayofweek

print(train.shape)
print(test.shape)
print(train["dayofweek"].head())
print(test["dayofweek"].head())

fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(18, 10)
plt.sca(axes[0])
plt.xticks(rotation=30, ha="right")
axes[0].set(ylabel="count", title="train windspeed")
sns.countplot(data=train, x="windspeed", ax=axes[0])

# windspeed 0 value => might be not measured => need adjustment
    # (method_1) mean of windspeed
    # (method_2) by machine learning => better
def predict_windspeed_by_random_forest_classifier(data):

    # divide the dataset into windspeed == 0 and != 0
    dataWind0 = data.loc[data['windspeed'] == 0]
    dataWindNot0 = data.loc[data['windspeed'] != 0]
    print(dataWind0.shape)
    print(dataWindNot0.shape)

    # pick some features for predicting windspeed
    wCol = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]

    # windspped != 0 data type => str
    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")

    # call Random Fores Classifier and begin to learning
    rfModel_wind = RandomForestClassifier()
    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])

    # prediction
    wind0Values = rfModel_wind.predict(X=dataWind0[wCol])
    predictWind0 = dataWind0
    predictWindNot0 = dataWindNot0
    predictWind0["windspeed"] = wind0Values

    # windspeed == 0 doesn't exist anymore
    data = predictWindNot0.append(predictWind0)
    data["windspeed"] = data["windspeed"].astype("float")
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)

    return data

# call function for adjusting windspeed value
train = predict_windspeed_by_random_forest_classifier(train)
# test = predict_windspeed(test)

# check visual
fig, ax1 = plt.subplots()
fig.set_size_inches(18,6)
plt.sca(ax1)
plt.xticks(rotation=30, ha='right')
ax1.set(ylabel='Count',title="train windspeed")
sns.countplot(data=train, x="windspeed", ax=ax1)

# feature selection
    # add feature one by one
    # if performance isn't good => remove that feature

# feature type
    # continuous (e.g) temp is continuous => can get all float value
    # categorical (e.g) season is categorical => only 4 seasons
categorical_feature_names = ["season",
                             "holiday",
                             "workingday",
                             "weather",
                             "dayofweek",
                             "month",
                             "year",
                             "hour"]

# raw => one hot encoding
for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")

print(train.head())

feature_names = ["season",
                 "weather",
                 "temp",
                 "atemp",
                 "humidity",
                 "windspeed",
                 "year",
                 "hour",
                 "dayofweek",
                 "holiday",
                 "workingday"]

# prepare input X
X_train = train[feature_names]
X_test = test[feature_names]

# prepare output Y
y_train = train["count"]

# evaluation model selection
rmsle_scorer = make_scorer(rmsle)
print(rmsle_scorer)

# cross validation by KFold method for checking generalization performance / dataset balance
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.ensemble import RandomForestRegressor
max_depth_list = []
# n_estimators => size of trees
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
print(model)

# check mean of score (this project => 10 steps validation)
score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)
score = score.mean()
# the smaller values are, the better learning model is
print("Score= {0:.5f}".format(score))

# predict real value of count
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions.shape)
print(predictions[:100])

fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(12, 5)
sns.distplot(y_train, ax=ax1, bins=50)
ax1.set(title="train")
sns.distplot(predictions, ax=ax2, bins=50)
ax2.set(title="test")


