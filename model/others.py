import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from keras.utils import to_categorical

plt.style.use("ggplot")
mpl.rcParams["axes.unicode_minus"] = False

train = pd.read_csv("C:/bike-sharing-demand-dataset/train.csv", parse_dates=["datetime"])
test = pd.read_csv("C:/bike-sharing-demand-dataset/test.csv", parse_dates=["datetime"])

train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["hour"] = test["datetime"].dt.hour
test["dayofweek"] = test["datetime"].dt.dayofweek

# categorical features
categorical_feature_names = ["season",
                             "holiday",
                             "workingday",
                             "weather",
                             "dayofweek",
                             "month",
                             "year",
                             "hour"]

for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(18, 4)
sns.scatterplot(data=train, x="year", y="count", ax=axes[0])
sns.scatterplot(data=train, x="month", y="count", ax=axes[1])

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

print(train.head())

# pick specific features
X_train = train[feature_names]
X_test = test[feature_names]

# result = bike sharing count
y_train = train["count"]

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)


def rmsle(predicted_values, actual_values, convertExp=True):
    if convertExp:
        predicted_values = np.exp(predicted_values),
        actual_values = np.exp(actual_values)

    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    print("========= predicted values =========")
    print(predicted_values)
    print("========= actual values =========")
    print(actual_values)
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)

    difference = log_predict - log_actual
    difference = np.square(difference)

    mean_difference = difference.mean()
    score = np.sqrt(mean_difference)
    return score

# (1) Linear Regression Model
print("========== Linear Regression ==========")
lModel = LinearRegression()

# for scaling => log1p = log(x+1) => if x=0 -> log1p(x)=0 => min 0 ~
y_train_log = np.log1p(y_train)

lModel.fit(X_train, y_train_log)

preds = lModel.predict(X_train)
print(preds)
print("RMSLE Value For Linear Regression: ", rmsle(np.exp(y_train_log), np.exp(preds), False))

# (2) Ridge Model
print("========== Ridge ==========")
ridge_m_ = Ridge()
ridge_params_ = {
    'max_iter': [3000],
    'alpha': [0.01, 0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000]
}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
# GridSearchCV => find the best alpha value (hyper parameter)
grid_ridge_m = GridSearchCV(ridge_m_, ridge_params_, scoring=rmsle_scorer, cv=5)

y_train_log = np.log1p(y_train)

grid_ridge_m.fit(X_train, y_train_log)

preds = grid_ridge_m.predict(X_train)
print(preds)
print(grid_ridge_m.best_params_)

print("RMSLE Value For Ridge Regression: ", rmsle(np.exp(y_train_log), np.exp(preds), False))
df = pd.DataFrame(grid_ridge_m.cv_results_)
df.head()

df["alpha"] = df["params"].apply(lambda x: x["alpha"])
df["rmsle"] = df["mean_test_score"].apply(lambda x: -x)
df[["alpha", "rmsle"]].head()

fig, ax = plt.subplots()
fig.set_size_inches(12, 5)
plt.xticks(rotation=30, ha='right')
sns.pointplot(data=df, x="alpha", y="rmsle", ax=ax)

# (3) Lasso Model
print("========== Lasso ==========")
lasso_m_ = Lasso()
alpha = 1 / np.array([0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])
lasso_params_ = {
    'max_iter': [3000],
    'alpha': alpha
}
grid_lasso_m = GridSearchCV(lasso_m_, lasso_params_, scoring=rmsle_scorer, cv=5)

y_train_log = np.log1p(y_train)

grid_lasso_m.fit(X_train, y_train_log)

preds = grid_lasso_m.predict(X_train)
print(preds)
print(grid_lasso_m.best_params_)
print("RMSLE Value For Lasso Regression: ", rmsle(np.exp(y_train_log), np.exp(preds), False))

df = pd.DataFrame(grid_lasso_m.cv_results_)
df["alpha"] = df["params"].apply(lambda x: x["alpha"])
df["rmsle"] = df["mean_test_score"].apply(lambda x: -x)
df[["alpha", "rmsle"]].head()

fig, ax=plt.subplots()
fig.set_size_inches(12, 5)
plt.xticks(rotation=30, ha='right')
sns.pointplot(data=df, x="alpha", y="rmsle", ax=ax)








