# load lots of visualizing packages
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno

# ggplot for expressing range of numbers in graph clearly
plt.style.use('ggplot')
# solution for a problem which minus font is broken in graph
mpl.rcParams['axes.unicode_minus'] = False

train = pd.read_csv("C:/bike-sharing-demand-dataset/train.csv", parse_dates=["datetime"])
print(train)
print(train.shape)
print(train.info)
print(train.head())

# check specific column -> temp
print(train.temp.describe())

# null check -> each columns
print(train.isnull().sum())

# na check -> each columns
# na = nan -> undefined value
print(train.isna().sum())

# visualizing null check
msno.matrix(train, figsize=(12, 5))

# make extra columns for datetime in detail
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second

print(train.head())

# using alias for axis
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(18, 8)

# 2 x 3 => 6 graphs
sns.barplot(data=train, x="year", y="count", ax=ax1)
sns.barplot(data=train, x="month", y="count", ax=ax2)
sns.barplot(data=train, x="day", y="count", ax=ax3)
sns.barplot(data=train, x="hour", y="count", ax=ax4)
sns.barplot(data=train, x="minute", y="count", ax=ax5)
sns.barplot(data=train, x="second", y="count", ax=ax6)

ax1.set(ylabel="count", title="연도별 대여량")
ax2.set(ylabel="month", title="월별 대여량")
ax3.set(ylabel="day", title="일별 대여량")
ax4.set(ylabel="hour", title="시간별 대여량")

# nrows, ncols => grid numbers
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=train, y="count", orient="v", ax=axes[0][0])
sns.boxplot(data=train, y="count", x="season", ax=axes[0][1])
sns.boxplot(data=train, y="count", x="hour", ax=axes[1][0])
sns.boxplot(data=train, y="count", x="workingday", ax=axes[1][1])

train["dayofweek"] = train["datetime"].dt.dayofweek
print(train["dayofweek"].value_counts())

# nrows=5, ncols=none => grids to 5rows (wide graph)
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5)
fig.set_size_inches(18, 25)

# hue => you can compare data by each of hue values (e.g) hourly count of spring, summer, autumn, winter
sns.pointplot(data=train, x="hour", y="count", ax=ax1)
sns.pointplot(data=train, x="hour", y="count", hue="workingday", ax=ax2)
sns.pointplot(data=train, x="hour", y="count", hue="dayofweek", ax=ax3)
sns.pointplot(data=train, x="hour", y="count", hue="weather", ax=ax4)
sns.pointplot(data=train, x="hour", y="count", hue="season", ax=ax5)

# make a heat map for checking correlation of each columns
corrMatt = train[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]]
corrMatt = corrMatt.corr()
print(corrMatt)

mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
sns.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)

# scatter plot
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
fig.set_size_inches(12, 5)
sns.regplot(x="temp", y="count", data=train, ax=ax1)
sns.regplot(x="windspeed", y="count", data=train, ax=ax2)
sns.regplot(x="humidity", y="count", data=train, ax=ax3)

# yyyy-MM
def concatenate_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)

train["year_month"] = train["datetime"].apply(concatenate_year_month)

print(train.shape)
train[["datetime", "year_month"]].head()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(18, 4)
sns.barplot(data=train, x="year", y="count", ax=ax1)
sns.barplot(data=train, x="month", y="count", ax=ax2)

fig, ax3 = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(18, 4)
sns.barplot(data=train, x="year_month", y="count", ax=ax3)

# remove outlier data
# std = standard value
# mean = average value
# so, too small or too large value of count is removed following below calculation
trainWithoutOutliers = train[np.abs(train["count"] - train["count"].mean()) <= (3 * train["count"].std())]
print(train.shape)
print(trainWithoutOutliers.shape)

figure, axes = plt.subplots(ncols=2, nrows=2)
figure.set_size_inches(12, 10)

# 0,0 fig => with outlier
sns.distplot(train["count"], ax=axes[0][0])
# 0,1 fig
# norm = normalization
stats.probplot(train["count"], dist='norm', fit=True, plot=axes[0][1])
# 1,0 fig => without outlier
# np.log => to ease the imbalance of count-dist
sns.distplot(np.log(trainWithoutOutliers["count"]), ax=axes[1][0])
# 1,1 fig
stats.probplot(np.log1p(trainWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])

