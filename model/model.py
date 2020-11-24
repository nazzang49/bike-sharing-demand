# load lots of visualizing packages
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
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








