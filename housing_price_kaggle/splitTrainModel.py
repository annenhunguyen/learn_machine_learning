import numpy as np
import pandas as pd
import itables

df = pd.read_csv("housing_price_kaggle/data/train.csv")
print(df.shape)
df.head(5)

##########  Split df into train and test ------------
## 1) Shuffle the df
shuffled_df = df.sample(frac=1, random_state=1).reset_index(drop=True)

## 2) Set the split ratio 80% for training
split_ratio = 0.8
df_rows, df_columns = df.shape
split_index = int(df_rows * split_ratio)
print(split_index)

## 3) Split the df into train and test
train_df = shuffled_df.iloc[:split_index, :]
print(f"train df: {train_df.shape}")

test_df = df.iloc[split_index:, :]
print(f"test df: {test_df.shape}")

########################################
# -- TRAINING --- #

########## Get numerical columns -----------
train_df_num = train_df.select_dtypes(include=["number"]).dropna()
print(f"train df numerical: {train_df_num.shape}")
train_df_num.head(5)

######### Set target vector ------------

target_column = "SalePrice"
target = train_df_num[target_column].to_numpy()

## Reshape target to a (X rows, 1)
target = target.reshape(-1, 1)

## Normalize values
target = (target - target.min()) / (target.max() - target.min())
print(f"target size: {target.shape}")

########### Get feature matrix -------------------

features_df = train_df_num.drop(target_column, axis=1)
for c in features_df.columns:
    features_df[c] = (features_df[c] - features_df[c].min()) / (
        features_df[c].max() - features_df[c].min()
    )
features = features_df.to_numpy()
print(f"features size: {features.shape}")
features_df.head(5)

############ Solve for W -----------------------

## Set W0
features_datapoints, features_count = features.shape
W0 = np.random.rand(features_count, 1)

## Get model
from model import solve_parameters, gradient_descent

W = gradient_descent(X=features, Y=target, W=W0)


######################################################
###### --- EVALUATE --- ###


## Get numerical df from test df ------------
test_df_num = test_df.select_dtypes(include=["number"])
print(f"test_df_num size: {test_df_num}")

## Get the true result (Y)
result_df = test_df[target]
