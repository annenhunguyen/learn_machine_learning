"""
Load the CSV file and prepare the training and testing data

1. Download the file
2. Read the CSV using pandas
3. Convert the 
"""
import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print(train_df.description)
