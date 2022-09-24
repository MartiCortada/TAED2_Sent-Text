# Loading the necessary libraries
import numpy as np
import pandas as pd

# We preset data types in advance, to save memory
dtypes = {
        'class' : 'uint8',
        'review_text' : 'str'
}

# Training

train_dataset = pd.read_csv("./All_Data/train.csv", header=None, usecols=[0,2], names=["class", "review_text"],
                            dtype=dtypes)
train_dataset.dropna(inplace=True)
train_dataset = train_dataset.sample(frac=1/3, random_state=0)
train_dataset.reset_index(inplace=True, drop=True)


# Testing

test_dataset = pd.read_csv("./All_Data/test.csv", header=None, usecols=[0,2], names=["class", "review_text"],
                            dtype=dtypes)
test_dataset.dropna(inplace=True)
test_dataset = test_dataset.sample(frac=1/3, random_state=0)
test_dataset.reset_index(inplace=True, drop=True)


# Saving csv files

train_dataset.to_csv('./All_Data/clean_amazon_reviews_train.csv',index=False)
test_dataset.to_csv('./All_Data/clean_amazon_reviews_test.csv',index=False)