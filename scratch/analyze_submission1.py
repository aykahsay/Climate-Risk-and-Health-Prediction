import pandas as pd
import numpy as np

# Let's inspect Train and Test details
train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Let's check target balance in train
print("Train target value counts:")
print(train['is_climate_sensitive'].value_counts(normalize=True))

# Let's check location distribution in Train vs Test
print("\nTrain locations:")
print(train['location'].value_counts())
print("\nTest locations:")
print(test['location'].value_counts())
