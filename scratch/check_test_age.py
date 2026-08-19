import pandas as pd

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')

print("Train age < 5 ratio:", (train['age'] < 5).mean(), "Count:", (train['age'] < 5).sum())
print("Test age < 5 ratio: ", (test['age'] < 5).mean(), "Count:", (test['age'] < 5).sum())

print("\nTrain age == 0 ratio:", (train['age'] == 0).mean())
print("Test age == 0 ratio: ", (test['age'] == 0).mean())
