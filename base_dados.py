import pandas as pd

# Load the dataset to inspect it
file_path = './base/Dog Breads Around The World.csv'
df = pd.read_csv(file_path)

# Display basic information and the first few rows of the dataset
print(df.info())
print(df.head())