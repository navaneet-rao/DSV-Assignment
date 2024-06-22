import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

null_counts = df.isnull().sum()
columns_with_nulls = null_counts[null_counts > 0]
print("Columns containing null values:\n", columns_with_nulls)


print("\nHead of the dataset:\n", df.head())
print("\nTail of the dataset:\n", df.tail())
print("\nInfo of the dataset:\n", df.info())
print("\nDescription of the dataset:\n", df.describe())
print("\nShape of the dataset:\n", df.shape)


numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


sns.pairplot(df.dropna())
plt.show()


# Fill missing values for simplicity (e.g., median for age, mode for embarked)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Display histograms of numerical features
df.hist(bins=20, figsize=(14, 10), grid=False)
plt.show()

# Visualize categorical features
plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Passenger Class Distribution by Survival')
plt.show()

plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Gender Distribution by Survival')
plt.show()

plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='Embarked', hue='Survived')
plt.title('Embarked Port Distribution by Survival')
plt.show()
