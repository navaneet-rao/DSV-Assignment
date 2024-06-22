import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset URL
data_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# Load the dataset
df = pd.read_csv(data_url)

# Display the first few rows of the dataset to verify it loaded correctly
print(df.head())

# Compute overall mean and standard deviation for each measurement column
overall_mean = df.iloc[:, :-1].mean()
overall_std = df.iloc[:, :-1].std()

print("\nOverall Mean and Standard Deviation for each measurement column:")
for col_name, mean, std in zip(df.columns[:-1], overall_mean, overall_std):
    print(f"{col_name}: Mean = {mean:.2f}, Std Dev = {std:.2f}")

# Compute mean and standard deviation for each measurement column, separately for each species
species_mean_std = df.groupby('species').agg(['mean', 'std'])

print("\nMean and Standard Deviation for each measurement column, by species:")
print(species_mean_std)

# Analysis to determine the "best" measurement for guessing the species
print("\nAnalysis to determine the 'best' measurement for guessing the species:")
range_spans = {}
for col_name in df.columns[:-1]:
    species_ranges = []
    for species_name, group in df.groupby('species'):
        species_means = group[col_name].mean()
        species_std_devs = group[col_name].std()
        species_ranges.append((species_means - species_std_devs, species_means + species_std_devs))
    min_range = min([r[0] for r in species_ranges])
    max_range = max([r[1] for r in species_ranges])
    range_span = max_range - min_range
    range_spans[col_name] = range_span
    print(f"{col_name}: Range Span = {range_span:.2f}")

# Convert range spans to a DataFrame for plotting
range_df = pd.DataFrame(list(range_spans.items()), columns=['Measurement', 'Range Span'])

# Plotting the range spans
plt.figure(figsize=(10, 6))
sns.barplot(x='Measurement', y='Range Span', data=range_df, hue='Measurement', palette='viridis', legend=False)
plt.title("Range Span of Measurements for Species Classification")
plt.ylabel("Range Span")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot for each measurement column by species
plt.figure(figsize=(12, 8))
for i, col_name in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=col_name, data=df)
    plt.title(f"{col_name} distribution by species")
plt.tight_layout()
plt.show()

# Pairplot to visualize pairwise relationships and distributions
sns.pairplot(df, hue='species', height=3, aspect=1.2)
plt.suptitle("Pairplot of Iris Dataset by Species", y=1.02)
plt.show()
