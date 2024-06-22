import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)  # For reproducibility

num_rows = 1000

ages = np.random.randint(18, 70, size=num_rows)
impressions = np.random.randint(100, 1000, size=num_rows)
clicks = np.random.randint(5, 100, size=num_rows)
genders = np.random.choice(['male', 'female'], size=num_rows)
dates = pd.date_range(start='2024-01-01', periods=num_rows)

# Create DataFrame
data = {
    'age': ages,
    'impressions': impressions,
    'clicks': clicks,
    'gender': genders,
    'date': dates
}
df = pd.DataFrame(data)

# Define age categories
age_bins = [0, 18, 25, 35, 45, 55, 65, np.inf]
age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

# Create 'age_group' column
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# Store data in CSV
df.to_csv('synthetic_data.csv', index=False)
