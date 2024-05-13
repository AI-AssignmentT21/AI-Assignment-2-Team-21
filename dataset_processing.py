import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats
# Load the dataset (assuming it's in CSV format for this example)
file_path = os.path.abspath('Fatality.csv' ) # Adjust the path to where the dataset is stored
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(data.head())
# Check for missing values
print(data.isnull().sum())

# Convert categorical 'yes'/'no' values to numeric values (0/1)
data[ 'jaild' ] = data[ 'jaild' ].map({'yes': 1, 'no': 0})
data[ 'comserd' ] = data[ 'comserd' ].map({'yes': 1, 'no': 0})

# Summary of the data to understand types and potential transformations needed

mean_mrall1 = data[ 'mrall' ].mean()
std_mrall1 = data[ 'mrall' ].std()
mean_vmiles1 = data[ 'vmiles' ].mean()
std_vmiles1 = data[ 'vmiles' ].std()
Q1 = data['vmiles'].quantile(0.25)
Q3 = data['vmiles'].quantile(0.75)
IQR = Q3 - Q1

# Define thresholds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
data_filtered = data[(data['vmiles'] >= lower_bound) & (data['vmiles'] <= upper_bound)]

# Alternatively, capping the outliers instead of removing them:
data['vmiles'] = np.where(data['vmiles'] > upper_bound, upper_bound,
                          np.where(data['vmiles'] < lower_bound, lower_bound, data['vmiles']))

# Z-Score method for handling outliers across multiple columns


# Calculate Z-scores of `vmiles`
z_scores = stats.zscore(data['vmiles'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)  # Filter using threshold of 3
data_clean = data[filtered_entries]

# Mean and standard deviation for key features
mean_mrall = data_clean[ 'mrall' ].mean()
std_mrall = data_clean[ 'mrall' ].std()
mean_vmiles = data_clean[ 'vmiles' ].mean()
std_vmiles = data_clean[ 'vmiles' ].std()


# Using robust statistics for central tendency
median_vmiles = data_clean['vmiles'].median()
print("Median of vmiles:", median_vmiles)


# Display summary of clean data

print(f"Before the cleaning of the data\nMean Traffic Fatality Rate: {mean_mrall1}, Standard Deviation: {std_mrall1}")
print(f"Mean Average Miles per Driver: {mean_vmiles1}, Standard Deviation: {std_vmiles1}")
print(f"After the cleaning of the data\nMean Traffic Fatality Rate: {mean_mrall}, Standard Deviation: {std_mrall}")
print(f"Mean Average Miles per Driver: {mean_vmiles}, Standard Deviation: {std_vmiles}")
print("Clean Data",data_clean.describe())
# Create a figure with subplots arranged in a 2x3 grid
# Full Correlation Matrix
full_correlation_matrix = data.corr(method='pearson')

# Subset Correlation Matrix
# Selecting specific features which might be of particular interest
subset_features = ['mrall', 'vmiles', 'unrate', 'perinc']
subset_correlation_matrix = data[subset_features].corr(method='pearson')

# Create a figure with subplots arranged in a 3x3 grid to include two new plots
fig, axes = plt.subplots(3, 2, figsize=(15, 24))
plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust spacing between plots

# First scatter plot: Traffic Fatality Rate vs. Average Miles per Driver
sns.scatterplot(data=data, x='vmiles', y='mrall', hue='jaild', style='comserd', palette='coolwarm', ax=axes[0, 0])
sns.kdeplot(data=data, x='vmiles', y='mrall', color='gray', levels=5, alpha=0.5, ax=axes[0, 0])
axes[0, 0].set_title('Traffic Fatality Rate vs. Average Miles per Driver')
axes[0, 0].set_xlabel('Average Miles per Driver')
axes[0, 0].set_ylabel('Traffic Fatality Rate (deaths per 10,000)')
axes[0, 0].legend(title='Mandatory Jail Sentence')

# Second scatter plot: Traffic Fatality Rate vs. Beer Tax
sns.scatterplot(data=data, x='beertax', y='mrall', hue='mlda', alpha=0.7, palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('Traffic Fatality Rate vs. Beer Tax')
axes[0, 1].set_xlabel('Beer Tax')
axes[0, 1].set_ylabel('Traffic Fatality Rate (deaths per 10,000)')
axes[0, 1].legend(title='Minimum Legal Drinking Age', bbox_to_anchor=(1.05, 1), loc='upper left')

# First histogram: Distribution of Traffic Fatality Rate
sns.histplot(data['mrall'], kde=True, color='blue', bins=20, kde_kws={"bw_adjust": 0.5}, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Traffic Fatality Rate with KDE')
axes[1, 0].set_xlabel('Traffic Fatality Rate (deaths per 10,000)')
axes[1, 0].set_ylabel('Frequency')

# Second histogram: Distribution of Beer Tax
sns.histplot(data['beertax'], kde=True, color='green', bins=20, kde_kws={"bw_adjust": 0.5}, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Beer Tax with KDE')
axes[1, 1].set_xlabel('Beer Tax')
axes[1, 1].set_ylabel('Frequency')

# Boxplot for Average Miles per Driver
sns.boxplot(x=data['vmiles'], color='lightblue', fliersize=5, whiskerprops={'linewidth':1.5}, ax=axes[2, 0])
axes[2, 0].set_title('Boxplot of Average Miles per Driver')
axes[2, 0].set_xlabel('Average Miles per Driver')

# Boxplot for Unemployment Rate
sns.boxplot(x=data['unrate'], color='lightblue', fliersize=5, whiskerprops={'linewidth':1.5}, ax=axes[2, 1])
axes[2, 1].set_title('Boxplot of Unemployment Rate')
axes[2, 1].set_xlabel('Unemployment Rate')
plt.show()
sns.heatmap(full_correlation_matrix, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Full Feature Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()
# Subset Correlation Matrix Heatmap
sns.heatmap(subset_correlation_matrix, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Subset Feature Correlation Matrix')
plt.xlabel('Selected Features')
plt.ylabel('Selected Features')
plt.show()
# Display the figure with all subplots
