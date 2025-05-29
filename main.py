import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tc = pd.read_csv("Cleaned-Titanic.csv")     #importing dataset

print("=== Mean ===")
print(tc.mean(numeric_only=True))

print("\n=== Median ===")
print(tc.median(numeric_only=True))

print("\n=== Mode ===")
print(tc.mode(numeric_only=True).iloc[0])

numeric_cols = ['Survived','Age']

# Set Seaborn style
sns.set(style="whitegrid")

# Create Histograms
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(tc[col].dropna(), kde=True, bins=30, color='skyblue')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Create Boxplots
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=tc[col], color='salmon')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

sns.pairplot(tc[['Age'] + ['Survived']].dropna())
plt.suptitle("Pairplot of Numeric Features (Colored by Survived)", y=1.02)
plt.show()

correlation = tc[['Age'] + ['Survived']].corr()

# Read that a heatmap is a must after co relating
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Titanic Features")
plt.show()


#trends

#more woman survived
print(tc['Sex'].value_counts())
print(tc.groupby('Sex')['Survived'].mean())

#higher the class, more people survived
print(tc['Pclass'].value_counts())
print(tc.groupby('Pclass')['Survived'].mean())

#most deaths were of people around 30
sns.histplot(data=tc, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age Distribution by Survival')
plt.show()

#anomalaties

sns.boxplot(data=tc, x='Fare')
plt.title('Boxplot of Fare')
plt.show()
#can see outliers, indicating very high fares








