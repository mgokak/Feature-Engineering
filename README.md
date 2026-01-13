# Data Preprocessing & Feature Engineering Practice Repository

## Overview

This repository focuses on understanding and practicing **data preprocessing and feature engineering techniques** that are essential for building reliable machine learning models. The Jupyter Notebooks in this project demonstrate how to handle common real-world data issues such as missing values, class imbalance, outliers, and categorical encoding.

The goal of this repository is to build a strong foundation in preparing data before modeling, since well-prepared data often has a bigger impact on model performance than the choice of algorithm itself.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. Handling Missing Values  
4. Handling Outliers  
5. Handling Imbalanced Datasets  
6. Encoding Categorical Variables  

---

## Installation

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
```

---

## Project Structure

- `Handling_Missing_values.ipynb`
- `Handling_Outliers.ipynb`
- `Handling_Imbalance_Dataset.ipynb`
- `SMOTE.ipynb`
- `Nominal_or_One_Hot_Encoding.ipynb`
- `Target_Guided_Ordinal_Encoding.ipynb`

---

## Handling Missing Values

### `Handling_Missing_values.ipynb`

This notebook covers:
- Identifying missing values
- Handling missing data using mean, median, and mode
- Understanding when to drop vs impute missing values
- Impact of missing values on model performance

Common commands used:
```python
df.isnull().sum()
df.fillna(df.mean())
df['column'].fillna(df['column'].median())
df.dropna()
```

Used to identify, impute, or remove missing data.

---

## Handling Outliers

### `Handling_Outliers.ipynb`

This notebook focuses on:
- Detecting outliers using statistical methods
- Visualizing outliers with boxplots
- Handling outliers using capping, removal, or transformation
- Understanding how outliers affect models

Common commands used:
```python
df.describe()
df.boxplot(column='feature')
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
```

Used to detect and handle extreme values.

---

## Handling Imbalanced Datasets

### `Handling_Imbalance_Dataset.ipynb`

This notebook explains:
- What imbalanced datasets are
- Why imbalance causes biased models
- Basic techniques to handle class imbalance
- Evaluating models on imbalanced data

Common commands used:
```python
df['target'].value_counts()
from sklearn.utils import resample
```

Used to understand class imbalance and apply basic resampling techniques.

---

## SMOTE (Synthetic Minority Oversampling Technique)

### `SMOTE.ipynb`

This notebook demonstrates:
- Oversampling using SMOTE
- How synthetic samples are generated
- Improving classification performance on minority classes
- Comparing results before and after SMOTE

Common commands used:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

Used to generate synthetic samples for minority classes.

---

## Encoding Categorical Variables

### One-Hot Encoding  
#### `Nominal_or_One_Hot_Encoding.ipynb`

Covers:
- Encoding nominal categorical variables
- One-hot encoding using pandas and scikit-learn
- Avoiding the dummy variable trap

Common commands used:
```python
pd.get_dummies(df, drop_first=True)
from sklearn.preprocessing import OneHotEncoder
```

Used for encoding nominal categorical variables.

---

### Target-Guided Ordinal Encoding  
#### `Target_Guided_Ordinal_Encoding.ipynb`

Covers:
- Encoding ordinal variables based on target statistics
- Reducing dimensionality compared to one-hot encoding
- Use cases and potential risks of target leakage

Common commands used:
```python
df.groupby('category')['target'].mean()
df['encoded'] = df['category'].map(mapping)
```

Used to encode ordinal variables based on target statistics.

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  
DePaul University
