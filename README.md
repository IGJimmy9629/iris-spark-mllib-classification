# Iris Classification using Spark MLlib

## Project Overview

This project uses Spark MLlib to classify Iris flower species based on sepal and petal measurements. The project is developed using VS Code, Jupyter Notebook, Python, and PySpark.

Three classification models are implemented:

1. Decision Tree
2. Random Forest
3. Logistic Regression

The project includes data loading, exploratory data analysis, preprocessing, model training, hyperparameter tuning, evaluation, comparative analysis, and conclusion.

## Dataset Description

The Iris dataset contains 150 records and five main variables:

| Variable | Description |
|---|---|
| sepal_length | Length of sepal |
| sepal_width | Width of sepal |
| petal_length | Length of petal |
| petal_width | Width of petal |
| species | Iris flower species |

The target variable is `species`, which contains three classes:

- setosa
- versicolor
- virginica

## Methodology

The project follows these steps:

1. Load Iris dataset using scikit-learn.
2. Convert the dataset into a Spark DataFrame.
3. Perform exploratory data analysis.
4. Convert species labels into numeric format using StringIndexer.
5. Combine feature columns using VectorAssembler.
6. Split the dataset into training and testing sets.
7. Train three classification models:
   - Decision Tree
   - Random Forest
   - Logistic Regression
8. Apply hyperparameter tuning using CrossValidator and ParamGridBuilder.
9. Evaluate models using:
   - Accuracy
   - Weighted precision
   - Weighted recall
   - F1-score
10. Select the best model based on the highest F1-score.

## Tools and Technologies

- VS Code
- Jupyter Notebook
- Python
- PySpark
- Spark MLlib
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- GitHub

## Results Summary

The tuned models were evaluated using accuracy, weighted precision, weighted recall, and F1-score.

| Model | Accuracy | Weighted Precision | Weighted Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | 0.9310 | 0.9310 | 0.9310 | 0.9310 |
| Random Forest | 0.9655 | 0.9698 | 0.9655 | 0.9658 |
| Logistic Regression | 0.9310 | 0.9310 | 0.9310 | 0.9310 |

Random Forest achieved the highest overall performance and was selected as the best-performing model.

## Key Findings

The Iris dataset is balanced, with equal records for each species. The visual analysis showed that petal length and petal width are useful features for separating Iris species.

The confusion matrices showed that most observations were correctly classified. Minor misclassification occurred between similar species classes.

Random Forest performed best because it combines multiple decision trees, improves prediction stability, and reduces overfitting compared to a single Decision Tree.

## How to Reproduce This Project

1. Clone this repository:

```bash
git clone https://github.com/yourusername/iris-spark-mllib-classification.git