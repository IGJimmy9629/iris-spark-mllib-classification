# Iris Classification using Spark MLlib

## Project Overview

This project applies Spark MLlib to perform a classification task on the Iris dataset. The objective is to predict Iris flower species based on sepal and petal measurements using three machine learning classification algorithms:

1. Decision Tree
2. Random Forest
3. Logistic Regression

The project was developed using VS Code, Jupyter Notebook, Python, and PySpark.

## Dataset Description

The Iris dataset contains 150 observations and five main variables:

| Variable | Description |
|---|---|
| sepal_length | Length of the sepal |
| sepal_width | Width of the sepal |
| petal_length | Length of the petal |
| petal_width | Width of the petal |
| species | Iris flower species |

The target variable is `species`, which consists of three classes:

- setosa
- versicolor
- virginica

## Methodology

The project follows a complete Spark MLlib classification workflow:

1. Load the Iris dataset using scikit-learn.
2. Convert the dataset into a Spark DataFrame.
3. Perform exploratory data analysis.
4. Validate input data before model training.
5. Convert the target variable using `StringIndexer`.
6. Combine feature columns using `VectorAssembler`.
7. Split the dataset into training and testing sets.
8. Train three Spark MLlib classification models:
   - Decision Tree
   - Random Forest
   - Logistic Regression
9. Apply hyperparameter tuning using `ParamGridBuilder`.
10. Apply cross-validation using `CrossValidator`.
11. Generate predictions on the testing dataset.
12. Evaluate model performance using accuracy, weighted precision, weighted recall, and F1-score.
13. Compare model performance using evaluation metrics and visualisations.
14. Analyse confusion matrices, misclassification, feature importance, and per-class accuracy.
15. Select the best-performing model based on F1-score.

## Hyperparameter Tuning

Hyperparameter tuning was performed for all three models.

### Decision Tree

- `maxDepth`: `[2, 3, 4, 5]`
- `impurity`: `["gini", "entropy"]`

### Random Forest

- `numTrees`: `[10, 20, 50]`
- `maxDepth`: `[2, 3, 5]`

### Logistic Regression

- `regParam`: `[0.01, 0.1, 0.5]`
- `elasticNetParam`: `[0.0, 0.5, 1.0]`

Grid search and cross-validation were used to identify the best-performing parameter combination for each model.

## Results Summary

The tuned models were evaluated using accuracy, weighted precision, weighted recall, and F1-score.

| Model | Accuracy | Weighted Precision | Weighted Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | 0.9310 | 0.9310 | 0.9310 | 0.9310 |
| Random Forest | 0.9655 | 0.9698 | 0.9655 | 0.9658 |
| Logistic Regression | 0.9310 | 0.9310 | 0.9310 | 0.9310 |

Random Forest achieved the best overall performance with the highest accuracy, weighted precision, weighted recall, and F1-score.

## Key Findings

The Iris dataset is balanced, with 50 observations for each species. This allows the classification models to learn from each class fairly.

Exploratory data analysis showed that petal length and petal width provide stronger separation between species compared to sepal measurements.

Feature importance analysis confirmed that petal width and petal length are the most influential predictors in the Random Forest model.

Random Forest produced only one misclassification out of 29 testing observations, making it the most reliable model in this project.

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

## Project Structure

```text
iris-spark-mllib-classification/
│
├── notebook/
│   └── iris_spark_mllib_classification.ipynb
│
├── data/
│   └── iris.csv
│
├── output/
│   ├── model_results.csv
│   ├── training_times.csv
│   ├── species_distribution.png
│   ├── petal_scatter_plot.png
│   ├── sepal_scatter_plot.png
│   ├── correlation_heatmap.png
│   ├── model_comparison_plot.png
│   ├── Decision_Tree_confusion_matrix.png
│   ├── Random_Forest_confusion_matrix.png
│   ├── Logistic_Regression_confusion_matrix.png
│   └── feature_importance.png
│
├── config.py
├── README.md
├── requirements.txt
└── .gitignore