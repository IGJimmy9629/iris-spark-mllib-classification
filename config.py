# config.py

# Data and output paths
DATA_PATH = "../data/"
OUTPUT_PATH = "../output/"

# Reproducibility settings
TRAIN_TEST_SPLIT = [0.8, 0.2]
RANDOM_SEED = 42
CV_FOLDS = 3

# Decision Tree hyperparameter grid
DT_MAX_DEPTH_GRID = [2, 3, 4, 5]
DT_IMPURITY_GRID = ["gini", "entropy"]

# Random Forest hyperparameter grid
RF_NUM_TREES_GRID = [10, 20, 50]
RF_MAX_DEPTH_GRID = [2, 3, 5]

# Logistic Regression hyperparameter grid
LR_REG_PARAM_GRID = [0.01, 0.1, 0.5]
LR_ELASTIC_NET_PARAM_GRID = [0.0, 0.5, 1.0]