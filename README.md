# 4th Place Accra Mobility Prediction Hackathon
<img width="100%" height="720" alt="thumb_8672472f-44a6-4d57-8e03-587b7fcb3099" src="https://github.com/user-attachments/assets/53205fc3-00e5-48ca-96d6-2f34112ec0ff" />


## 1. Overview and Objectives
Solution Purpose: The task involves predicting the average speed of vehicles in Accra based on temporal and spatial features. The solution aims to support urban mobility optimization, reduce traffic congestion, and enhance decision-making for transportation planning.
### Objectives:
Achieve high predictive accuracy with a reproducible workflow.

Provide a scalable, understandable solution for real-world deployment.

Ensure the submission adheres to Zindi’s code review standards.
## 2. Repository Structure
<code>accra-mobility-prediction-hackathon.ipynb</code>: Notebook containing the full implementation of the solution (data preprocessing, model training, and evaluation).

<code>requirements.txt</code>: List of dependencies with versions for reproducibility.

<code>Dataset</code>: Place the dataset file in the directory where the notebook resides. Specify paths in the notebook if needed.
## 3. Architecture Diagram

The solution flow:

### Extract: 
Dataset is loaded directly from the Zindi-provided files.
### Transform: 
Includes data cleaning, feature engineering, and scaling.
### Modeling: 
Machine learning model (Ensemble model of LightGBM and Catboost gradient boosting models) is trained and validated.
### Inference: 
Predictions are generated and saved in the required submission format.

## 4. Environment Setup
Install Python 3.11.

### Install dependencies with:
```bash
pip install -r requirements.txt
```
Use a Jupyter Notebook or compatible environment (e.g., Colab, Kaggle).

### Environment where this solution was developed:

Platform: Kaggle

GPU: None

RAM: 30 GB

Python Version: 3.11

## 5. Data Preprocessing (ETL Process)
### Extract:

1. Data loaded in CSV format as provided on the Zindi platform.
2. Ensure data files are in the correct paths before running the notebook.
### Transform:

Cleaning: Missing values and outliers addressed.

## Feature Engineering:

Merged the graph.csv file on these columns ("persistent_id","length", "speed_limit", "segments","category","is_residential", "traffic_side") and the train.csv.
```bash
def look_up_from_graph(train,graph,lookup_columns):
  # returns dataframe by looking up columns from the graph.csv

  train = train.copy()
  graph = graph.copy()

  train = train.merge(graph[lookup_columns], on = "persistent_id", how = "left")

  return train

"""
Save columns to lookup from the graph dataset, I found that these did not work well even though they seem very useful -- they lead to overfitting quite easily didn't have time to investigate why this happened.
The best model should include at least a subset of these
"""
preliminary_columns = ["persistent_id","length", "speed_limit", "segments","category","is_residential", "traffic_side"]

train = look_up_from_graph(train,graph_df,preliminary_columns)
train.info()
```
Change objects columns to category and return dataframe and list
```bash
def change_object_to_cat(df):
  # changes objects columns to category and returns dataframe and list

  df = df.copy()
  list_str_obj_cols = df.columns[df.dtypes == "object"].tolist()
  for str_obj_col in list_str_obj_cols:
      df[str_obj_col] = df[str_obj_col].astype("category")

  return df,list_str_obj_cols
```

### Normalization
The count is already normalized from the dataset. (eg. count_norm_00_3_)

### Splitting: 
Data split into training (80%) and validation (20%).
```bash
# get features and target
X =  train.drop(['ID', 'persistent_id', 'target'], axis = 1)
y = train.target

X,cat_list = change_object_to_cat(X)
X.info()

# split the dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

### Load:
Preprocessed data loaded into memory for training.

## Data Modeling
### Model Architecture
For this competition, an ensemble approach combining LightGBM (LGBM) and CatBoost regressors was employed. Both models are robust and well-suited for tabular data with numerical and categorical features, achieving high accuracy while handling missing values and categorical features effectively.

### Features
The key features engineered and used for the models included:

### Numerical Features:

Distances (e.g., calculated travel distances between points)

Normalized Count of vehicles

Average Speed at a particular time of the day

### Categorical Features:

Identified categorical variables, converted to categorical types for compatibility with both LGBM and CatBoost.

## LightGBM (LGBM) Model
### Objective: 
Regressor optimized for Root Mean Squared Error (RMSE).
### Cross-validation: 
14-fold cross-validation was used to ensure model robustness and minimize overfitting.
### Hyperparameters:
<strong>metric</strong>: RMSE

<strong>verbose</strong>: -100 (silent mode)

<strong>Number of boosting rounds</strong>: 1000

### Validation Strategy: 
Each fold's out-of-fold predictions were evaluated using RMSE, and the mean RMSE across folds was calculated.
## CatBoost Model
### Objective: 
Regressor optimized for Root Mean Squared Error (RMSE) with automatic handling of categorical features.
### Cross-validation: 
14-fold cross-validation identical to the LGBM model.
### Hyperparameters:
<strong>iterations</strong>: 1000

<strong>Loss function</strong>: RMSE

<strong>Early stopping rounds</strong>: 17

<strong>Random seed</strong>: 48 (to ensure reproducibility)

### Validation Strategy: 
CatBoost's out-of-fold predictions were evaluated using RMSE, and the mean RMSE across folds was computed.

## Ensemble Method
The predictions from the LGBM and CatBoost models were ensembled to improve overall prediction accuracy. Two ensemble strategies were applied:

<strong>Mean Ensemble:</strong> Averaging predictions from all LGBM and CatBoost models.

<strong>Median Ensemble:</strong> Calculating the median of predictions from all models.

The mean ensemble was selected for the final submission as it provided better performance.

## Evaluation Metric
### Internal Validation Metric:
The internal metric used to validate model performance was Root Mean Squared Error (RMSE), calculated on the validation sets during cross-validation.
### Leaderboard Validation:
The public leaderboard score, based on RMSE, validated the performance of the submission on Zindi's test set.
### Model Performance
### Internal Validation RMSE:
<strong>LGBM:</strong> Computed across 14 folds, with a mean RMSE of <code>1.5841566980352786</code>.

<strong>CatBoost:</strong> Computed across 14 folds, with a mean RMSE of <code>1.5989421099589922</code>.

<strong>Public Leaderboard Score:</strong>  <code>1.810764462</code>

This ensemble approach demonstrated strong performance on the test set, ensuring reproducibility and robustness in predictions.

## 7. Inference Pipeline:
The inference pipeline for this project follows these structured steps to ensure consistency with the training pipeline and maximize prediction accuracy:

### 1. Data Preparation
Dataset: The test dataset (Test.csv) is loaded from the specified data path. A copy of the test data (test_df) is created to preserve the original data.

### Feature Engineering:
The same preprocessing steps applied to the training dataset are applied to the test data to ensure compatibility with the trained models:

<strong>Graph-Based Feature Lookup</strong>: The function look_up_from_graph() is used to derive relevant features from a graph-based dataset (graph_df), guided by preliminary_columns.

<strong>Feature Alignment</strong>: The test dataset columns are aligned to match the training set's feature columns (X_train.columns.tolist()).

<strong>Categorical Conversion</strong>: The function change_object_to_cat() ensures that categorical features are appropriately converted to the correct data types, as required by the models.

## 2. Generating Predictions with Individual Models
### LightGBM Predictions:
Each fold-specific LightGBM model in lgbm_models generates predictions for the test data.

If the model has best_iteration defined (from early stopping), predictions are made using the optimal number of iterations for maximum accuracy.

Otherwise, predictions use the full trained model.

Predictions are stored as columns in the pred_df DataFrame (pred_lgbm_0, pred_lgbm_1, etc.).

### CatBoost Predictions:
Similarly, each fold-specific CatBoost model in cat_models generates predictions for the test data.

Predictions are stored in separate columns of pred_df (pred_catboost_0, pred_catboost_1, etc.).

This step ensures that predictions from each model (and each fold) are retained for ensemble processing.

## 3. Ensemble Predictions
To leverage the strengths of both LightGBM and CatBoost models, their predictions are combined into a single, more robust prediction using ensemble methods:

### Combining Predictions:

Predictions from LightGBM models are extracted from columns containing pred_lgbm.

Predictions from CatBoost models are extracted from columns containing pred_catboost.

These predictions are concatenated into a single DataFrame for ensemble computation.

### Calculating Ensemble Outputs:

<strong>Mean Prediction</strong>: For each test instance, the mean of all predictions across both models is computed, stored in the mean_pred column of sub_df.

<strong>Median Prediction</strong>: For each test instance, the median of all predictions is computed, stored in the median_pred column of sub_df.

The mean prediction (mean_pred) was chosen as the final prediction for submission, as it often smooths out extreme values and improves generalization.

## 4. Submission File Creation
A new DataFrame (sub_file) is created with:

<strong>ID:</strong> The unique identifier from the test dataset (test.ID).

<strong>Target:</strong> The ensemble prediction (mean_pred).

The submission file is saved in .csv format (lgbm_catboost_ensemble_mean_submission14.csv), which adheres to the competition's submission format.

### Inference Pipeline
```bash
# Set the data path for test data
DATA_PATH = '/kaggle/input/yango-accra-mobility-dataset'
test = pd.read_csv(os.path.join(DATA_PATH, 'Test.csv'))
test_df = test.copy()

# Preprocess the dataset to match train set (custom functions assumed to exist)
test_df = look_up_from_graph(test_df, graph_df, preliminary_columns)
test_df = test_df[X_train.columns.tolist()]
test_df, _ = change_object_to_cat(test_df)
pred_df = test_df.copy()

# --- Prediction on the Test Set ---
# Predict using LGBM models
for i, model in enumerate(lgbm_models):
    if hasattr(model, 'best_iteration') and model.best_iteration:
        pred_df["pred_lgbm_{}".format(i)] = model.predict(test_df, num_iteration=model.best_iteration)
    else:
        pred_df["pred_lgbm_{}".format(i)] = model.predict(test_df)

# Predict using CatBoost models
for i, model in enumerate(cat_models):
    pred_df["pred_catboost_{}".format(i)] = model.predict(test_df)

# --- Ensemble the Predictions (Mean and Median) ---
# Calculate mean and median of the predictions from the models
sub_df = pred_df.copy()
lgbm_preds = pred_df.filter(like='pred_lgbm').values
catboost_preds = pred_df.filter(like='pred_catboost').values

# Combine LGBM and CatBoost predictions
combined_preds = pd.concat([pd.DataFrame(lgbm_preds), pd.DataFrame(catboost_preds)], axis=1)

# Calculate the mean and median
sub_df["mean_pred"] = combined_preds.mean(axis=1)
sub_df["median_pred"] = combined_preds.median(axis=1)

# --- Create a Submission File ---
sub_file = pd.DataFrame({'ID': test.ID, 'target': sub_df.mean_pred})  # Choose mean or median here
sub_file.to_csv('lgbm_catboost_ensemble_mean_submission14.csv', index=False)
sub_file.head()
```

<strong>Output Format:</strong>
```bash
Columns: image_id, average_speed.
```


## 5. Key Advantages of the Inference Pipeline
<strong>Consistency with Training</strong>: Ensures that test data undergoes identical preprocessing and feature engineering as the training data.

<strong>Model Diversity</strong>: By using predictions from both LightGBM and CatBoost models, the ensemble approach captures complementary strengths of both algorithms.

<strong>Robustness</strong>: Cross-validation ensures that predictions are less sensitive to data splits, and ensembling further reduces variance in predictions.
This structured pipeline ensures accurate, reproducible, and high-quality predictions for the competition.


## 8. Error Handling and Logging
All code includes appropriate exception handling to manage missing values or invalid inputs.
Debug messages and warnings are included where necessary.


## 9. How to Run
To use this project for prediction, follow these steps:

#### 1. Install Dependencies
Ensure the required libraries are installed. You can install them using pip:

```bash
pip install -r requirements.txt
```
For Colab users, ensure the necessary dependencies are added to the environment.

#### 2. Set Up the Dataset
Place the required dataset files in the specified directory (/kaggle/input/yango-accra-mobility-dataset or any directory of your choice).
In my case I used:
```bash
DATA_PATH = '/kaggle/input/yango-accra-mobility-dataset' # You can change it to your directory of choice as long as the dataset files from the competition can be found there
```

Train dataset: Train.csv

Test dataset: Test.csv

Sample Submission: SampleSubmission.csv

Graph dataset: Graph.csv

#### 3. Run all cells and save submission csv

## 10. Runtime
Runtime session is relatively 8m.

# Thank You!
