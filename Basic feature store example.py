# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC %pip install databricks-feature-store
# MAGIC
# MAGIC import pandas as pd
# MAGIC
# MAGIC from pyspark.sql.functions import monotonically_increasing_id, expr, rand
# MAGIC import uuid
# MAGIC
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
# MAGIC
# MAGIC import mlflow
# MAGIC import mlflow.sklearn
# MAGIC
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

# MAGIC %md
# MAGIC **Load dataset**
# MAGIC
# MAGIC The code in the following cell loads the dataset and does some minor data preparation: creates a unique ID for each observation and removes spaces from the column names. The unique ID column (wine_id) is the primary key of the feature table and is used to lookup features.

# COMMAND ----------

raw_data = spark.read.load("/databricks-datasets/wine-quality/winequality-red.csv",format="csv",sep=";",inferSchema="true",header="true" )

def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

def renameColumns(df):
    """Rename columns to be compatible with Feature Engineering in UC"""
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df

# Run functions
renamed_df = renameColumns(raw_data)
df = addIdColumn(renamed_df, 'wine_id')

# Drop target column ('quality') as it is not included in the feature table
features_df = df.drop('quality')
display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Reuse an existing catalog**
# MAGIC
# MAGIC To create a new catalog, you must have the CREATE CATALOG privilege on the metastore. To use an existing catalog, you must have the USE CATALOG privilege on the catalog

# COMMAND ----------

spark.sql("USE CATALOG ml")

# COMMAND ----------

# MAGIC %md
# MAGIC **Create a new schema in the catalog**
# MAGIC
# MAGIC To create a new schema in the catalog, you must have the CREATE SCHEMA privilege on the catalog.

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS wine_db")
spark.sql("USE SCHEMA wine_db")

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"ml.wine_db.wine_db_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Create the feature table

# COMMAND ----------

fe = FeatureEngineeringClient()

# You can get help in the notebook for feature engineering client API functions:
# help(fe.<function_name>)

# For example:
# help(fe.create_table)

# COMMAND ----------

fe.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=features_df,
    schema=features_df.schema,
    description="wine features"
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Train a model with Feature Engineering in Unity Catalog**
# MAGIC
# MAGIC The feature table does not include the prediction target. However, the training dataset needs the prediction target values. There may also be features that are not available until the time the model is used for inference.
# MAGIC This example uses the feature real_time_measurement to represent a characteristic of the wine that can only be observed at inference time. This feature is used in training and the feature value for a wine is provided at inference time.

# COMMAND ----------

## inference_data_df includes wine_id (primary key), quality (prediction target), and a real time feature
inference_data_df = df.select("wine_id", "quality", (10 * rand()).alias("real_time_measurement"))
display(inference_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Use a FeatureLookup to build a training dataset that uses the specified lookup_key to lookup features from the feature table and the online feature real_time_measurement. If you do not specify the feature_names parameter, all features except the primary key are returned.

# COMMAND ----------

def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fe.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fe.create_training_set(df=inference_data_df, feature_lookups=model_feature_lookups, label="quality", exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

# Create the train and test datasets
X_train, X_test, y_train, y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_name = "ml.wine_db.wine_model"

client = MlflowClient()

try:
    client.delete_registered_model(model_name) # Delete the model if already created
except:
    None

# COMMAND ----------

# MAGIC %md
# MAGIC The code in the next cell trains a scikit-learn RandomForestRegressor model and logs the model with the Feature Engineering in UC.
# MAGIC The code starts an MLflow experiment to track training parameters and results. Note that model autologging is disabled (mlflow.sklearn.autolog(log_models=False)); this is because the model is logged using fe.log_model.

# COMMAND ----------

# Disable MLflow autologging and instead log the model using Feature Engineering in UC
mlflow.sklearn.autolog(log_models=False)

def train_model(X_train, X_test, y_train, y_test, training_set, fe):
    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fe.log_model(
            model=rf,
            artifact_path="wine_quality_prediction",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
        )

train_model(X_train, X_test, y_train, y_test, training_set, fe)

# COMMAND ----------

# MAGIC %md
# MAGIC LOGGED MODEL CAN BE SEE AT EXPERIMENTS ON THE SIDE BAR

# COMMAND ----------

# MAGIC %md
# MAGIC **Batch scoring**
# MAGIC
# MAGIC Use score_batch to apply a packaged Feature Engineering in UC model to new data for inference. The input data only needs the primary key column wine_id and the realtime feature real_time_measurement. The model automatically looks up all of the other feature values from the feature tables.

# COMMAND ----------

# Helper function
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

## For simplicity, this example uses inference_data_df as input data for prediction
batch_input_df = inference_data_df.drop("quality") # Drop the label column

latest_model_version = get_latest_model_version(model_name)

predictions_df = fe.score_batch(model_uri=f"models:/{model_name}/{latest_model_version}", df=batch_input_df)

display(predictions_df["wine_id", "prediction"])
