# Databricks notebook source
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

spark.sql("USE CATALOG ml")

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS wine_db")
spark.sql("USE SCHEMA wine_db")

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"ml.wine_db.wine_db_" + str(uuid.uuid4())[:6]
print(table_name)

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
