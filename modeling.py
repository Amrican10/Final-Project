from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression,
    MultilayerPerceptronClassifier,
    RandomForestClassifier,
)
from pyspark.ml.feature import StringIndexer, VectorAssembler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import time

# Initialize SparkSession
spark = SparkSession.builder.appName("ParallelMLExperiments").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load data into Pandas
df = pd.read_csv("./data/final_train.csv", header=None)

# Separate features and target
target_col = 0
X = df.drop(columns=[target_col])
y = df[target_col]
df[target_col] = y
spark_df = spark.createDataFrame(df)

# Preprocess Data: Vectorize features for PySpark
feature_columns = [str(i) for i in X.columns]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
spark_df = assembler.transform(spark_df)

# Index target column
indexer = StringIndexer(inputCol=str(target_col), outputCol="label")
spark_df = indexer.fit(spark_df).transform(spark_df)

# Split into train and test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# Helper function to evaluate metrics
def evaluate_model(predictions):
    pred_and_labels = predictions.select("prediction", "label").toPandas()
    y_true = pred_and_labels["label"]
    y_pred = pred_and_labels["prediction"]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    return accuracy, precision, recall

# Function to train models and record metrics
def train_model(model, train_df, test_df, partition_counts, results_dict):
    for num_partitions in partition_counts:
        print(f"Training with {num_partitions} partitions...")
        train_df_repartitioned = train_df.repartition(num_partitions)
        start_time = time.time()
        parallel_model = model.fit(train_df_repartitioned)
        training_time = time.time() - start_time
        predictions = parallel_model.transform(test_df)
        accuracy, precision, recall = evaluate_model(predictions)

        results_dict["partitions"].append(num_partitions)
        results_dict["accuracy"].append(accuracy)
        results_dict["precision"].append(precision)
        results_dict["recall"].append(recall)
        results_dict["training_time"].append(training_time)

# Plotting function with best/worst annotations
def plot_metrics_with_annotations(results_df, title_prefix, save_prefix):
    metrics = ["accuracy", "precision", "recall", "training_time"]
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        plt.plot(results_df["partitions"], results_df[metric], marker="o", label=metric.capitalize())
        best_idx = results_df[metric].idxmin() if metric == "training_time" else results_df[metric].idxmax()
        worst_idx = results_df[metric].idxmax() if metric == "training_time" else results_df[metric].idxmin()
        best_partition = results_df.loc[best_idx, "partitions"]
        worst_partition = results_df.loc[worst_idx, "partitions"]
        best_value = results_df.loc[best_idx, metric]
        worst_value = results_df.loc[worst_idx, metric]

        plt.text(best_partition, best_value, f"Best: t={best_partition} ({best_value:.2f})", fontsize=9, color="green")
        plt.text(worst_partition, worst_value, f"Worst: t={worst_partition} ({worst_value:.2f})", fontsize=9, color="red")

        plt.xlabel("Number of Partitions")
        plt.ylabel(metric.capitalize())
        plt.title(f"{title_prefix} {metric.capitalize()} vs Partitions")
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_prefix}_{metric}_vs_partitions.png")
        plt.show()

# Training Models
partition_counts = [1, 2, 4, 8, 16, 32]
results_dicts = {
    "LR": {"partitions": [], "accuracy": [], "precision": [], "recall": [], "training_time": []},
    "MLP": {"partitions": [], "accuracy": [], "precision": [], "recall": [], "training_time": []},
    "RF": {"partitions": [], "accuracy": [], "precision": [], "recall": [], "training_time": []},
}

models = {
    "LR": LogisticRegression(maxIter=10, regParam=0.01),
    "MLP": MultilayerPerceptronClassifier(maxIter=100, layers=[len(feature_columns), 5, 4, 2]),
    "RF": RandomForestClassifier(numTrees=50, maxDepth=10, featuresCol="features", labelCol="label"),
}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    train_model(model, train_df, test_df, partition_counts, results_dicts[model_name])

# Plot results for each model
for model_name, results in results_dicts.items():
    results_df = pd.DataFrame(results)
    plot_metrics_with_annotations(results_df, model_name, model_name)

# Combine results for histograms
time_metrics = {"Model": [], "Best Time": [], "Worst Time": [], "Optimal Improvement (%)": []}

for model_name, results in results_dicts.items():
    results_df = pd.DataFrame(results)
    best_time = results_df["training_time"].min()
    worst_time = results_df["training_time"].max()
    improvement_percentage = (worst_time - best_time) / worst_time * 100

    time_metrics["Model"].append(model_name)
    time_metrics["Best Time"].append(best_time)
    time_metrics["Worst Time"].append(worst_time)
    time_metrics["Optimal Improvement (%)"].append(improvement_percentage)

# Plot training time metrics histogram
time_metrics_df = pd.DataFrame(time_metrics)
time_metrics_df.set_index("Model")[["Optimal Improvement (%)"]].plot(
    kind="bar", figsize=(10, 6), title="Optimal Improvement as Percentage of Worst Case Scenario"
)
plt.ylabel("Improvement (%)")
plt.xlabel("Model")
plt.grid()
plt.savefig("Optimal_Improvement_Histogram.png")
plt.show()

print("Training complete. All plots saved.")
