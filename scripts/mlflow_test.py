import mlflow
import random

# Step 2: set up the Mlflow tracking server
mlflow.set_tracking_uri("http://ec2-100-30-183-132.compute-1.amazonaws.com:5000/")

# 🔥 Set or create a new experiment
mlflow.set_experiment("Yt-comment-sentiment")

with mlflow.start_run():
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_param("param2", random.random())

    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(0.5, 1.5))

    print("Logged random parameters and metrics.")
