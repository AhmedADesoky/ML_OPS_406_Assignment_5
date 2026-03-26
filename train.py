import os

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI environment variable is not set")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("assignment_5_pipeline")

    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )

    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=300, random_state=42)
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        measured_accuracy = accuracy_score(y_test, preds)

        forced_accuracy = os.getenv("SIMULATED_ACCURACY")
        accuracy = float(forced_accuracy) if forced_accuracy else float(measured_accuracy)

        mlflow.log_param("model", "logistic_regression")
        mlflow.log_metric("accuracy", accuracy)

        run_id = run.info.run_id
        with open("model_info.txt", "w", encoding="utf-8") as f:
            f.write(run_id)

        print(f"Run ID: {run_id}")
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
