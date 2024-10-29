import mlflow
import mlflow.sklearn

class MLFlowLogger:
    def __init__(self, experiment_name, run_name, tag, tag_des, tracking_uri=None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tag = tag
        self.tag_des = tag_des
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self):
        # Start a new MLFlow run
        return mlflow.start_run(run_name=self.run_name)

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_params(self, params):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metric(self, key, value):
        mlflow.log_metric(key, value)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_model(self, sk_model, artifact_path):
        # Log the trained model
        mlflow.sklearn.log_model(sk_model, artifact_path)

    def set_tag(self):
        mlflow.set_tag(key=self.tag_des, value=self.tag)

