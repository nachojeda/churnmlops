import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import mlflow
from mlflow.models import infer_signature
from ..utils.preprocessor import DataPreprocessor

class Trainer:
    def __init__(self, model, data_path, target_column, exclude_columns, test_size, random_state, mlflow_logger, logger):
        self.model = model
        self.data_path = data_path
        self.target_column = target_column
        self.exclude_columns = exclude_columns
        self.test_size = test_size
        self.random_state = random_state
        self.mlflow_logger = mlflow_logger
        self.preprocessor = DataPreprocessor()  # Initialize the preprocessor
        self.logger = logger

    def load_data(self):
        # Load data from CSV
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=[self.target_column])
        X = data.drop(columns=self.exclude_columns) # Rearrange this
        y = data[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train(self):
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()

        # Preprocess the data
        X_train = self.preprocessor.preprocess(X_train)
        X_test = self.preprocessor.transform(X_test)

        # Start MLFlow run
        with self.mlflow_logger.start_run():
            # Train the model
            self.model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

            # Calculate and log metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

            self.mlflow_logger.log_param('model_type', self.model.__class__.__name__)
            self.mlflow_logger.log_params(self.model.get_params())
            self.mlflow_logger.log_metric('accuracy', accuracy)
            if roc_auc is not None:
                self.mlflow_logger.log_metric('roc_auc', roc_auc)

            # Log the model in MLFlow
            # self.mlflow_logger.log_model(self.model)
            # Infer the model signature
            signature = infer_signature(X_train, y_pred)

            # Log the model
            self.mlflow_logger.log_model(
                sk_model=self.model,
                artifact_path="churn_model",
                # registered_model_name="pickle",
                # signature=signature,
                # input_example=X_train,
            )

            # Set a tag that we can use to remind ourselves what this run was for
            self.mlflow_logger.set_tag()

            self.logger.info(f"Training complete. Accuracy: {accuracy}, ROC-AUC: {roc_auc}")