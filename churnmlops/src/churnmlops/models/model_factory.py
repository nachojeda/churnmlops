from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class ModelFactory:
    @staticmethod
    def create_model(model_type, model_params):
        if model_type == 'logistic_regression':
            return LogisticRegression(**model_params)
        elif model_type == 'random_forest':
            return RandomForestClassifier(**model_params)
        elif model_type == 'xgboost':
            return XGBClassifier(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
