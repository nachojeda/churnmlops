"""Scripts of the project"""

import sys
import yaml
import logging
from .models.model_factory import ModelFactory
from .trainers.trainer import Trainer
from .utils.mlflow_logger import MLFlowLogger

# Initialize logging
logging.basicConfig(
    level=logging.INFO,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Define log message format
    handlers=[
        logging.StreamHandler()  # Logs to the console; you can add FileHandler to log to a file as well
    ]
)
logger = logging.getLogger(__name__)  # Get a logger instance

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main(argv: list[str] | None = None) -> int:
    """Run the main script function."""
    args = argv or sys.argv[1:]
    logger.info(f"\nArgs: {args}\n")

    # Load the YAML configuration
    config = load_config(args[0])
    
    # Initialize the MLFlow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=config['mlflow']['experiment_name'], 
        run_name=config['mlflow']['run_name'],
        tracking_uri=config['mlflow']['tracking_uri'],
        tag = config['mlflow']['tag'],
        tag_des = config['mlflow']['tag_des']
    )
    logger.info(f"Experiment name: {config['mlflow']['experiment_name']}\n")
    logger.info(f"Run name: {config['mlflow']['run_name']}\n")
    logger.info(f"Tracking URI: {config['mlflow']['tracking_uri']}\n")
    logger.info(f"Tag: {config['mlflow']['tag']}\n")

    # Get the model
    model = ModelFactory.create_model(
        model_type=config['model']['type'], 
        model_params=config['model']['params']
    )
    logger.info(f"Model selected: {config['model']['type']}\n")
    logger.info(f"Hyperparameters selected: {config['model']['params']}\n")

    # Initialize the Trainer with the config and model
    trainer = Trainer(
        model=model, 
        data_path=config['data']['path'], 
        target_column=config['data']['target'],
        exclude_columns=config['data']['exclude'],
        test_size=config['train_test_split']['test_size'],
        random_state=config['train_test_split']['random_state'],
        mlflow_logger=mlflow_logger,
        logger=logger
    )
    
    # Execute training
    trainer.train()