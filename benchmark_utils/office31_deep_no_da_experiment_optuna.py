import sys
import os
from pathlib import Path

path_root = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(str(path_root))

import itertools
import torch
import optuna
from sklearn.metrics import accuracy_score
from skorch.callbacks import LRScheduler
from datasets.office31 import Dataset
from solvers.deep_no_da_source_only import Solver
from objective import Objective
import argparse

cache_dir = Path('__cache__')
cache_dir.mkdir(exist_ok=True)

# Load the dataset once
source, target = 'webcam', 'amazon'
print(f'Loading data for {source} -> {target}')
dataset = Dataset()
dataset.source_target = (source, target)

# Load data once
data = dataset.get_data()

# Set the data in the objective once
objective = Objective()
objective.set_data(**data)

# Get data
X_train = objective.X[objective.sample_domain > 0]
y_train = objective.y[objective.sample_domain > 0]
sample_domain_train = objective.sample_domain[objective.sample_domain > 0]
X_test = objective.X[objective.sample_domain < 0]
y_test = objective.y[objective.sample_domain < 0]
sample_domain_test = objective.sample_domain[objective.sample_domain < 0]

def objective_optuna(trial):
    # Get the model from the Solver
    n_classes = len(set(objective.y))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    dataset_name = dataset.name

    # Hyperparameter search space
    optimizer = torch.optim.SGD
    lr = trial.suggest_float('lr', 1e-4, 1, log=True)
    max_epochs = trial.suggest_int('max_epochs', 10, 50)
    scheduler_step = trial.suggest_int('scheduler_step', 1, max_epochs)
    scheduler_gamma = trial.suggest_float('scheduler_gamma', 0.1, 0.99)
    optimizer_momentum = trial.suggest_float('optimizer_momentum', 0, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    # Define the estimator
    solver = Solver()
    estimator = solver.get_estimator(n_classes=n_classes, device=device, dataset_name=dataset_name)
    hp = {
        'max_epochs': max_epochs,
        'lr': lr,
        'optimizer': optimizer,
        'optimizer__momentum': optimizer_momentum,
        'optimizer__weight_decay': weight_decay,
        'callbacks': [LRScheduler(policy='StepLR', step_size=scheduler_step, gamma=scheduler_gamma)]
    }
    estimator = estimator.set_params(**hp)

    # Train the model on the training data
    estimator.fit(X_train, y_train, sample_domain=sample_domain_train)

    # Get the test target set
    X_test_target = X_test[sample_domain_test < 0]
    y_test_target = y_test[sample_domain_test < 0]
    sample_domain_test_target = sample_domain_test[sample_domain_test < 0]

    # Predict on the test target set
    y_pred_test_target = estimator.predict(X_test_target, sample_domain=sample_domain_test_target)

    # Compute the accuracy on the test set
    accuracy = accuracy_score(y_test_target, y_pred_test_target)
    print(f"Test Target Accuracy: {accuracy:.4f}")

    # Return the accuracy for this trial
    return accuracy

if __name__ == "__main__":
    # Set up Optuna's RDBStorage (SQLite in this case)
    storage_name = "sqlite:///optuna_study.db"
    study_name = "office31_study"

    # Create a study with RDBStorage, so all Slurm jobs can share this study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True  # If the study exists, load it; this is critical for synchronization
    )

    # Start the optimization process, each worker runs one trial at a time
    study.optimize(objective_optuna, n_trials=40)

    # Output the best hyperparameters after the study is finished
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
