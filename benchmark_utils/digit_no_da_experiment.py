""" File to test the models on the digit dataset
- preprocessed data are available in data/digit.pkl file
- a model is trained on the preprocessed data
- Hyperparameters are tuned using GridSearchCV
- Model is evaluated on the test data
"""

import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

N_JOBS = -1
DATASET_SOURCE = 'usps'
DATASET_TARGET = 'mnist'

# Load the preprocessed data
with open('data/digit_preprocessed.pkl', 'rb') as f:
    data = pickle.load(f)
X, y = data[DATASET_SOURCE]['X'], data[DATASET_SOURCE]['y']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_train = X_train[::10]
y_train = y_train[::10]
print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create a pipeline
pipe = make_pipeline(SVC(kernel='rbf', C=100, gamma=0.01))

# Perform GridSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1]
}
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=N_JOBS)
grid.fit(X_train, y_train)
print(f"Best parameters: {grid.best_params_}")
print(f"Training accuracy: {grid.score(X_train, y_train)*100:.2f}%")
print(f"Test accuracy: {grid.score(X_test, y_test)*100:.2f}%")

# Predict the target data
X_target, y_target = data[DATASET_TARGET]['X'], data[DATASET_TARGET]['y']
print(f"Target data shape: {X_target.shape}")
print(f"Target accuracy: {grid.score(X_target, y_target)*100:.2f}%")
