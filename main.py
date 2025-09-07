import logging

from data_processing import load_dataset, load_and_merge_datasets, preprocess_data, vectorize_text
from model_training import train_model, evaluate_model, save_artifacts
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Load and preprocess data
data = load_and_merge_datasets(['data/spam.csv', 'data/Dataset_5971.csv'])
if data is None:
    logging.error("Training aborted: Failed to load datasets.")
    exit(1)

data = preprocess_data(data)
if data is None or data.empty:
    logging.error("Training aborted: No usable data after preprocessing.")
    exit(1)

# Split the data
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the data
vectorizer, X_train_vec, X_test_vec = vectorize_text(X_train, X_test)
if vectorizer is None or X_train_vec is None or X_test_vec is None:
    logging.error("Training aborted: Vectorization failed.")
    exit(1)

# Train the model
model = train_model(X_train_vec, y_train)

# Evaluate the model
evaluate_model(model, X_test_vec, y_test)

# Save the artifacts
save_artifacts(model, vectorizer)
