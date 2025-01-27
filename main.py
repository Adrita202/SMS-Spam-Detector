from data_processing import load_dataset, preprocess_data, vectorize_text
from model_training import train_model, evaluate_model, save_artifacts
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = load_dataset('spam.csv')
if data is None:
    exit()

data = preprocess_data(data)

# Split the data
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the data
vectorizer, X_train_vec, X_test_vec = vectorize_text(X_train, X_test)

# Train the model
model = train_model(X_train_vec, y_train)

# Evaluate the model
evaluate_model(model, X_test_vec, y_test)

# Save the artifacts
save_artifacts(model, vectorizer)
