from sklearn.naive_bayes import MultinomialNB
import pickle

# Train the model
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
def save_artifacts(model, vectorizer):
    try:
        with open('spam_classifier.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        with open('vectorizer.pkl', 'wb') as vec_file:
            pickle.dump(vectorizer, vec_file)
    except Exception as e:
        print(f"Error saving artifacts: {e}")
