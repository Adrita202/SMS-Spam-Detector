from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
def load_artifacts():
    try:
        with open('spam_classifier.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vec_file:
            loaded_vectorizer = pickle.load(vec_file)
        return loaded_model, loaded_vectorizer
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        exit()

loaded_model, loaded_vectorizer = load_artifacts()

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message')  # Get message from the form
    if not message or not isinstance(message, str):
        return jsonify({'error': 'Invalid input, please provide a valid message.'}), 400
    
    vectorized_message = loaded_vectorizer.transform([message])
    prediction = loaded_model.predict(vectorized_message)[0]
    
    # Return prediction to the HTML template
    result = "Spam" if prediction else "Not Spam"
    return render_template('index.html', message=message, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
