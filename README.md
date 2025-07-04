# SMS Spam Detector

This project is an SMS Spam Detector that uses a machine learning model (Multinomial Naive Bayes) to classify SMS messages as "Spam" or "Not Spam". It provides a web interface (built with Flask) where users can input a message and receive a prediction.

---

## Workflow

### 1. Data Processing (`data_processing.py`)
- Loads the dataset (`spam.csv`), keeping only the relevant columns.
- Preprocesses the data by mapping labels (`ham` to 0, `spam` to 1).
- Vectorizes the text messages using TF-IDF.

### 2. Model Training (`model_training.py` & `main.py`)
- Splits the data into training and test sets.
- Trains a Multinomial Naive Bayes classifier on the vectorized messages.
- Evaluates the model and prints accuracy, classification report, and confusion matrix.
- Saves the trained model and vectorizer as `spam_classifier.pkl` and `vectorizer.pkl`.

### 3. Web Application (`app.py` & `templates/index.html`)
- Loads the trained model and vectorizer.
- Provides a web form for users to input an SMS message.
- On submission, vectorizes the input and predicts whether it is spam.
- Displays the result ("Spam" or "Not Spam") on the same page.

---

## How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model and save artifacts:**
   ```bash
   python main.py
   ```
   This will generate `spam_classifier.pkl` and `vectorizer.pkl`.

3. **Run the web app:**
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000/` in your browser.

4. **Use the web interface:**
   - Enter an SMS message.
   - Click "Detect" to see if it's spam or not.

---

## File Descriptions

- `app.py`: Flask web server for prediction.
- `main.py`: Orchestrates data loading, training, evaluation, and saving.
- `data_processing.py`: Functions for loading, preprocessing, and vectorizing data.
- `model_training.py`: Functions for training, evaluating, and saving the model.
- `spam.csv`: Dataset of SMS messages.
- `spam_classifier.pkl`, `vectorizer.pkl`: Saved model and vectorizer.
- `templates/index.html`: Web interface template.

## Screenshot
![image](https://github.com/user-attachments/assets/b609827f-605e-47bd-a518-86c162174a57)
