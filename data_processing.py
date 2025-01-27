import pandas as pd

# Load the dataset
def load_dataset(file_path):
    try:
        data = pd.read_csv('spam.csv', encoding='latin-1')
        data = data[['class', 'message']]
        data.columns = ['label', 'message']
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Data preprocessing
def preprocess_data(data):
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

# Vectorize the text data
def vectorize_text(train_data, test_data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    train_vec = vectorizer.fit_transform(train_data)
    test_vec = vectorizer.transform(test_data)
    return vectorizer, train_vec, test_vec
