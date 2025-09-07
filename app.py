import os
import logging
import secrets
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle

app = Flask(__name__)

# Basic logging to stdout (Docker-friendly) with robust level parsing
_raw_level = os.getenv("LOG_LEVEL", "INFO")
try:
    _level = int(_raw_level)
except (TypeError, ValueError):
    _level = getattr(logging, str(_raw_level).upper(), logging.INFO)
logging.basicConfig(level=_level)

# Secret key from environment for session security
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))

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
    # Read and clear any prior values stored for PRG (one-time display)
    prior_message = session.pop('message', None)
    prior_prediction = session.pop('prediction', None)
    prior_confidence = session.pop('confidence', None)
    prior_spam_prob = session.pop('spam_prob', None)
    prior_ham_prob = session.pop('ham_prob', None)

    # Keep history persistent (bounded list)
    history = session.get('history', [])

    return render_template(
        'index.html',
        message=prior_message,
        prediction=prior_prediction,
        confidence=prior_confidence,
        spam_prob=prior_spam_prob,
        ham_prob=prior_ham_prob,
        history=history,
    )


@app.route('/healthz')
def healthz():
    """Container health check endpoint.
    Returns 200 if app is alive and artifacts are loaded.
    """
    healthy = loaded_model is not None and loaded_vectorizer is not None
    status_code = 200 if healthy else 503
    return jsonify({
        'status': 'ok' if healthy else 'unhealthy',
        'model_loaded': bool(loaded_model),
        'vectorizer_loaded': bool(loaded_vectorizer),
    }), status_code

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message')  # Get message from the form
    if not message or not isinstance(message, str):
        return jsonify({'error': 'Invalid input, please provide a valid message.'}), 400
    
    vectorized_message = loaded_vectorizer.transform([message])
    prediction = loaded_model.predict(vectorized_message)[0]
    prediction_proba = None
    try:
        prediction_proba = loaded_model.predict_proba(vectorized_message)[0]
    except Exception:
        # Not all models support predict_proba; default to equal probabilities
        prediction_proba = [0.5, 0.5]

    # Calculate human-friendly outputs
    confidence_pct = float(max(prediction_proba) * 100.0)
    spam_probability_pct = float(prediction_proba[1] * 100.0)
    ham_probability_pct = float(prediction_proba[0] * 100.0)
    result = "Spam" if prediction else "Not Spam"

    # Store in session for one-time display, then redirect (PRG)
    session['message'] = message
    session['prediction'] = result
    session['confidence'] = round(confidence_pct, 1)
    session['spam_prob'] = round(spam_probability_pct, 1)
    session['ham_prob'] = round(ham_probability_pct, 1)

    # Append to a small rolling history for display
    history = session.get('history', [])
    history.append({
        'message': message,
        'prediction': result,
        'confidence': round(confidence_pct, 1),
    })
    # Keep only last 10
    session['history'] = history[-10:]
    return redirect(url_for('home'), code=303)


@app.after_request
def add_no_cache_headers(response):
    # Prevent browsers from caching pages which can cause stale content when navigating back
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    # Security headers
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    # Relax CSP to allow inline scripts used in template; tighten as needed
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
    )
    if os.getenv("ENABLE_HSTS", "0") == "1":
        response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
    return response


@app.route('/clear-history', methods=['POST'])
def clear_history():
    session['history'] = []
    return redirect(url_for('home'), code=303)

if __name__ == '__main__':
    debug_flag = os.getenv('FLASK_DEBUG', '1') in ('1', 'true', 'True')
    host = os.getenv('APP_HOST', '0.0.0.0')
    port = int(os.getenv('APP_PORT', '5000'))
    app.run(host=host, port=port, debug=debug_flag)
