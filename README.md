# 📱 SMS Spam Detector
<!-- 
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.1-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg) -->

A SMS spam detection system with a beautiful web interface. Built with machine learning (TF-IDF + Naive Bayes) and deployed with Docker.

## ✨ Features

- 🤖 **Smart Detection** - TF-IDF vectorization with Multinomial Naive Bayes
- 🎨 **Modern UI** - Glassmorphism design with confidence scores
- 📊 **Prediction History** - Track recent predictions with confidence levels
- 🐳 **Docker Ready** - Multi-stage build with Gunicorn for production
- 🔒 **Secure** - Security headers, non-root container, input validation
- 📱 **Responsive** - Works on desktop, tablet, and mobile

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/yourusername/sms-spam-detector.git
cd sms-spam-detector
docker compose up --build -d
```
Visit: http://localhost:8080

### Option 2: Local Development
```bash
git clone https://github.com/yourusername/sms-spam-detector.git
cd sms-spam-detector

# Create virtual environment
py -3 -m venv .venv
./.venv/Scripts/Activate.ps1  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install dependencies and train model
pip install -r requirements.txt
python main.py

# Start the app
python app.py
```
Visit: http://127.0.0.1:5000

![Screenshot](<Images/Screenshot (189).png>)

## 🎯 How It Works

1. **Training**: Combines multiple datasets, preprocesses text, trains ML model
2. **Prediction**: Vectorizes input text, classifies as spam/ham with confidence
3. **UI**: Shows results with confidence scores and maintains prediction history

## 📊 Model Performance

- **Algorithm**: TF-IDF + Multinomial Naive Bayes
- **Features**: English stop-words removed, max_df=0.7
- **Datasets**: UCI SMS Spam + SMS Phishing (5,971 messages)
- **Accuracy**: ~97%+ on test data

## 🛠️ Tech Stack

- **Backend**: Python, Flask, scikit-learn, pandas
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
- **Deployment**: Docker, Gunicorn, multi-stage builds
- **Security**: CSP headers, input validation, non-root containers

## 📁 Project Structure

```
├── app.py                 # Flask web application
├── main.py               # Model training orchestrator
├── data_processing.py    # Data loading and preprocessing
├── model_training.py     # ML model training and evaluation
├── templates/            # HTML templates
├── static/              # CSS and JavaScript
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Production deployment
└── requirements.txt     # Python dependencies
```

## 🔧 Configuration

Key environment variables:
- `SECRET_KEY`: Session encryption (required in production)
- `LOG_LEVEL`: Logging verbosity (debug, info, warning, error)
- `GUNICORN_WORKERS`: Number of worker processes
- `ENABLE_HSTS`: Enable HTTPS security headers

## 🐳 Docker Features

- **Multi-stage build**: Separate training and runtime stages
- **Security**: Non-root user, minimal attack surface
- **Health checks**: Automatic container health monitoring
- **Development mode**: Hot reloading with volume mounts



## 🙏 Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- Mendeley Data for the SMS Phishing dataset
- Flask and scikit-learn communities for excellent documentation

---

⭐ **Star this repo** if you found it helpful! | 