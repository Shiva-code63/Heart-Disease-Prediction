# ❤️ Heart Disease Risk Predictor Using Machine Learning

An interactive web application that predicts the risk of heart disease using Machine Learning and helps users discover nearby healthcare professionals.

## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Early identification of potential risk factors can encourage individuals to seek timely medical advice.

This project uses Machine Learning algorithms to estimate heart disease risk based on patient health parameters. It also provides a nearby doctor recommendation feature to help users find relevant healthcare professionals.

The application is developed using Python and Streamlit, with an interactive user interface.

## ✨ Key Features

- **Heart Disease Risk Prediction:** Predicts the likelihood of heart disease based on user-provided health parameters.
- **Multiple ML Models:** Implements Logistic Regression, Decision Tree, and Random Forest.
- **Model Evaluation:** Compares models using accuracy, precision, recall, F1-score, and ROC-AUC.
- **Nearby Doctor Discovery:** Helps users find healthcare professionals within a 30 km radius.
- **Location-Based Search:** Supports browser location access and manual location input.
- **Doctor Recommendations:** Suggests cardiologists or general physicians based on the predicted risk category.
- **Doctor Ranking:** Considers ratings and geographical distance when ranking available doctors.
- **AI-Generated Lifestyle Suggestions:** Provides personalized lifestyle suggestions using the Groq API.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Database | SQLite, SQLAlchemy |
| Maps & Location | Folium, OpenStreetMap |
| Geographical Calculations | Haversine Formula |
| AI Integration | Groq API |

## 🤖 Machine Learning Models

The project implements and compares three classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models are evaluated using an 80:20 train-test split and 5-fold cross-validation.

### Dataset Information

- Total records: 918
- Input features: 11
- Target variable: `HeartDisease`
- Problem type: Binary Classification

### Model Performance

Random Forest was selected based on its cross-validation F1-score.

| Evaluation Metric | Random Forest |
|---|---:|
| Accuracy | 88.0% |
| Precision | 87.7% |
| Recall | 91.2% |
| F1-Score | 89.4% |
| ROC-AUC | 92.7% |

*Note: These are the evaluation results obtained during this project's reported experiment. Results may vary with changes to the dataset or experimental setup.*

## 🩺 Nearby Doctor Recommendation System

The project also includes a location-based healthcare discovery feature.

The workflow:

1. Gets the user's location through browser permission or manual address input.
2. Uses OpenStreetMap services to discover nearby healthcare facilities.
3. Calculates geographical distance using the Haversine formula.
4. Filters facilities within a 30 km radius.
5. Ranks available doctors using ratings and distance.
6. Displays relevant healthcare recommendations.

Doctor verification is supported through manual checks against the Indian Medical Register. OpenStreetMap listings themselves should not be considered verified medical registrations.

## 📂 Project Structure

```text
Heart-Disease-Risk-Predictor/
│
├── .streamlit/
│   └── config.toml
│
├── doctor_module/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── discovery.py
│   ├── geo.py
│   ├── ranking.py
│   └── ui.py
│
├── app.py
├── manage_doctors.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shiva-code63/Heart-Disease-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Heart-Disease-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API credentials

If using the Groq-powered AI suggestions, create a file named `secrets.toml` inside the `.streamlit` directory.

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Replace the placeholder with your actual Groq API key.

**Important:** Never upload `secrets.toml` or your API key to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌍 External Services

This project uses the following external services:

- **OpenStreetMap:** For geographical data and nearby healthcare facility discovery.
- **Nominatim:** For address and location search.
- **Overpass API:** For retrieving nearby healthcare facility information.
- **Groq API:** For AI-generated lifestyle suggestions.

Availability of these services depends on their respective APIs and usage policies.

## 🚀 Future Improvements

- Integration with more reliable and verified doctor databases.
- Improved model performance using larger and more diverse datasets.
- Additional machine learning algorithms.
- User authentication and patient history management.
- Improved doctor filtering and recommendation features.
- Deployment on a cloud hosting platform.

## ⚠️ Disclaimer

This application is developed for educational and academic purposes.

The machine learning model provides an estimated risk based on the input parameters. It is not a medical diagnosis and should not replace professional medical advice, diagnosis, or treatment.

Users should consult qualified healthcare professionals for medical concerns.

## 👨‍💻 Author

**Shiva-code63**

GitHub: [Shiva-code63](https://github.com/Shiva-code63)

---

⭐ If you find this project interesting, consider giving the repository a star!