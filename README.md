# 🪐 Exoplanet Habitability & Earth-Likeness Prediction

An end-to-end Machine Learning project that analyzes exoplanet and stellar parameters to estimate **planet habitability** and **Earth-likeness scores** using regression models.

The project includes data preprocessing, feature engineering, model training and evaluation, model serialization, and a Flask-based web application for interactive predictions.

> **Note:** The habitability and Earth-likeness targets in this project are generated using predefined scoring criteria based on selected planetary and stellar characteristics. They are not scientifically validated measurements of actual habitability.

---

## 📌 Project Overview

The goal of this project is to explore whether machine learning can learn relationships between measurable exoplanet characteristics and predefined habitability-related scoring criteria.

The project performs two prediction tasks:

1. **Planet Habitability Score**
2. **Earth-Likeness Score**

Two separate Random Forest regression models are trained for these objectives.

The trained models are then integrated into a Flask web application where users can enter planetary parameters and receive predicted scores.

---

## 🔄 Project Workflow

```text
Exoplanet Dataset
       ↓
Data Exploration & Cleaning
       ↓
Feature Selection
       ↓
Missing Value Handling
       ↓
Rule-Based Score Generation
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
Random Forest Regression
       ↓
Model Evaluation
       ↓
Model Serialization
       ↓
Flask Web Application
       ↓
User Input → Prediction → Results
```

---

## 📊 Dataset

The project uses an exoplanet dataset containing planetary and stellar properties.

The original dataset contains hundreds of columns. A smaller set of relevant features was selected for this project.

The selected parameters include:

- Orbital Period
- Orbital Semi-Major Axis
- Equilibrium Temperature
- Orbital Eccentricity
- Planet Radius
- Planet Mass
- Planet Density
- Stellar Effective Temperature
- Orbital Inclination
- Planet/Star Radius Ratio

The processed dataset used in the project contains **545 observations**.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Selected relevant planetary and stellar features
- Analyzed missing values
- Removed features with more than 50% missing values
- Removed observations with excessive missing values
- Filled remaining numerical missing values using column mean
- Separated features and prediction targets
- Split the data into training and testing sets
- Applied feature scaling using `StandardScaler`

---

## 🎯 Target Generation

Since the dataset does not contain directly validated labels for planetary habitability or Earth-likeness, custom scoring functions were created.

### Habitability Score

The habitability score considers factors such as:

- Planet mass
- Orbital eccentricity
- Orbital semi-major axis
- Stellar temperature
- Planet/star radius ratio
- Planet radius
- Planet density
- Equilibrium temperature

### Earth-Likeness Score

The Earth-likeness score considers factors such as:

- Orbital period
- Orbital semi-major axis
- Planet radius
- Planet mass
- Planet density
- Equilibrium temperature
- Orbital inclination
- Planet/star radius ratio
- Orbital eccentricity

These rules generate numerical scores that are subsequently used as regression targets.

---

## 🤖 Machine Learning Models

Two separate **Random Forest Regression** models were trained.

### 1. Habitability Model

Features:

```text
pl_orbsmax
pl_eqt
pl_orbeccen
pl_rade
pl_masse
pl_dens
st_teff
pl_ratror
```

### 2. Earth-Likeness Model

Features:

```text
pl_orbper
pl_orbsmax
pl_eqt
pl_orbeccen
pl_rade
pl_masse
pl_dens
pl_orbincl
pl_ratror
```

The Random Forest models were configured with:

```python
n_estimators = 100
random_state = 42
```

---

## 📈 Model Evaluation

The models were evaluated using multiple regression metrics.

| Model | R² | MAE | MSE |
|---|---:|---:|---:|
| Habitability | 0.8741 | 1.1662 | 10.2081 |
| Earth-Likeness | 0.9726 | 0.9944 | 5.7491 |

Additional metrics including Explained Variance and MAPE were also calculated during evaluation.

> These metrics measure how well the models reproduce the generated scoring targets. They should not be interpreted as scientific evidence that the models can determine whether a real exoplanet is habitable.

---

## 🌐 Flask Web Application

A Flask web application was created to provide an interactive interface for the trained models.

Users can enter:

- Orbital Period
- Orbital Semi-Major Axis
- Equilibrium Temperature
- Orbital Eccentricity
- Planet Radius
- Planet Mass
- Planet Density
- Stellar Temperature
- Orbital Inclination
- Radius Ratio

The application sends the inputs to the trained models and displays:

- Planet Habitability Score
- Earth-Likeness Score

---

## 🗂️ Project Structure

```text
exoplanet-habitability/
│
├── app.py
├── README.md
│
├── model/
│   ├── habitability_model.pkl
│   └── earth_likeness_model.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── images/
│       └── background.jpg
│
├── Exoplanet_notebook.ipynb
│
└── exoplanet_dataset.csv
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/exoplanet-habitability.git
cd exoplanet-habitability
```

Create a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install flask pandas numpy scikit-learn joblib
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

The application will run locally.

Open your browser and visit:

```text
http://127.0.0.1:5000
```

Enter the planetary parameters and click **Predict** to view the results.

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regression
- Joblib
- Flask
- HTML
- Bootstrap

---

## 💡 Key Learning Outcomes

Through this project, I gained hands-on experience with:

- Working with real-world scientific datasets
- Exploratory data analysis
- Missing-value analysis and preprocessing
- Feature selection
- Feature engineering
- Regression modeling
- Model evaluation
- Model serialization using Joblib
- Building a Flask-based ML application
- Connecting trained ML models with a web interface

---

## ⚠️ Limitations

This project is primarily an educational and experimental Machine Learning application.

The main limitations are:

- The dataset is relatively small after preprocessing.
- The prediction targets are generated from predefined scoring rules rather than experimentally validated habitability labels.
- The scoring criteria are simplified representations of selected planetary characteristics.
- The model evaluation therefore measures how well the ML models reproduce the generated scores, rather than measuring scientifically validated planetary habitability.
- Further validation with larger datasets and scientifically established labels would be required for a research-grade system.

---

## 🚀 Future Improvements

Potential improvements include:

- Use scientifically validated habitability-related datasets
- Increase the number of observations
- Apply cross-validation for more robust evaluation
- Compare Random Forest with XGBoost and Gradient Boosting
- Perform systematic hyperparameter tuning
- Improve feature selection
- Investigate more scientifically meaningful habitability indicators
- Package preprocessing and models into a unified inference pipeline
- Containerize and deploy the application

---

## 👨‍💻 Author

**Arun Arumugam**

AI/ML Engineer

Interested in Machine Learning, Deep Learning, Generative AI, and building practical AI applications.
