# 🏠 House Price Prediction

A machine learning web application that predicts house prices based on property-related features such as location, BHK, area, construction status, and resale information.

The project uses **Python, Pandas, Scikit-learn, Random Forest Regression, and Streamlit** to build and deploy an interactive house price prediction system.

---
<img width="902" height="214" alt="111" src="https://github.com/user-attachments/assets/33a34fe7-ca3f-4374-aa13-a849363aec59" />

## 📌 Project Overview

House prices can vary significantly depending on factors such as property size, location, number of bedrooms, construction status, and other characteristics.

This project aims to build a machine learning model that learns patterns from historical housing data and predicts the expected price of a property based on user-provided information.

The project follows an end-to-end machine learning workflow:

**Data → Data Analysis → Preprocessing → Feature Engineering → Model Training → Evaluation → Model Saving → Web Application**

---

## ✨ Features

* 📊 Exploratory Data Analysis (EDA)
* 🧹 Data preprocessing and cleaning
* ⚙️ Feature engineering
* 🤖 Machine learning using Random Forest Regression
* 📈 Model evaluation using MAE, MSE, RMSE, and R²
* 💾 Trained model serialization using Joblib
* 🌐 Interactive Streamlit web application
* 🎨 Clean and professional user interface
* 🔮 Real-time house price predictions

---

## 🛠️ Technologies Used

| Technology           | Purpose                               |
| -------------------- | ------------------------------------- |
| **Python**           | Core programming language             |
| **Pandas**           | Data manipulation and analysis        |
| **NumPy**            | Numerical computations                |
| **Matplotlib**       | Data visualization                    |
| **Seaborn**          | Statistical visualization             |
| **Scikit-learn**     | Machine learning                      |
| **Joblib**           | Model saving/loading                  |
| **Streamlit**        | Web application                       |
| **Jupyter Notebook** | Model development and experimentation |
| **Git & GitHub**     | Version control                       |

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── data/
│   └── train.csv
│
├── notebook.ipynb
├── model.py
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── house_price_model.pkl
```

> **Note:** The trained model file is excluded from GitHub when necessary because large binary files can exceed GitHub's file-size limits.

---

## 📊 Dataset

The dataset contains information about residential properties, including:

* `POSTED_BY`
* `UNDER_CONSTRUCTION`
* `RERA`
* `BHK_NO.`
* `BHK_OR_RK`
* `SQUARE_FT`
* `READY_TO_MOVE`
* `RESALE`
* `ADDRESS`
* `LONGITUDE`
* `LATITUDE`
* `TARGET(PRICE_IN_LACS)`

The target variable is:

```text
TARGET(PRICE_IN_LACS)
```

which represents the property price in lakhs.

---

## 🔄 Machine Learning Workflow

### 1. Data Loading

The housing dataset is loaded using Pandas and inspected for its structure, data types, missing values, and statistical information.

### 2. Exploratory Data Analysis

Different visualizations and statistical techniques are used to understand:

* Price distribution
* Property size
* BHK distribution
* Location-based patterns
* Correlations between features
* Relationships between property characteristics and price

### 3. Data Preprocessing

The dataset is prepared for machine learning by:

* Handling categorical variables
* Removing unnecessary features
* Converting data into numerical representations
* Preparing features and target variables

### 4. Feature Engineering

The `ADDRESS` column is removed because it contains a large number of unique categorical values and can introduce excessive dimensionality.

Categorical features are encoded so that they can be processed by the machine learning model.

### 5. Train-Test Split

The processed dataset is divided into training and testing sets.

```text
Training Data → 80%
Testing Data  → 20%
```

### 6. Model Training

A **Random Forest Regressor** is trained on the processed housing data.

Random Forest was selected because it can capture nonlinear relationships between property features and prices while being relatively robust to variations in the data.

### 7. Model Evaluation

The trained model is evaluated using:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

These metrics help measure how accurately the model predicts house prices.

### 8. Model Saving

The trained model is saved using Joblib:

```python
joblib.dump(model, "house_price_model.pkl")
```

The saved model can later be loaded by the Streamlit application without retraining it.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit interface where users can enter property information and receive a predicted house price.

The application allows users to provide information such as:

* Number of bedrooms
* Property area
* Construction status
* RERA availability
* Ready-to-move status
* Resale status
* Latitude
* Longitude
* Property type
* Posted by

The trained machine learning model then generates the estimated property price.

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/aizazahmad736/House-price-prediction.git
```

### 2. Navigate to the Project

```bash
cd House-price-prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```bash
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Model Performance

The model was evaluated using standard regression metrics including **MAE, MSE, RMSE, and R²**.

Model performance may vary depending on preprocessing, feature selection, and the train-test split used during experimentation.

---

## 🎯 Future Improvements

Potential improvements include:

* 🔹 Hyperparameter tuning
* 🔹 Cross-validation
* 🔹 Advanced feature engineering
* 🔹 Location-based feature extraction
* 🔹 Testing XGBoost and Gradient Boosting models
* 🔹 Model comparison dashboard
* 🔹 Deployment using Streamlit Cloud or another cloud platform
* 🔹 Improved prediction confidence and error analysis

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Regression algorithms
* Model evaluation
* Model serialization
* Streamlit application development
* Git and GitHub
* Building an end-to-end machine learning project

---

## 👨‍💻 Author

**Aizaz Ahmad**

AI Engineer & Python Developer
AI Ambassador @MindHYVE
Cohere Labs Participant

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.
