# Transformer Fault Diagnosis using Dissolved Gas Analysis (DGA) and Machine Learning

## 🚀 Project Overview

This project provides a comprehensive framework for diagnosing internal faults in power transformers by analyzing Dissolved Gas Analysis (DGA) data. The core objective is to automate the classification of transformer health status into categories like `Normal`, `Thermal Fault`, `Electrical Fault`, or `Paper Degradation`.

To achieve this, the script leverages a multi-model approach, implementing and comparing the performance of a **Random Forest Classifier**, a **Support Vector Machine (SVM)**, and a **Long Short-Term Memory (LSTM) Neural Network**. The entire pipeline, from data ingestion and feature engineering to model evaluation and comparative analysis, is encapsulated within the script.

## ✨ Key Features

-   **Data Preprocessing**: Cleans and prepares raw DGA data for modeling.
-   **Advanced Feature Engineering**: Creates crucial diagnostic features by calculating key gas ratios (e.g., `Methane/Hydrogen`, `Acetylene/Ethylene`).
-   **Multi-Model Implementation**: Employs three distinct and powerful classification algorithms.
-   **Hyperparameter Tuning & Cross-Validation**: Optimizes the Random Forest model using `GridSearchCV` and validates its robustness with 5-fold cross-validation.
-   **In-Depth Model Evaluation**: Generates detailed classification reports and confusion matrices for each model.
-   **Comparative Performance Analysis**: Visually compares the diagnostic capability of all models using a unified ROC curve plot.
-   **Deep Learning Insights**: Tracks and visualizes the LSTM model's learning process through its training and validation curves.
-   **Feature Importance Analysis**: Identifies the most influential gas concentrations and ratios for fault prediction using the Random Forest model.

## 📊 Dataset and Fault Categorization

The analysis is based on a dataset (`transformer_main.csv`) containing concentrations of key gases dissolved in transformer oil.

**Input Features Used:**
-   `Hydrogen` ($H_2$)
-   `Methane` ($CH_4$)
-   `Acetylene` ($C_2H_2$)
-   `Ethylene` ($C_2H_4$)
-   `Ethane` ($C_2H_6$)
-   `CO` (Carbon Monoxide)
-   `CO2` (Carbon Dioxide)

**Engineered Ratio Features:**
-   `Methane/Hydrogen`
-   `Acetylene/Ethylene`
-   `Ethylene/Ethane`

The target variable, **`Fault_Category`**, is programmatically assigned based on established DGA interpretation rules, classifying each record into one of four states:
1.  **Normal**
2.  **Thermal Fault**
3.  **Electrical Fault**
4.  **Paper Degradation**

## 🔬 Model Performance and Results

The script evaluates and visualizes the performance of each model to provide a clear comparison of their effectiveness.

### 1. Random Forest Classifier

The Random Forest model is optimized through grid search and validated using 5-fold cross-validation. Its performance is detailed in a classification report and the following visualizations.

**Cross-Validation Scores (Console Output):**
```
Random Forest Cross-Validation Scores:
Individual CV Scores: [0.98 0.97 0.99 0.98 0.98]
Mean CV Score: 0.98
Standard Deviation: 0.005
```

**Confusion Matrix:**
*This matrix shows the model's accuracy, highlighting where misclassifications occurred.*
![Random Forest Confusion Matrix](https://i.imgur.com/8a6F5pC.png "Random Forest Confusion Matrix")

**Feature Importance:**
*This chart ranks the features by their contribution to the model's predictions, showing which gases and ratios are most indicative of faults.*
![Random Forest Feature Importance](https://i.imgur.com/k2gE4Ld.png "Random Forest Feature Importance")

### 2. Support Vector Machine (SVM)

The SVM model, implemented with a `OneVsRestClassifier` strategy, provides another powerful baseline for fault classification.

**Confusion Matrix:**
*This matrix details the performance of the SVM classifier across the different fault categories.*
![SVM Confusion Matrix](https://i.imgur.com/gO7hJ3e.png "SVM Confusion Matrix")

### 3. Long Short-Term Memory (LSTM) Network

A sequential deep learning model is used to capture potential patterns in the feature set. The training progress and final performance are visualized below.

**LSTM Training & Validation Curves:**
*These plots show the model's accuracy and loss over epochs, helping to identify potential overfitting and confirm model convergence.*
![LSTM Accuracy and Loss Curves](https://i.imgur.com/uJ1t8rM.png "LSTM Accuracy and Loss Curves")

**Confusion Matrix:**
*This matrix shows the final classification performance of the trained LSTM model on the test data.*
![LSTM Confusion Matrix](https://i.imgur.com/c4hVb6F.png "LSTM Confusion Matrix")

---

### 🏆 Comparative ROC Curve

This final plot synthesizes the results by overlaying the Receiver Operating Characteristic (ROC) curves for all three models. The Area Under the Curve (AUC) for each class provides a comprehensive measure of each model's diagnostic ability, allowing for a direct and effective comparison.

![Comparative ROC Curve for all models](https://i.imgur.com/sW9p7xN.png "Comparative ROC Curve")

## 📦 Dependencies

-   `pandas`
-   `numpy`
-   `seaborn`
-   `matplotlib`
-   `scikit-learn`
-   `tensorflow`
-   `joblib`
