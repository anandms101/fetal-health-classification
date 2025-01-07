# **Fetal Health Classification Using Machine Learning**
![Fetal Health Cover Image](/images/fetalHealth.jpg)

A machine learning project to classify fetal health based on features derived from cardiotocogram (CTG) data. This project leverages data preprocessing, visualization, and classification techniques to predict fetal health status as **Normal**, **Suspect**, or **Pathological**.

[**Visit Kaggle Notebook**](https://www.kaggle.com/code/anandms101/fetal-health-classification-model)

---

## **Project Overview**

Cardiotocograms (CTGs) are widely used in healthcare to monitor fetal health during pregnancy. This project demonstrates a complete machine learning pipeline for classifying fetal health conditions to assist healthcare professionals in early diagnosis and intervention.

---

## **Dataset**

- **Source:** [Fetal Health Classification Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data)
- **Description:** Contains 2,126 records with 22 features extracted from CTG readings.
- **Target Variable:** `fetal_health` with three classes:
  - 1: Normal
  - 2: Suspect
  - 3: Pathological

---

## **Project Workflow**

1. **Data Loading and Exploration**
   - Loaded and explored the dataset for structure and basic insights.
   - Visualized class distributions and feature correlations.

2. **Data Preprocessing**
   - Handled missing/infinite values.
   - Scaled features for better model performance.

3. **Model Training and Evaluation**
   - Trained a baseline Random Forest model.
   - Evaluated performance using metrics like confusion matrix, classification report, and ROC-AUC score.

4. **Hyperparameter Tuning**
   - Used `GridSearchCV` to optimize Random Forest hyperparameters.

5. **Optimized Model Evaluation**
   - Trained an optimized model with the best parameters.
   - Visualized feature importance and final performance metrics.

6. **Model Saving**
   - Saved the trained model for future use or deployment.

---

## **Results**

- **Best Model:** Random Forest Classifier with optimized hyperparameters.
- **Performance Metrics:**
  - **Accuracy:** 95.3%
  - **ROC-AUC Score:** 0.97
- **Key Features:**
  - `baseline value`, `accelerations`, `histogram_mean`, `fetal_movement`.

---

## **Visualizations**

The project includes the following visualizations:
- Correlation heatmap to understand feature relationships.
- Pair plots to explore feature distribution across classes.
- Boxplots for outlier detection.
- Feature importance chart for understanding model behavior.

---

## **How to Run the Notebook**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/anandms101/fetal-health-classification.git
   cd fetal-health-classification
2. **Install Required Libraries**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Notebook**
    ```bash
    jupyter notebook fetal_health_classification.ipynb
    ```
## **Technologies Used**
- Languages: Python
- Libraries: 
   - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
- Development Tools: Jupyter Notebook, GitHub

## **Future Improvements**

- Implement advanced algorithms like Gradient Boosting (XGBoost, LightGBM).
- Add cross-validation for additional robustness.
- Deploy the model using Flask, FastAPI, or Streamlit for real-world usability.

## **Acknowledgments**
- Dataset by [Kaggle](https://www.kaggle.com/).
- This project is inspired by the need to leverage machine learning for healthcare applications.