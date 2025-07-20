# Heart Disease Prediction Project

## Project Overview
This project focuses on predicting heart disease using machine learning techniques. It uses a dataset containing various medical attributes to build a classification model that can predict whether a patient has heart disease.

## Dataset Description
The dataset (`heart.csv`) contains medical and demographic information about patients along with a binary target variable indicating the presence of heart disease. The dataset includes the following features:

- **Age**: Age of the patient in years
- **Sex**: Gender of the patient (M: Male, F: Female)
- **ChestPainType**: Type of chest pain experienced
  - TA: Typical Angina
  - ATA: Atypical Angina
  - NAP: Non-Anginal Pain
  - ASY: Asymptomatic
- **RestingBP**: Resting blood pressure in mm Hg
- **Cholesterol**: Serum cholesterol in mg/dl
- **FastingBS**: Fasting blood sugar > 120 mg/dl (1: true, 0: false)
- **RestingECG**: Resting electrocardiogram results
  - Normal: Normal
  - ST: Having ST-T wave abnormality
  - LVH: Showing probable or definite left ventricular hypertrophy
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y: Yes, N: No)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: Slope of the peak exercise ST segment
  - Up: Upsloping
  - Flat: Flat
  - Down: Downsloping
- **HeartDisease**: Target variable (1: heart disease, 0: no heart disease)

## Project Structure
The project follows these main steps:

1. **Data Loading and Exploration**
   - Loading the dataset
   - Checking for missing values
   - Exploring data types and unique values
   - Statistical summary of the data

2. **Data Visualization**
   - Distribution of features
   - Relationship between features and heart disease

3. **Data Preprocessing**
   - Encoding categorical variables
     - Binary encoding for 'Sex' and 'ExerciseAngina'
     - One-hot encoding for 'ChestPainType', 'RestingECG', and 'ST_Slope'
   - Feature scaling
     - Standard scaling for 'Age', 'RestingBP', and 'MaxHR'
     - Robust scaling for 'Cholesterol' and 'Oldpeak'

4. **Model Building**
   - Train-test split (85% training, 15% testing)
   - Random Forest Classifier implementation
   - Feature importance analysis
   - Feature selection (top 10 features)

5. **Model Evaluation**
   - Confusion matrix
   - Classification report (precision, recall, F1-score)
   - Balanced accuracy score

6. **Model Optimization**
   - Hyperparameter tuning for Random Forest
   - K-Nearest Neighbors implementation with GridSearchCV

## Key Files
- `heart.py`: Main Python script containing the entire analysis and modeling process
- `heart.csv`: Dataset file containing patient records
- `label_encodings.json`: JSON file storing the label encoding mappings
- `onehot_feature_names.json`: JSON file storing one-hot encoded feature names
- `standard_scaler.pkl`: Saved StandardScaler model for numerical features
- `robust_scaler.pkl`: Saved RobustScaler model for numerical features with outliers

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

## Model Performance
The Random Forest Classifier achieved strong performance after feature selection and hyperparameter tuning:
- Balanced accuracy: ~0.9
- High precision and recall for both classes
- Effective identification of key predictive features

## Feature Importance
The analysis identified the most important features for predicting heart disease, which include:
- ST_Slope (encoded features)
- Oldpeak
- ChestPainType (encoded features)
- ExerciseAngina
- Age
- MaxHR

## Future Improvements
- Implement additional models (SVM, Neural Networks, etc.)
- Perform more extensive hyperparameter tuning
- Explore feature engineering opportunities
- Implement cross-validation for more robust evaluation
- Deploy the model as a web application for practical use

## Usage
To run this project:
1. Ensure all dependencies are installed
2. Place the dataset in the appropriate directory
3. Run the `heart.py` script
4. Examine the output visualizations and model performance metrics

## Conclusion
This project demonstrates the effective use of machine learning techniques to predict heart disease based on medical attributes. The Random Forest model shows promising results, and the feature importance analysis provides valuable insights into the key factors associated with heart disease.