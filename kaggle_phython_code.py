# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
data_path = "/kaggle/input/zipthing/student/student-mat.csv"
student_mat_df = pd.read_csv(data_path, sep=';')

features = student_mat_df.drop(columns=['G3'])
target = student_mat_df['G3']

categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

model = model_pipeline.named_steps['model']
preprocessed_features = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
feature_importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': preprocessed_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Top Features:")
print(feature_importance_df.head(10))

# Function to take input and predict the grade
def predict_grade(model_pipeline):
    print("Enter the following details to predict the student's final grade (G3):")
    
    # Input prompts for all features
    user_input = {
        'school': input("School (GP or MS): "),
        'sex': input("Sex (F or M): "),
        'age': int(input("Age: ")),
        'address': input("Address type (U - Urban or R - Rural): "),
        'famsize': input("Family size (LE3 - <=3 or GT3 - >3): "),
        'Pstatus': input("Parent's cohabitation status (T - Together or A - Apart): "),
        'Medu': int(input("Mother's education (0 to 4): ")),
        'Fedu': int(input("Father's education (0 to 4): ")),
        'Mjob': input("Mother's job (teacher, health, services, at_home, or other): "),
        'Fjob': input("Father's job (teacher, health, services, at_home, or other): "),
        'reason': input("Reason for choosing this school (home, reputation, course, other): "),
        'guardian': input("Guardian (mother, father, or other): "),
        'traveltime': int(input("Travel time to school (1 to 4): ")),
        'studytime': int(input("Weekly study time (1 to 4): ")),
        'failures': int(input("Number of past class failures (0 to 3): ")),
        'schoolsup': input("Extra educational support (yes or no): "),
        'famsup': input("Family educational support (yes or no): "),
        'paid': input("Extra paid classes (yes or no): "),
        'activities': input("Extra-curricular activities (yes or no): "),
        'nursery': input("Attended nursery school (yes or no): "),
        'higher': input("Wants higher education (yes or no): "),
        'internet': input("Internet access at home (yes or no): "),
        'romantic': input("In a romantic relationship (yes or no): "),
        'famrel': int(input("Quality of family relationships (1 to 5): ")),
        'freetime': int(input("Free time after school (1 to 5): ")),
        'goout': int(input("Going out with friends (1 to 5): ")),
        'Dalc': int(input("Workday alcohol consumption (1 to 5): ")),
        'Walc': int(input("Weekend alcohol consumption (1 to 5): ")),
        'health': int(input("Current health status (1 to 5): ")),
        'absences': int(input("Number of school absences: ")),
        'G1': int(input("First period grade (0 to 20): ")),
        'G2': int(input("Second period grade (0 to 20): "))
    }
    
    # Convert input to a DataFrame for prediction
    input_df = pd.DataFrame([user_input])
    
    # Make prediction
    predicted_grade = model_pipeline.predict(input_df)[0]
    
    print(f"\nPredicted Final Grade (G3): {predicted_grade:.2f}")

# Call the function to predict
predict_grade(model_pipeline)
