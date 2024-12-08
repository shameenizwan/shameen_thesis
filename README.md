# shameen_thesis


CUSTOMER CHURN LIKELIHOOD PREDICTION FOR TELECOMMUNICATION SERVICES USING MACHINE LEARNING![image](https://github.com/user-attachments/assets/0137a0a0-d6b7-4d75-8ccc-7fe900b06cb0)



# Load the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/content/Telco-churn-ml.csv'
data = pd.read_csv(file_path)

# Prepare features and target
X = data.drop('Churn', axis=1)  # Assuming 'Churn' is the target variable
y = data['Churn']

# Define numerical and categorical columns
num_features = ['Tenure Months', 'Monthly Charges']
cat_features = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
                'Multiple Lines', 'Internet Service', 'Online Security',
                'Online Backup', 'Device Protection', 'Tech Support',
                'Streaming TV', 'Streaming Movies', 'Contract',
                'Paperless Billing', 'Payment Method']

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train and evaluate model
def train_and_evaluate(model, model_name):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred))
    print(f'ROC AUC Score: {roc_auc_score(y_test, y_proba)}\n')

    return pipeline

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_pipeline = train_and_evaluate(rf_model, "Random Forest")

# Train and evaluate Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_pipeline = train_and_evaluate(gb_model, "Gradient Boosting")
