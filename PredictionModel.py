import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pickle
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('generated_data.csv')
df.dropna(inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df[['Sex_Of_Driver', 'Vehicle_Type', 'Road_Type']] = imputer.fit_transform(df[['Sex_Of_Driver', 'Vehicle_Type', 'Road_Type']])

imputer = SimpleImputer(strategy='mean')
df[['Speed_limit', 'Number_of_Pasengers']] = imputer.fit_transform(df[['Speed_limit', 'Number_of_Pasengers']])

# Label encoding
le_day = LabelEncoder()
le_light = LabelEncoder()
le_severity = LabelEncoder()

df['Day'] = le_day.fit_transform(df['Day_of_Week'])
df['Light'] = le_light.fit_transform(df['Light_Conditions'])
df['Severity'] = le_severity.fit_transform(df['Accident_Severity'])

df.drop(['Day_of_Week', 'Light_Conditions', 'Accident_Severity'], axis=1, inplace=True)

# Split data
X = df.drop(['Severity'], axis=1)
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
xgb_model = xgb.XGBClassifier()

# Hyperparameter tuning for XGBoost
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'min_child_weight': [1, 5, 10],
}
random_search = RandomizedSearchCV(xgb_model, param_distributions, n_iter=10, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
random_search.fit(X_train, y_train)
best_xgb_model = random_search.best_estimator_

# Train models with best hyperparameters
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
best_xgb_model.fit(X_train, y_train)

# Evaluate models
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
gb_accuracy = accuracy_score(y_test, gb.predict(X_test))
xgb_accuracy = accuracy_score(y_test, best_xgb_model.predict(X_test))

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', rf),
    ('gb', gb),
    ('xgb', best_xgb_model)
], voting='soft')

voting_clf.fit(X_train, y_train)
voting_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))
print(f"Voting Classifier Accuracy: {voting_accuracy:.4f}")

# Save the best model
best_model = max([(rf, rf_accuracy), (gb, gb_accuracy), (best_xgb_model, xgb_accuracy), (voting_clf, voting_accuracy)], key=lambda x: x[1])[0]
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Generate correlation matrix
corr_matrix = df.corr()
corr_matrix.to_csv('correlation_matrix.csv')
