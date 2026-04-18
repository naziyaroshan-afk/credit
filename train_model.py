import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('credit (3) (1).csv')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df['approved'].value_counts())

# ==================== DATA PREPROCESSING ====================
print("\n" + "="*50)
print("PREPROCESSING STEP")
print("="*50)

X = df.drop('approved', axis=1)
y = df['approved']

# Check for missing values and handle them
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set - Approved: {y_train.sum()}, Rejected: {len(y_train)-y_train.sum()}")
print(f"Test set - Approved: {y_test.sum()}, Rejected: {len(y_test)-y_test.sum()}")

# ==================== MULTIPLE CLASSIFIERS ====================
print("\n" + "="*50)
print("TRAINING MULTIPLE CLASSIFICATION MODELS")
print("="*50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==================== FIND BEST MODEL ====================
print("\n" + "="*50)
print("MODEL COMPARISON - BEST MODEL SELECTION")
print("="*50)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    model_name: {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'ROC-AUC': metrics['roc_auc'],
        'CV Mean': metrics['cv_mean']
    }
    for model_name, metrics in results.items()
}).T

print("\n" + comparison_df.to_string())

# Best model selection based on F1-Score (balanced metric)
best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
best_model = trained_models[best_model_name]

print(f"\n✓ BEST MODEL: {best_model_name}")
print(f"  F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"  ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print(f"  Accuracy: {results[best_model_name]['accuracy']:.4f}")

# ==================== DETAILED ANALYSIS OF BEST MODEL ====================
print("\n" + "="*50)
print(f"DETAILED ANALYSIS: {best_model_name}")
print("="*50)

y_pred_best = results[best_model_name]['predictions']
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Rejected', 'Approved']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
print(f"  True Negatives: {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives: {cm[1,1]}")

# ==================== SAVE MODELS AND SCALER ====================
print("\n" + "="*50)
print("SAVING MODELS")
print("="*50)

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Saved: {best_model_name} -> best_model.pkl")

# Save all models for app
with open('all_models.pkl', 'wb') as f:
    pickle.dump({'models': trained_models, 'results': results, 'best_model': best_model_name}, f)
print("✓ Saved: All models -> all_models.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: Scaler -> scaler.pkl")

# Save model comparison
comparison_df.to_csv('model_comparison.csv')
print("✓ Saved: Model comparison -> model_comparison.csv")

# Save X_test and y_test for app
with open('test_data.pkl', 'wb') as f:
    pickle.dump({'X_test': X_test, 'y_test': y_test, 'feature_names': X.columns.tolist()}, f)
print("✓ Saved: Test data -> test_data.pkl")

print("\n" + "="*50)
print("PIPELINE COMPLETE!")
print("="*50)
print(f"\nBest Model: {best_model_name}")
print("Run 'streamlit run app.py' to launch the web interface")
