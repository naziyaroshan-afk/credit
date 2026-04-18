import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Credit Approval Classifier", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    with open('all_models.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    return data['models'], data['results'], data['best_model'], scaler, test_data

models, results, best_model_name, scaler, test_data = load_models()
X_test = test_data['X_test']
y_test = test_data['y_test']
feature_names = test_data['feature_names']

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Model Performance", "Predictions", "Model Comparison"])

# ==================== HOME PAGE ====================
if page == "Home":
    st.title("💳 Credit Card Approval Classifier")
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Models Trained", len(models))
    
    with col2:
        st.metric("Best Model", best_model_name)
    
    with col3:
        st.metric("Best F1-Score", f"{results[best_model_name]['f1']:.4f}")
    
    st.write("\n")
    st.subheader("📊 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        ### Dataset Summary
        - **Total Records**: 1000+
        - **Features**: 5 (age, income, years_at_job, credit_score, existing_credit_cards)
        - **Target**: Credit card approval (binary: 1=Approved, 0=Rejected)
        - **Test Set Size**: 20%
        """)
    
    with col2:
        st.write("""
        ### Models Trained
        1. Logistic Regression
        2. Decision Tree
        3. Random Forest
        4. Gradient Boosting
        5. Support Vector Machine (SVM)
        6. K-Nearest Neighbors (KNN)
        """)
    
    st.write("\n")
    st.subheader("🎯 Best Performing Model")
    
    best_metrics = results[best_model_name]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
    
    with col2:
        st.metric("Precision", f"{best_metrics['precision']:.4f}")
    
    with col3:
        st.metric("Recall", f"{best_metrics['recall']:.4f}")
    
    with col4:
        st.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "Model Performance":
    st.title("📈 Model Performance Analysis")
    st.write("---")
    
    # Comparison metrics
    st.subheader("Overall Model Comparison")
    
    comparison_dict = {
        model_name: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc']
        }
        for model_name, metrics in results.items()
    }
    
    comparison_df = pd.DataFrame(comparison_dict).T
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
    
    # Best model detailed metrics
    st.subheader(f"🏆 Best Model: {best_model_name}")
    
    best_metrics = results[best_model_name]
    y_pred_best = best_metrics['predictions']
    y_proba_best = best_metrics['probabilities']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_best)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba_best)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{best_metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{best_metrics['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{best_metrics['f1']:.4f}")
    
    # Classification metrics by class
    st.subheader("Per-Class Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        st.metric("Specificity (True Negative Rate)", f"{specificity:.4f}")
    
    with col2:
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        st.metric("Sensitivity (True Positive Rate)", f"{sensitivity:.4f}")

# ==================== PREDICTIONS PAGE ====================
elif page == "Predictions":
    st.title("🔮 Make Predictions")
    st.write("---")
    
    st.subheader("Enter Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=80, value=35)
        income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=50000)
    
    with col2:
        years_at_job = st.slider("Years at Current Job", min_value=0, max_value=50, value=5)
        existing_credit_cards = st.slider("Existing Credit Cards", min_value=0, max_value=10, value=2)
    
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=600)
    
    if st.button("🎯 Predict", use_container_width=True):
        # Prepare input
        input_data = np.array([[age, income, years_at_job, credit_score, existing_credit_cards]])
        input_scaled = scaler.transform(input_data)
        
        # Get prediction from best model
        best_model = models[best_model_name]
        prediction = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0]
        
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Prediction Result")
            if prediction == 1:
                st.success("✅ APPROVED")
                confidence = probability[1] * 100
            else:
                st.error("❌ REJECTED")
                confidence = probability[0] * 100
            
            st.metric("Confidence", f"{confidence:.2f}%")
        
        with col2:
            st.subheader("📊 Probability Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Rejected', 'Approved']
            values = probability
            colors = ['#ff6b6b', '#51cf66']
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel('Probability')
            ax.set_ylim([0, 1])
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.2%}', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        st.write("---")
        st.subheader("📝 Input Summary")
        input_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_data[0]
        })
        st.dataframe(input_df, hide_index=True, use_container_width=True)

# ==================== MODEL COMPARISON PAGE ====================
elif page == "Model Comparison":
    st.title("⚖️ Model Comparison")
    st.write("---")
    
    st.subheader("Performance Metrics Across All Models")
    
    # Prepare comparison data
    comparison_dict = {
        model_name: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc']
        }
        for model_name, metrics in results.items()
    }
    
    comparison_df = pd.DataFrame(comparison_dict).T
    
    # Show table
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen').format('{:.4f}'))
    
    # Select metrics to visualize
    st.subheader("📊 Visualization")
    
    selected_metrics = st.multiselect(
        "Select metrics to compare",
        ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        default=['Accuracy', 'F1-Score', 'ROC-AUC']
    )
    
    if selected_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df.index))
        width = 0.15
        
        for i, metric in enumerate(selected_metrics):
            offset = (i - len(selected_metrics)/2 + 0.5) * width
            ax.bar(x + offset, comparison_df[metric], width, label=metric, alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
    
    # Feature importance (for tree-based models)
    st.subheader("🌳 Feature Importance")
    
    tree_models = {name: model for name, model in models.items() 
                   if hasattr(model, 'feature_importances_')}
    
    if tree_models:
        selected_tree_model = st.selectbox("Select tree-based model", list(tree_models.keys()))
        
        model = tree_models[selected_tree_model]
        importances = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance - {selected_tree_model}')
        st.pyplot(fig)
        
        st.dataframe(feature_importance_df.set_index('Feature').style.format('{:.4f}'))
    else:
        st.info("Tree-based models are required for feature importance analysis.")

# Footer
st.write("---")
st.write("💻 Developed with Streamlit | Machine Learning Pipeline with Multiple Algorithms")
