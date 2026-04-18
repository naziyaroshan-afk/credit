# 💳 Credit Card Approval Classifier

A machine learning classification project that predicts credit card approval using multiple algorithms and provides an interactive Streamlit dashboard.

## 📊 Project Overview

This project implements a complete ML pipeline for credit card approval prediction using supervised learning techniques. It includes:
- Multiple classification algorithms
- Comprehensive preprocessing techniques
- Model evaluation and comparison
- Interactive web dashboard with predictions
- Model deployment ready

## 🎯 Best Model Performance

**Random Forest Classifier** (Best Model)
- **Accuracy**: 94.00%
- **F1-Score**: 90.48%
- **ROC-AUC**: 93.83%
- **Precision**: 83.82%
- **Recall**: 98.28%

### All Models Trained
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 94.00% | 90.48% | 93.83% |
| Gradient Boosting | 92.50% | 87.80% | 93.23% |
| Decision Tree | 91.50% | 85.71% | 90.44% |
| SVM | 91.00% | 85.00% | 93.01% |
| KNN | 87.50% | 80.00% | 93.12% |
| Logistic Regression | 84.00% | 71.43% | 90.52% |

## 📁 Dataset

**File**: `credit (3) (1).csv`

**Features** (5 input variables):
- `age`: Age of the applicant
- `income`: Annual income ($)
- `years_at_job`: Years employed at current job
- `credit_score`: Credit score (300-850)
- `existing_credit_cards`: Number of existing credit cards

**Target**: `approved` (0=Rejected, 1=Approved)

**Statistics**:
- Total Records: 1,000
- Approved: 289 (28.9%)
- Rejected: 711 (71.1%)
- Train/Test Split: 80/20 with stratification

## 🛠️ Preprocessing Techniques

1. **Missing Value Handling**: Mean imputation for numerical features
2. **Feature Scaling**: StandardScaler normalization (zero mean, unit variance)
3. **Train-Test Split**: Stratified 80-20 split to preserve class distribution
4. **Class Imbalance**: Handled through stratification and evaluation metrics

## 🤖 Classification Algorithms

1. **Logistic Regression** - Linear classifier
2. **Decision Tree** - Tree-based classifier
3. **Random Forest** - Ensemble of decision trees (BEST)
4. **Gradient Boosting** - Sequential boosting algorithm
5. **Support Vector Machine (SVM)** - Kernel-based classifier
6. **K-Nearest Neighbors (KNN)** - Instance-based classifier

## 📦 Installation

### Requirements
- Python 3.8+
- pip or conda

### Step 1: Clone Repository
```bash
git clone https://github.com/naziyaroshan-afk/credit.git
cd credit
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Option 1: Train Models from Scratch
```bash
python train_model.py
```

This will:
- Load the dataset
- Preprocess the data
- Train all 6 classification models
- Evaluate and compare models
- Save the best model and artifacts

### Option 2: Launch Interactive Dashboard
```bash
streamlit run app.py
```

Then open your browser and navigate to:
- **Local**: http://localhost:8501
- **Network**: http://192.168.0.181:8501

## 📊 Dashboard Features

### 🏠 Home Page
- Project overview
- Best model metrics
- Key statistics

### 📈 Model Performance
- Confusion matrix visualization
- ROC curve analysis
- Detailed classification metrics
- Per-class performance statistics

### 🔮 Predictions
- Interactive form to input customer details
- Real-time prediction with confidence score
- Probability distribution visualization
- Input summary

### ⚖️ Model Comparison
- Performance metrics comparison table
- Visualization of multiple metrics
- Feature importance analysis (tree-based models)

## 📁 Project Files

```
credit/
├── train_model.py              # ML training pipeline
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── credit (3) (1).csv          # Dataset
├── best_model.pkl              # Trained Random Forest model
├── all_models.pkl              # All trained models
├── scaler.pkl                  # Feature scaler
├── test_data.pkl               # Test dataset and features
├── model_comparison.csv        # Model metrics summary
├── .gitignore                  # Git configuration
└── README.md                   # This file
```

## 🔍 Model Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Accuracy of positive predictions (Approved)
- **Recall**: Coverage of actual positive cases (True Positive Rate)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (0-1 scale)

## 📊 Confusion Matrix (Best Model)

```
                Predicted
              Rejected  Approved
Actual
Rejected         131       11      (99% Specificity)
Approved           1       57      (98% Sensitivity)
```

## 🎯 Key Insights

1. **Random Forest** outperforms other models with 94% accuracy
2. **High Recall (98.3%)** ensures most approvals are caught
3. **Low False Negative Rate** (1 out of 58) minimizes missed approvals
4. **Balanced Precision (83.8%)** reduces false positives
5. **Cross-validation score (93.3%)** indicates minimal overfitting

## 🔬 Hyperparameters

### Random Forest (Best Model)
- Number of trees: 100
- Max depth: default
- Min samples split: 2
- Random state: 42

## 💡 Future Improvements

1. Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
2. Class imbalance handling (SMOTE, class weights)
3. Feature engineering (polynomial features, interactions)
4. Cross-validation with more folds
5. Model explainability using SHAP or LIME
6. Docker containerization for deployment
7. API deployment using FastAPI or Flask

## 👤 Author

**Naziya Roshan**
- GitHub: [@naziyaroshan-afk](https://github.com/naziyaroshan-afk)
- Email: naziyaroshan@gmail.com

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: April 18, 2026
