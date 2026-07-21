import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import os

FEATURES = [
    'anchor_age', 'gender', 'admission_type', 'admission_location',
    'discharge_location', 'insurance', 'metacanc', 'carit', 'obes',
    'pvd', 'depre', 'solidtum', 'rheumd', 'diabc', 'ld', 'aids',
    'dane', 'drug', 'pud', 'hypunc', 'rf', 'diabunc', 'hypothy',
    'cpd', 'lymph', 'ond', 'wloss', 'coag', 'hypc', 'fed', 'pcd',
    'chf', 'valv', 'psycho', 'alcohol', 'blane', 'comorbidity_score'
]

CAT_COLS = ['gender', 'admission_type', 'admission_location', 'discharge_location', 'insurance']


def prepare_features(df):
    X = df[FEATURES].copy()
    X = pd.get_dummies(X, columns=CAT_COLS)
    X = X.fillna(0)
    y = df['readmitted_30']
    return X, y

def train(df,test_size=0.2,random_state=None, model=None):
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model is None:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=3000))
        ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos

    return model, y_test, y_pred, y_prob, scale


def evaluate(y_test, y_pred, y_prob, model_name='model', extra_metrics=None):
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    output_dir = os.path.join(r'C:\Users\amari\OneDrive\Desktop\readmission-predictor\outputs', model_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"=== {model_name} ===\n\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nROC-AUC: {roc_auc_score(y_test, y_prob):.3f}\n")
        if extra_metrics:
            f.write("\n--- Additional Results ---\n")
            for line in extra_metrics:
                f.write(line + "\n")

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f'{model_name} — ROC Curve')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    #plt.show()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f'{model_name} — Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    #plt.show()

    print(f"Outputs saved to outputs/{model_name}/")