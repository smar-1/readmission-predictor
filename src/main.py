from src.load_data import load_data
from src.preprocess import preprocess
from src.model import train
from src.model import evaluate
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


admissions, patients, diagnoses = load_data()
df = preprocess(admissions, patients, diagnoses)


# Logistic regression
_, y_test, y_pred, y_prob, _ = train(df,random_state=22, model=Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=3000))
        ]))
evaluate(y_test, y_pred, y_prob, model_name='LogisticRegression')


# KNN
_, y_test, y_pred, y_prob, _ = train(df,random_state=22, model=Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=51, weights='distance'))
]))
evaluate(y_test, y_pred, y_prob, model_name='knn')


# Decision Tree
_, y_test, y_pred, y_prob, _ = train(df,random_state=22, model=Pipeline([
    ('scaler', StandardScaler()),
    ('clf', DecisionTreeClassifier(max_depth=10,class_weight='balanced',random_state=42))
]))
evaluate(y_test, y_pred, y_prob, model_name='DecisionTree')

# RandomForest
_, y_test, y_pred, y_prob, _ = train(df,random_state=22, model=Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1))
]))
evaluate(y_test, y_pred, y_prob, model_name='RandomForest')


# SVM
_, y_test, y_pred, y_prob, _ = train(df,random_state=22, model=Pipeline([
    ('scaler', StandardScaler()),
    ('clf', CalibratedClassifierCV(
            LinearSVC(C=0.1, class_weight='balanced', max_iter=3000)))
]))
evaluate(y_test, y_pred, y_prob, model_name='SVM')


# XGBClassifier

# class imbalance weight
_, _, _, _, scale = train(df, random_state=22)
#  -----------------  KEEP RANDOM STATE SAME HERE       -------------------------------
_, y_test, y_pred, y_prob, _ = train(df,random_state=22, model=Pipeline([
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=4,scale_pos_weight=scale,random_state=42,eval_metric='auc',erbosity=0))
]))

evaluate(y_test, y_pred, y_prob, model_name='XGBClassifier')
