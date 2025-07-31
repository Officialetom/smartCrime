from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_local_model(data):
    X = data.drop("label", axis=1)
    y = data["label"]
    
    # Label encode all object (text) columns in features
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Also encode the label column
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test))

def aggregate_models(models):
    coefs = np.array([model.coef_ for model in models])
    intercepts = np.array([model.intercept_ for model in models])
    avg_coef = np.mean(coefs, axis=0)
    avg_intercept = np.mean(intercepts, axis=0)

    global_model = LogisticRegression()
    global_model.coef_ = avg_coef
    global_model.intercept_ = avg_intercept
    global_model.classes_ = models[0].classes_
    return global_model
