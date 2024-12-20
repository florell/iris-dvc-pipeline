import pandas as pd
import joblib
from sklearn.metrics import classification_report

def evaluate_model(test_path, model_path):
    df = pd.read_csv(test_path)
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    model = joblib.load(model_path)
    predictions = model.predict(X)
    print(classification_report(y, predictions))

if __name__ == "__main__":
    evaluate_model("data/test.csv", "models/iris_model.pkl")