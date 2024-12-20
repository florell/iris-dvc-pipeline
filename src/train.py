import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json
import yaml

def train_model(train_path, model_path, metrics_path, params):
    # Загружаем данные
    df = pd.read_csv(train_path)
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']

    # Обучаем модель
    model = MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        max_iter=params["max_iter"],
        random_state=params["random_state"],
    )
    model.fit(X, y)
    joblib.dump(model, model_path)

    # Предсказания
    y_pred = model.predict(X)

    # Рассчитываем метрики
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro', labels=y.unique())
    recall = recall_score(y, y_pred, average='macro', labels=y.unique())
    f1 = f1_score(y, y_pred, average='macro', labels=y.unique())
    conf_matrix = confusion_matrix(y, y_pred).tolist()  # Преобразуем в список для сериализации

    # Собираем все метрики в одном месте
    metrics = {
        "train_accuracy": accuracy,
        "train_precision": precision,
        "train_recall": recall,
        "train_f1_score": f1,
        "train_confusion_matrix": conf_matrix
    }

    # Сохраняем метрики в JSON, перезаписывая файл
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        print(f"Ошибка при записи метрик: {e}")
        raise

if __name__ == "__main__":
    # Загружаем параметры из params.yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    # Запуск модели
    train_model("data/train.csv", "models/iris_model.pkl", "metrics/train.json", params)