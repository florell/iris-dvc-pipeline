import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json
import yaml
import numpy as np
import os

def train_model(train_path, model_path, metrics_path, params):
    # Загружаем данные
    df = pd.read_csv(train_path)
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']

    # Добавляем больше шума к признакам
    noise = np.random.normal(0, 1.0, X.shape)  # Увеличиваем стандартное отклонение шума
    X_noisy = X + noise

    # Обучаем модель с изменёнными гиперпараметрами для снижения точности
    model = MLPClassifier(
        hidden_layer_sizes=(5, 5),  # Меньше нейронов в скрытых слоях
        max_iter=100,  # Меньше итераций
        random_state=params["random_state"],
        alpha=0.01,  # Добавляем регуляризацию для предотвращения переобучения
    )
    model.fit(X_noisy, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Предсказания
    y_pred = model.predict(X_noisy)

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