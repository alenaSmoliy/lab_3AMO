import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загружаем встроенный датасет Iris
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Формируем обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Применяем стандартизацию к признакам
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Инициализируем модель логистической регрессии
classifier = LogisticRegression(max_iter=200)

# Обучаем модель на обучающих данных
classifier.fit(X_train, y_train)

# Делаем предсказания на тестовой выборке и оцениваем точность
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность (Accuracy): {accuracy * 100:.2f}%")

# Пример индивидуального предсказания
sample = np.array([[1, 1, 1, 1]])  # один цветок с заданными параметрами
sample_scaled = scaler.transform(sample)
predicted_id = classifier.predict(sample_scaled)[0]
predicted_name = iris.target_names[predicted_id]

print(f"Предсказанный вид для [1, 1, 1, 1]: {predicted_name}")
