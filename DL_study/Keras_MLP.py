import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# 예제 데이터 생성 (10개의 특성을 가진 100개의 샘플)
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)  # 이진 분류 (0 또는 1)

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링 (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Keras를 사용한 MLP 모델 생성
model = keras.Sequential(
    [
        keras.Input(shape=(10,)),  # 입력층 (10개의 특성)
        layers.Dense(64, activation="relu"),  # 첫 번째 은닉층 (64개 뉴런, ReLU 활성화 함수)
        layers.Dense(32, activation="relu"),  # 두 번째 은닉층 (32개 뉴런, ReLU 활성화 함수)
        layers.Dense(1, activation="sigmoid"),  # 출력층 (이진 분류이므로 시그모이드 활성화 함수)
    ]
)

# 모델 요약
model.summary()

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 훈련
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")