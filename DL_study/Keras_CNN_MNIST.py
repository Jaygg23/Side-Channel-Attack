import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 데이터 정규화(0~1 범위로 스케일링)
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

# 입력 데이터 차원 확장 (채널 차원 추가)
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)

# CNN 모델 구성
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),  # 입력층: MNIST 이미지 크기 28x28, 흑백 이미지이므로 채널 1
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),  # 합성곱 층: 32개 필터, 3x3 커널, ReLU 활성화 함수
        layers.MaxPooling2D(pool_size=(2, 2)),  # 풀링 층: 2x2 Max Pooling
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),  # 합성곱 층: 64개 필터, 3x3 커널, ReLU 활성화 함수
        layers.MaxPooling2D(pool_size=(2, 2)),  # 풀링 층: 2x2 Max Pooling
        layers.Flatten(),  # Flatten 층: 2차원 특징 맵을 1차원 벡터로 변환
        layers.Dropout(0.5),  # Dropout 층: 과적합 방지를 위해 일부 뉴런을 무작위로 비활성화
        layers.Dense(10, activation="softmax"),  # 출력층: 10개 숫자 클래스에 대한 확률 분포 출력
    ]
)

# 모델 요약
model.summary()

# 모델 컴파일
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 학습
model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

# 모델 평가
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
