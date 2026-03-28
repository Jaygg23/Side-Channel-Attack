import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

# IMDB 데이터셋 로드 (단어 인덱스로 변환된 데이터)
max_features = 10000  # 사용할 단어의 개수 (빈도가 높은 10000개 단어)
maxlen = 200  # 입력 시퀀스의 최대 길이

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 데이터를 maxlen 길이로 패딩
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# RNN 모델 구성
model = keras.Sequential()
model.add(layers.Embedding(max_features, 32))  # Embedding 층: 단어 인덱스를 밀집 벡터(dense vector)로 변환
model.add(layers.SimpleRNN(32))  # SimpleRNN 층
model.add(layers.Dense(1, activation='sigmoid'))  # 출력층: 이진 분류 (긍정/부정)

# 모델 컴파일
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# 모델 학습
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# 모델 평가
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 훈련 및 검증 정확도와 손실 시각화
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
