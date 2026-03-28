import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
train_df = pd.read_csv('titanic_train.csv')
y = train_df['Survived']

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) 사용할 컬럼 선택 (너무 복잡하게 안 가는 기본 세트)
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features]

    # 2) 결측치 처리
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Fare"] = X["Fare"].fillna(X["Fare"].median())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # 3) 범주형 -> 숫자로 변환 (원-핫 인코딩)
    X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)

    return X

X = preprocess_data(train_df)

# 훈련/검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Keras를 사용한 MLP 모델 생성
model = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[1],)), # 입력층
        layers.Dense(64,activation='relu'), # 첫 번째 은닉층(64개 뉴런, ReLU 활성화 함수)
        layers.Dense(32,activation='relu'), # 두 번째 은닉층(32개 뉴런, ReLU 활성화 함수)
        layers.Dense(1,activation='sigmoid'), # 출력층(이진분류이므로 시그모이드 활성화 함수)
    ]
)

# 모델 요약
model.summary()

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 훈련
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_val, y_val))

# 훈련 과정 시각화 (손실)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Trajectory')

# 훈련 과정 시각화 (정확도)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Trajectory')

plt.show()