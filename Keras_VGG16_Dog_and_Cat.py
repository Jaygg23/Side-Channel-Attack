import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 크기
img_height = 150
img_width = 1150

# 데이터 경로(train, validation)
train_data_dir = 'PetImages' # 훈련 데이터 경로
validation_data_dir = 'PetImages' # 검증 데이터 경로

# VGG16 모델 로드 (ImageNet 가중치 사용, 최상위 분류층 제외)
base_model = VGG16(weights='imagenet',include_top=False, input_shape=(img_height, img_width, 3))

# 특성 추출 부분 고정
for layer in base_model.layers:
    layer.trainable = False

# 새로운 분류층 추가
x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation='sigmoid')(x)  # 이진 분류

# 새로운 모델 생성
model = keras.Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 데이터 증강 (Data Augmentation) - 선택 사항
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# 데이터 제너레이터 생성
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary')

# 모델 학습
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10)