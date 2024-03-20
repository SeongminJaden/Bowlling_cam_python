import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 경로
train_data_dir = r"C:\Users\ansco\OneDrive\문서\GitHub\Bowling_cam_python\Bowlling_cam_python\train_data"
model_path = "./bowling_ball_model.h5"

# 이미지 데이터 생성기 설정
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# 모델 구성
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(train_generator, epochs=5)

# 학습된 모델 저장
model.save(model_path)

# 모델을 다시 불러올 때는 아래와 같이 사용합니다.
# loaded_model = tf.keras.models.load_model(model_path)
