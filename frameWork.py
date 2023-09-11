# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Dense, Flatten, Dropout,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Path del dataset
path = "/content/drive/MyDrive/IAcone/frameWork2/dataChess"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU(s) found and configured.")
else:
    print("No GPU(s) found.")

# Cargar el conjunto de datos
# Cargar el conjunto de datos


train_data = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.15,  # 15% para validación
    image_size=(224, 224),
    batch_size=32,
    subset="training",
    seed=42
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.15,  # 15% para prueba
    image_size=(224, 224),
    batch_size=32,
    subset="validation",
    seed=42
)

inputs = tf.keras.Input(shape = (224,224,3))
inputs = tf.keras.Input(shape = (224,224,3))
preprocess = tf.keras.applications.mobilenet.preprocess_input(inputs)
upscale = tf.keras.layers.Lambda(lambda x : tf.image.resize_with_pad(x,
                                                                     224,
                                                                     224,
                                                                     method = tf.image.ResizeMethod.BILINEAR))(inputs)
#Clases
class_names = ['Queen', 'Rook', 'bishop', 'knight', 'pawn']

# Cargar MobileNet

mobilenet = MobileNet(include_top=True, weights='imagenet',input_tensor = (upscale), input_shape=(224, 224, 3))

data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
# Crear el modelo
model = tf.keras.Sequential([
    Rescaling(1./255, input_shape=(224, 224, 3)),
    data_augmentation,
    mobilenet,
    Flatten(),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])




#optimizer = CustomOptimizer(learning_rate=0.01)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)


# Resumen del modelo
model.summary()

# Definir EarlyStopping
my_callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
# Entrenar el modelo

hist = model.fit(train_data, validation_data=test_data, epochs=50, callbacks=my_callbacks)

# Guardar el modelo entrenado
model.save('FinalFrameWorBenji.h5')

# Guardar el modelo entrenado
model.save('FinalFrameWorBenji.h5')

# Imprimir el historial de precisión y pérdida




train_accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Entrenamiento')
plt.plot(val_accuracy, label='Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Entrenamiento')
plt.plot(val_loss, label='Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
