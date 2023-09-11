# -*- coding: utf-8 -*-


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('/content/drive/MyDrive/IAcone/frameWork2/Final1FrameBenji.h5')  # Reemplaza 'model.h5' con la ruta de tu archivo de modelo

# Ruta de la imagen que deseas predecir
image_path = '/content/drive/MyDrive/IAcone/frameWork2/pawn.jpeg'

# Cargar y preprocesar la imagen
img = image.load_img(image_path, target_size=(224, 224))  # Asegúrate de ajustar el tamaño objetivo según tu modelo
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalizar los valores de píxeles al rango [0, 1]

# Realizar la predicción
predictions = model.predict(img_array)

# Obtener la clase predicha
predicted_class_index = np.argmax(predictions)

# Mapear el índice de clase a la etiqueta correspondiente
class_labels = ['Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']  # Asegúrate de que las etiquetas coincidan con tu modelo

predicted_class = class_labels[predicted_class_index]

# Imprimir la predicción
print("Clase predicha:", predicted_class)

# Obtener las etiquetas verdaderas y predichas del conjunto de prueba

