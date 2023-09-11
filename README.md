# Uso_de_framework



# README - Aplicación de Framework de Clasificación de Piezas de Ajedrez

Este repositorio contiene dos archivos Python: `frameWork.py` y `testMolde.py`. Estos códigos se utilizan para entrenar un modelo de clasificación de piezas de ajedrez y realizar predicciones sobre imágenes de piezas de ajedrez.

## `frameWork.py`

### Descripción
`frameWork.py` es un archivo que contiene el código para entrenar un modelo de clasificación de piezas de ajedrez utilizando TensorFlow y MobileNet como base. El modelo se entrena utilizando un conjunto de datos de imágenes de piezas de ajedrez y se guarda como `FinalFrameWorBenji.h5`. Además, se proporciona un análisis del historial de precisión y pérdida durante el entrenamiento.

### Uso
Para ejecutar `frameWork.py`, asegúrate de tener TensorFlow instalado en tu entorno. Puedes ejecutarlo directamente desde un entorno de desarrollo Python.

```bash
python frameWork.py
```

### Dependencias
- TensorFlow
- scikit-learn
- Matplotlib
- Seaborn

## `testMolde.py`

### Descripción
`testMolde.py` es un archivo que se utiliza para cargar el modelo previamente entrenado (`Final1FrameBenji.h5`) y realizar predicciones sobre imágenes de piezas de ajedrez. El resultado de la predicción se muestra en la consola.

### Uso
Antes de ejecutar `testMolde.py`, asegúrate de que el modelo `Final1FrameBenji.h5` está en la ubicación correcta. Luego, puedes ejecutar el archivo para realizar predicciones sobre imágenes de piezas de ajedrez.

```bash
python testMolde.py
```

### Dependencias
- TensorFlow
- Numpy

## Notas importantes
- Asegúrate de que las rutas de los directorios y nombres de archivo en ambos códigos estén configurados correctamente según tu entorno.
- En `frameWork.py`, se utiliza MobileNet preentrenado con imágenes de ImageNet como base para la clasificación. Asegúrate de que esta elección sea adecuada para tu tarea específica.
- Asegúrate de que las etiquetas de clases en `frameWork.py` coincidan con las clases que deseas clasificar.

---

Espero que este README sea útil para comprender y utilizar los códigos proporcionados. Asegúrate de personalizar las rutas de los archivos y las etiquetas de clase según tus necesidades específicas.