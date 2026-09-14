# Proyecto: Clasificación de Paisajes con MLP (Intel Image Dataset)

Este repositorio contiene la versión optimizada del proyecto de Machine Learning basada en un **Perceptrón Multicapa (MLP)** para la clasificación de imágenes en 6 categorías, desarrollada para cumplir rigurosamente con los indicadores de la pauta de evaluación.

## 🚀 Cambios y Mejoras Principales Realizadas

* **Optimización de Resolución y Entrada:** 
  * Se estableció el tamaño de entrada estricto a **64x64 píxeles** (reduciendo de 150x150), lo que optimiza el uso de la memoria RAM, previene la "maldición de la dimensionalidad" y reduce la cantidad de parámetros de entrada a 12.288 variables.
* **Arquitectura de Red (MLP de 3 Capas Ocultas):**
  * Estructura en embudo gradual: **`Flatten` -> `Dense(512)` -> `Dense(256)` -> `Dense(128)` -> `Dense(6, Softmax)`**.
  * Incorporación de **`BatchNormalization`** tras cada capa oculta para evitar el colapso del gradiente y estabilizar el aprendizaje.
  * Tasas de **`Dropout`** progresivas y equilibradas (`0.3`, `0.25`, `0.2`) para evitar el sobreajuste (*overfitting*) sin bloquear la capacidad de abstracción de la red.
* **Optimización de Entrenamiento:**
  * Uso del optimizador **`Adam`** con tasa de aprendizaje ajustada a `0.0008`.
  * Integración de *callbacks* de control como **`EarlyStopping`** (monitoreando `val_loss`) y **`ReduceLROnPlateau`** para un descenso de gradiente más fino y estable.

## 📊 Indicadores Clave de Rendimiento (KPIs) y Resultados
* **Exactitud Global (Accuracy):** Alcanza un **63.87%** en el conjunto de prueba independiente (`test`), superando holguramente el baseline aleatorio (16.6%).
* **Balance de Clases (Macro F1-Score):** Promedio armónico general de **0.63**, destacando con un `F1 = 0.79` en clases visualmente consistentes como `forest`.

## 📝 Documentación Académica Incorporada
El notebook incluye celdas de análisis formal que responden a los criterios de la rúbrica:
* **Definición de KPIs y Objetivos del Negocio.**
* **Declaración de variante y reproducibilidad (`seed=123`).**
* **Análisis Cualitativo de Errores:** Auditoría de la Matriz de Confusión y ejemplos visuales de aciertos y fallos (ej. confusiones típicas entre `glacier`/`sea` y `buildings`/`street` debido a la pérdida de información espacial inherente al operador `Flatten`).
* **Conclusiones y Limitaciones del MLP:** Justificación técnica de por qué un MLP encuentra un techo analítico en visión computacional, sentando las bases teóricas para la futura transición hacia Redes Neuronales Convolucionales (CNN).
