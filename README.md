# Proyecto: Clasificación de Paisajes con MLP (Intel Image Dataset)

Este repositorio contiene la versión optimizada del proyecto de Machine Learning basada en un **Perceptrón Multicapa (MLP puro)** para la clasificación de imágenes en 6 categorías, desarrollada en la rama `feature/mlp-francisco` por **@FatZuack** para cumplir rigurosamente con los indicadores de la pauta de evaluación académica.

---

##  Cambios y Mejoras Principales Realizadas

* **Optimización de Resolución y Entrada:** 
  * Se fijó el tamaño de entrada en **64x64 píxeles** (reduciendo desde los 150x150 originales). Esto disminuye las características de entrada de 67.500 a **12.288 variables**, reduciendo la carga de memoria RAM y mitigando la "maldición de la dimensionalidad".
* **Arquitectura de Red (MLP de 3 Capas Ocultas):**
  * Estructura en embudo gradual: **`Flatten` -> `Dense(512, ReLU)` -> `Dense(256, ReLU)` -> `Dense(128, ReLU)` -> `Dense(6, Softmax)`**.
  * Incorporación de **`BatchNormalization`** tras cada capa densa para estabilizar la distribución de las activaciones intermedias y acelerar la convergencia.
  * Tasas de **`Dropout`** progresivas y calibradas (`0.30`, `0.25`, `0.20`) para prevenir el sobreajuste (*overfitting*) sin estrangular la capacidad de representación de la red.
* **Canalización Eficiente de Datos (`tf.data`):**
  * Desacoplamiento del aumento de datos del grafo secuencial e integración directa en el pipeline asíncrono con `AUTOTUNE`, reduciendo el tiempo de entrenamiento a **~39 segundos por época**.
* **Optimización del Descenso de Gradiente:**
  * Optimizador **`Adam`** con tasa de aprendizaje inicial de `0.0005`.
  * Integración de *callbacks*: **`EarlyStopping`** (monitoreando `val_loss` con restauración de mejores pesos) y **`ReduceLROnPlateau`** para un ajuste fino ante mesetas de convergencia.

---

##  Indicadores Clave de Rendimiento (KPIs) y Resultados

Evaluación final sobre el conjunto de prueba independiente (`seg_test`, 3.000 imágenes no vistas):

* **Exactitud Global (Accuracy):** **`64.47%`**, superando holguramente el baseline aleatorio (16.6%) y el objetivo de la pauta (>60%).
* **Pérdida en Test (Loss):** **`0.9521`** (por debajo de 1.0, reflejando predicciones con mayor certidumbre).
* **Balance Multiclase (Macro F1-Score):** Promedio armónico de **`0.64`**, destacando clases con alta consistencia visual como `forest` (`F1 = 0.79`) y `street` (`F1 = 0.70`).
* **Control de Sobreajuste:** Brecha de rendimiento entre entrenamiento y validación mantenida en **< 3%**, cumpliendo la exigencia de tolerancia de la rúbrica.

---

## 📝 Documentación y Bitácoras del Repositorio

Para transparentar el ciclo de desarrollo y experimentación científica, se adjuntan los siguientes documentos en la raíz del proyecto:

1. **`EXPERIMENT_LOG.md`:** Bitácora completa de experimentos que documenta las hipótesis evaluadas, las causas del fracaso en modelos sobredimensionados (overfitting al 71% train / 59% test) y las lecciones aprendidas sobre estocasticidad en Deep Learning y Git.
2. **`COMPARATIVA_MODELOS.md`:** Análisis comparativo técnico detallado entre la propuesta arquitectónica de **@gabriel1-du** (62.83% de accuracy) y la versión optimizada de **@FatZuack** (64.47% de accuracy y reducción de tiempos de entrenamiento a más de la mitad).
3. **Análisis Cualitativo de Errores en Notebook:** Inspección visual de la Matriz de Confusión y discusión de las limitaciones del operador `Flatten` (pérdida de invariancia espacial) que justifican el techo del MLP frente a futuras arquitecturas Convolucionales (CNN).