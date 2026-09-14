# Buena cabros, subo este documento que explica la diferencia del ultimo modelo subido por el Gabo "notebookCorregidoMLP3Capas100train.ipynb"
# y el mio que esta en esta rama 

# Comparativa Técnica de Arquitecturas MLP: Rama @gabriel1-du vs. Rama @FatZuack

Este documento detalla las diferencias de arquitectura, canalización de datos (pipeline) y rendimiento empírico entre la implementación subida por **@gabriel1-du** y la versión optimizada en la rama de trabajo de **@FatZuack** (`feature/mlp-francisco`).

---

## 1. Resumen Ejecutivo de Métricas (Evaluación en Test)

Ambas versiones fueron evaluadas sobre el conjunto de prueba independiente (`seg_test`, 3.000 imágenes no vistas).

| Métrica | @gabriel1-du | @FatZuack (feature/mlp-francisco) | Variación |
| :--- | :---: | :---: | :---: |
| **Accuracy Global** | 62.83% | **64.47%** | **+1.64%** |
| **Test Loss** | 1.0013 | **0.9521** | **-0.0492 (Mayor certidumbre)** |
| **Macro F1-Score** | 0.62 | **0.64** | **+0.02** |
| **Weighted F1-Score** | 0.62 | **0.64** | **+0.02** |
| **Tiempo prom. por época** | ~85s - 140s | **~39s** | **~2.5x más rápido** |

---

## 2. Desglose de Rendimiento por Clase (F1-Score)

| Clase | @gabriel1-du (F1) | @FatZuack (F1) | Diferencia |
| :--- | :---: | :---: | :---: |
| **buildings** | 0.47 | **0.53** | +0.06 |
| **forest** | 0.79 | **0.79** | = |
| **glacier** | 0.63 | **0.66** | +0.03 |
| **mountain** | 0.63 | **0.65** | +0.02 |
| **sea** | 0.49 | **0.51** | +0.02 |
| **street** | 0.70 | **0.70** | = |

*Diagnóstico:* El ajuste en la rama de **@FatZuack** mejora notablemente la clasificación en clases complejas con alta dispersión geométrica (`buildings` pasa de 0.47 a 0.53) y paisajes con similitud cromática (`glacier` de 0.63 a 0.66).

---

## 3. Principales Diferencias Técnicas Aplicadas

### A. Ubicación del Data Augmentation y Rendimiento de Hardware
* **Implementación de @gabriel1-du:** Declaró las transformaciones (`RandomFlip`, `RandomRotation`, `RandomZoom`) como capas directas dentro de la definición del modelo (`tf.keras.Sequential`).
* **Implementación de @FatZuack:** Desacopló el aumento de datos de la estructura del modelo, integrándolo exclusivamente en la transformación del dataset (`dataset_train_prep.map(...)`).
* **Impacto técnico:** 
  1. El cómputo en grafo secuencial sobrecarga el hilo principal del procesador. 
  2. La versión de **@FatZuack** aprovecha la precarga asíncrona mediante `AUTOTUNE`, permitiendo que la CPU prepare los lotes en paralelo mientras se computan los gradientes. Esto reduce los tiempos de entrenamiento de ~110s por época a solo ~39s.

### B. Calibración del Dropout (Control de Regularización)
* **Implementación de @gabriel1-du:** Utilizó tasas de desconexión de `0.40` $\rightarrow$ `0.30` $\rightarrow$ `0.20`.
* **Implementación de @FatZuack:** Ajustó las tasas a `0.30` $\rightarrow$ `0.25` $\rightarrow$ `0.20`.
* **Impacto técnico:** Dado que la capa `Flatten` descompone la matriz espacial en un vector plano unidimensional, apagar el 40% de las 512 neuronas iniciales genera una pérdida prematura de información representativa (*underfitting leve*). La reducción al 30% en la rama de **@FatZuack** permitió preservar características clave del paisaje sin caer en sobreajuste, facilitando el incremento del accuracy al 64.47%.

### C. Puntos de Consistencia Compartidos
* Ambos desarrolladores mantuvieron la resolución espacial reducida a $64 \times 64 \times 3$ (reduciendo las entradas de 67.500 a 12.288 valores).
* Se mantuvo la normalización a rango $[0, 1]$ mediante `Rescaling(1./255)`.
* Se respetó la arquitectura canónica de 3 capas densas (512 - 256 - 128) con estabilización mediante `BatchNormalization` y optimizador `Adam` a `learning_rate=0.0005`.

---

## 4. Conclusión Técnica para la Integración (Merge)
La estructura de 3 capas propuesta por **@gabriel1-du** sentó las bases arquitectónicas del proyecto. Las optimizaciones introducidas por **@FatZuack** en la canalización del flujo de datos y la calibración fina del Dropout permitieron alcanzar el rendimiento óptimo del modelo dentro de los límites matemáticos del MLP, logrando **64.47% de accuracy**, **0.64 de Macro F1** y tiempos de cómputo sustancialmente más eficientes.