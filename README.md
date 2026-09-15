# Proyecto: Clasificación de Paisajes con MLP (Intel Image Dataset)

Este repositorio contiene el proyecto de Machine Learning basado en un **Perceptrón Multicapa (MLP puro)** para la clasificación de imágenes en 6 categorías (edificios, bosques, glaciares, montañas, mar y calles). En este README se comparan las métricas de **validación (`val_accuracy` y `loss`)** de las variantes de MLP de **3 capas ocultas** desarrolladas en el proyecto.

---

##  Colaboradores

| Usuario | Rol |
| --- | --- |
| **@FatZuack** (Francisco Salazar) | Autor del notebook optimizado (`notebookCorregidoMLP3Capas(FRANCISCO) (1).ipynb`), desarrollado en la rama `feature/mlp-francisco`. |
| **@Gabriel1-du** (Gabriel Duran) | Autor de los notebooks `NotebookMLPLay3.ipynb` y `notebookMLPPat2Lay3.ipynb`. El resto del trabajo se realizó en la rama `master`. |
| **Martín Higuera** | Integrante del equipo; autor de la presentación (`4. Presentacion.ipynb`), documentación y guion de live coding. |

---

## 🧠Estructura de Uso (Arquitectura del MLP)

Los notebooks comparten la misma estructura de MLP de **3 capas ocultas** en forma de embudo:

**Canalización de uso:**

1. **Entrada (`Flatten`):** las imágenes se redimensionan a **64x64 píxeles** (RGB) y se aplanan a un vector de **12.288** características.
2. **Capa oculta 1:** `Dense(512, ReLU)` → `BatchNormalization` → `Dropout(0.30)`.
3. **Capa oculta 2:** `Dense(256, ReLU)` → `BatchNormalization` → `Dropout(0.25)`.
4. **Capa oculta 3:** `Dense(128, ReLU)` → `BatchNormalization` → `Dropout(0.20)`.
5. **Salida:** `Dense(6, Softmax)` (una neurona por clase).

**Parámetros totales:** 6.460.550 (~6,46 millones) — 6.291.968 en la primera capa, 131.328 en la segunda, 32.896 en la tercera y 774 en la salida.

**Optimización:** optimizador `Adam`, pérdida `sparse_categorical_crossentropy` y métrica `accuracy`, con los callbacks `EarlyStopping` (restaura los mejores pesos) y `ReduceLROnPlateau`.

---

## 🔍 Modelos Comparados

| Nombre | Archivo | Usuario |
| --- | --- | --- |
| **`notebook_MLP_ley3`** | `1. NotebookMLPLay3.ipynb` | @Gabriel1-du |
| **`notebook_MLP_AT2_ley3`** | `2. notebookMLPPat2Lay3.ipynb` | @Gabriel1-du |
| **`notebook_corregido_MLP3_capas_de_Francisco`** | `3. notebookCorregidoMLP3Capas(FRANCISCO) (1).ipynb` | @FatZuack |
| **`presentacion`** | `4. Presentacion.ipynb` | @Gabriel1-du / Martín Higuera |

### Diferencia clave en el uso de los datos

* El notebook de **Francisco** **extrae el 20 % de los datos de entrenamiento** (`seg_train`) y los separa para usarlos como conjunto de **validación**: entrena con **11.228** imágenes (80 %) y valida con **2.806** (20 %).
* Los notebooks de **@Gabriel1-du** (y el `4. Presentacion.ipynb`) ocupan la **carpeta de entrenamiento por completo**: entrenan con las **14.034** imágenes (100 % de `seg_train`) y usan las **3.000** imágenes de `seg_test` como validación.
* Los tres modelos comparados se evalúan finalmente sobre el mismo `seg_test` (3.000 imágenes nunca vistas durante el entrenamiento).

### Tabla de configuración

| Aspecto | `notebook_MLP_ley3` | `notebook_MLP_AT2_ley3` | `notebook_corregido_MLP3_capas_de_Francisco` |
| --- | --- | --- | --- |
| Imágenes de entrenamiento | 14.034 (100 % `seg_train`) | 14.034 (100 % `seg_train`) | **11.228 (80 % `seg_train`)** |
| Imágenes de validación | 3.000 (`seg_test`) | 3.000 (`seg_test`) | **2.806 (20 % `seg_train`)** |
| Imágenes de test | 3.000 (`seg_test`) | 3.000 (`seg_test`) | 3.000 (`seg_test`) |
| Épocas programadas | 50 | 50 | 40 |
| Tasa de aprendizaje inicial | 0.0005 | 0.0005 | 0.0008 |
| Arquitectura | 512 → 256 → 128 + BatchNorm + Dropout (0.30/0.25/0.20) | 512 → 256 → 128 + BatchNorm + Dropout (0.30/0.25/0.20) | 512 → 256 → 128 + BatchNorm + Dropout (0.30/0.25/0.20) |
| Parámetros totales | 6.460.550 | 6.460.550 | 6.460.550 |

---

##  Comparación de Métricas (Validación)

La **`val_accuracy`** (mejor cuanto más alta) y la **`loss`** (mejor cuanto más baja) fueron las métricas principales de la comparación. Se muestran tanto el **mejor valor alcanzado** durante el entrenamiento como el valor de la **última época registrada**.

### Mejores valores de validación

| Métrica | `notebook_MLP_ley3` | `notebook_MLP_AT2_ley3` | `notebook_corregido_MLP3_capas_de_Francisco` |
| --- | --- | --- | --- |
| **Mejor `val_accuracy`** | 0.6453 (época 28) | 0.6430 (época 43) | **0.6586 (época 37)** |
| **Mejor `loss` (mínima)** | 0.9444 (época 43) | 0.9510 (época 46) | **0.9205 (época 37)** |

### Última época registrada

| Métrica | `notebook_MLP_ley3` (ép. 50) | `notebook_MLP_AT2_ley3` (ép. 50) | `notebook_corregido_MLP3_capas_de_Francisco` (ép. 39/40) |
| --- | --- | --- | --- |
| **`val_accuracy`** | 0.6453 | 0.6423 | **0.6515** |
| **`loss`** | 0.9448 | 0.9545 | **0.9320** |

> El valor de `loss` (error) desciende de forma continua durante el entrenamiento hasta las cifras de la tabla, y `val_accuracy` va diferenciándose entre los tres modelos: el notebook de Francisco alcanza el puntaje más alto en ambas métricas.

---

##  Comparación de Métricas (Evaluación en Test)

Evaluación final sobre `seg_test` (3.000 imágenes):

| Métrica | `notebook_MLP_ley3` | `notebook_MLP_AT2_ley3` | `notebook_corregido_MLP3_capas_de_Francisco` |
| --- | --- | --- | --- |
| **Accuracy (test)** | 0.6423 | 0.6413 | **0.6447** |
| **Loss (test)** | **0.9444** | 0.9510 | 0.9521 |
| **Macro F1-score** | 0.64 | 0.63 | 0.64 |
| Mejor clase (F1) | `forest` (0.80) | `forest` (0.79) | `forest` (0.79) |

---

##  KPIs del Proyecto

| KPI | Objetivo | Resultado (mejor modelo) | Estado |
| --- | --- | --- | --- |
| Accuracy en test | > 60 % | 64.47 % | Cumplido |
| Brecha \|train − val\| | ≤ 5 % | 3.84 % (acc: 0.6131 − 0.6515, última época) | Cumplido |
| Macro F1-score | ≥ 0.60 | 0.64 | Cumplido |
| Tiempo por época | < 60 s | ~29 s (GPU) | Cumplido |

> Los cuatro KPIs establecidos en la pauta del proyecto se cumplen con holgura en el mejor modelo (`3. notebookCorregidoMLP3Capas(FRANCISCO) (1).ipynb`).

---

##  F1-score por Clase (Mejor Modelo)

| Clase | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| buildings | 0.55 | 0.51 | 0.53 |
| forest | 0.79 | 0.80 | 0.79 |
| glacier | 0.65 | 0.66 | 0.66 |
| mountain | 0.60 | 0.71 | 0.65 |
| sea | 0.58 | 0.45 | 0.51 |
| street | 0.68 | 0.72 | 0.70 |
| **Macro avg** | **0.64** | **0.64** | **0.64** |

> Las clases con mejor desempeño son `forest` y `street`, que presentan patrones visuales más distintivos. Las clases con más confusión son `buildings` y `sea`, que comparten tonos y texturas con `glacier` y `mountain` respectivamente.

---

##  Conclusión

El modelo que tuvo **mejor resultado** es aquel que obtiene el **mejor puntaje en las métricas de la comparación** (`val_accuracy` y `loss`): el **`notebook_corregido_MLP3_capas_de_Francisco`**.

* Alcanza la **mayor `val_accuracy`** (0.6586 frente a 0.6453 y 0.6430).
* Registra la **menor `loss`** (0.9205 frente a 0.9444 y 0.9510).
* También obtiene la **mayor exactitud en test** (64.47 % frente a 64.23 % y 64.13 %).

Este resultado se logró incluso entrenando con **menos imágenes** (el 80 % de `seg_train`, al reservar el 20 % para validación): al separar parte del propio `seg_train` como conjunto de validación, el `EarlyStopping` y el `ReduceLROnPlateau` monitorean el desempeño sobre datos `seg_train` reales y detienen/ajustan mejor el entrenamiento. En contraste, los modelos que usan el 100 % de `seg_train` para entrenar terminan con `val_accuracy` y `loss` ligeramente inferiores.

Por lo tanto, para este proyecto el modelo ganador es el del **notebook de Francisco**, que con la misma arquitectura MLP (512 → 256 → 128, con BatchNorm y Dropout) y una mejor estrategia de validación consigue el desempeño más alto en todas las métricas principales.

> **Nota sobre la presentación:** el notebook de presentación (`4. Presentacion.ipynb`) utiliza el **mismo modelo y arquitectura** que el de Francisco, pero con la estrategia del 100 % de `seg_train` para entrenar (tal como los notebooks de @Gabriel1-du). Las diferencias en las métricas se deben únicamente a la partición de datos: el 80/20 de Francisco produce un `EarlyStopping` más informativo, lo que resulta en una métrica ligeramente superior.

---

##  Mejoras y Resultados Clave del Modelo Ganador

* **Optimización de Resolución y Entrada:** imágenes con resolución nativa de **150×150×3** (67.500 valores) reducidas a **64×64×3** (12.288 valores), lo que disminuye la memoria RAM utilizada y mitiga la "maldición de la dimensionalidad".
* **Canalización eficiente de datos (`tf.data`):** desacoplamiento del aumento de datos del grafo secuencial e integración en el pipeline asíncrono con `AUTOTUNE`.
* **Optimización del descenso de gradiente:** optimizador `Adam`, `EarlyStopping` (monitoreando `loss` con restauración de los mejores pesos) y `ReduceLROnPlateau` para un ajuste fino ante mesetas de convergencia.
* **Resultados sobre `seg_test`:** exactitud global de **64.47 %** (supera el baseline aleatorio de 16.6 % y el objetivo de la pauta >60 %), pérdida en test **0.9521** y balance multiclase (**Macro F1 = 0.64**), destacando `forest` (F1 = 0.79) y `street` (F1 = 0.70).

---

##  Documentación del Repositorio

**Notebooks:**

* **`1. NotebookMLPLay3.ipynb`:** MLP de 3 capas sobre el 100 % de `seg_train` (@Gabriel1-du).
* **`2. notebookMLPPat2Lay3.ipynb`:** MLP de 3 capas sobre el 100 % de `seg_train` (@Gabriel1-du).
* **`3. notebookCorregidoMLP3Capas(FRANCISCO) (1).ipynb`:** MLP de 3 capas con 20 % de `seg_train` separado para validación (@FatZuack).
* **`4. Presentacion.ipynb`:** notebook de presentación del proyecto, reproduce el pipeline del mejor modelo y genera las gráficas de evaluación (Martín Higuera / @Gabriel1-du).

**Documentos (`documentos/`):**

* **`COMPARATIVA_MODELOS.md`:** tabla comparativa completa de todas las variantes con métricas reales extraídas de cada notebook.
* **`EXPERIMENT_LOG.md`:** registro detallado de los experimentos, incluyendo valores de cada época y conclusiones.
* **`Estructura presentacion 2.0.docx`:** guion para la presentación en clase.
* **`GUION_PRESENTACION_LIVE.md`:** guion paso a paso para la demostración en vivo sobre `4. Presentacion.ipynb`.
