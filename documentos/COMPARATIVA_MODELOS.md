# Comparativa Técnica de Arquitecturas MLP

Documento actualizado con los resultados **reales** extraídos de los notebooks del repositorio (evaluación sobre `seg_test`, 3.000 imágenes nunca vistas durante el entrenamiento).

> Versiones analizadas: los 3 notebooks de la raíz (`1.` `2.` `3.`), `4. Presentacion.ipynb` y las 6 variantes de `scraps/`.

---

## 1. Resumen Ejecutivo de Métricas (Evaluación en Test)

| Variante | Estrategia de datos | lr | Épocas (máx) | Test Acc | Test Loss | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `scraps/notebookCorregidoMLP3Capas.ipynb` | 80/20 de seg_train | 0.0005 | 35 | 60.60% | 1.0438 | ~0.59 |
| `scraps/notebookCorregidoMLP2Capas.ipynb` | 80/20 de seg_train (2 capas) | 0.0005 | – | 61.27% | 1.0340 | ~0.60 |
| `scraps/notebookCorregidoMLP2Capas100train.ipynb` | 100% seg_train (2 capas) | 0.0005 | – | 60.83% | 1.0530 | ~0.60 |
| `scraps/notebookCorregidoMLP3Capas100trainpatience2.ipynb` | 100% seg_train | 0.0005 | 50 | 64.13% | 0.9510 | 0.63 |
| `scraps/notebookCorregidoMLP3Capas100train.ipynb` | 100% seg_train | 0.0005 | 50 | 64.23% | 0.9444 | 0.64 |
| `1. NotebookMLPLay3.ipynb` (@Gabriel1-du) | 100% seg_train | 0.0005 | 50 | 64.23% | 0.9444 | 0.64 |
| `2. notebookMLPPat2Lay3.ipynb` (@Gabriel1-du) | 100% seg_train | 0.0005 | 50 | 64.13% | 0.9510 | 0.63 |
| `4. Presentacion.ipynb` | 100% seg_train + seg_test como val/test | 0.0005 | 50 | 64.13% | 0.9510 | 0.63 |
| `scraps/notebookCorregidoMLP3Capas(FRANCISCO).ipynb` | 80/20 de seg_train | 0.0008 | 40 | **64.47%** | **0.9521** | **0.64** |
| `3. notebookCorregidoMLP3Capas(FRANCISCO) (1).ipynb` (@FatZuack) | 80/20 de seg_train | 0.0008 | 40 | **64.47%** | **0.9521** | **0.64** |

**Mejor modelo global:** `3. notebookCorregidoMLP3Capas(FRANCISCO) (1).ipynb` → **64.47%** de accuracy en test, **0.64** de Macro F1.

**Modelo de presentación elegido:** `4. Presentacion.ipynb` (100% de `seg_train` para entrenar, `seg_test` como validación/test) → **64.13%**, 0.63 Macro F1, curvas de train/val estables.

---

## 2. Desglose de Rendimiento por Clase (F1-Score)

### 2.1 Modelo ganador — `3. notebookCorregidoMLP3Capas(FRANCISCO) (1).ipynb`

| Clase | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
| buildings | 0.55 | 0.51 | 0.53 | 437 |
| forest | 0.79 | 0.80 | 0.79 | 474 |
| glacier | 0.65 | 0.66 | 0.66 | 553 |
| mountain | 0.60 | 0.71 | 0.65 | 525 |
| sea | 0.58 | 0.45 | 0.51 | 510 |
| street | 0.68 | 0.72 | 0.70 | 501 |
| **macro avg / accuracy** | **0.64** | **0.64** | **0.64** | 3000 |

### 2.2 Modelo de presentación — `4. Presentacion.ipynb` (100% train)

| Clase | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
| buildings | 0.56 | 0.44 | 0.49 | 437 |
| forest | 0.78 | 0.81 | 0.79 | 474 |
| glacier | 0.63 | 0.66 | 0.64 | 553 |
| mountain | 0.61 | 0.70 | 0.65 | 525 |
| sea | 0.61 | 0.48 | 0.54 | 510 |
| street | 0.65 | 0.74 | 0.69 | 501 |
| **macro avg / accuracy** | **0.64** | **0.64** | **0.63** | 3000 |

*Diagnóstico común:* `forest` es la clase más fuerte (F1 ~0.79) por su color dominante y homogéneo. Las confusiones persistentes son `glacier ↔ sea` (tonos azulados) y `street ↔ buildings` (geometrías rectas), inherentes al aplanado del MLP.

---

## 3. Comparación de Estrategias de Datos

### A. 100% de `seg_train` (Gabriel, notebooks 1/2/4 y scraps 100train)

* **Entrenamiento:** las 14.034 fotos de `seg_train` (439 lotes de 32).
* **Validación/Test:** las 3.000 fotos de `seg_test` se usan tanto de "ensayo" por época como de examen final.
* **Resultado:** 64.13–64.23%. Al aprovechar todo el `seg_train`, el modelo dispone del máximo de datos, pero la validación siempre es la misma carpeta.
* **Tiempo por época:** ~38–47 s en CPU (439 lotes/época).

### B. Split 80/20 de `seg_train` (Francisco, notebooks 3 y scraps)

* **Entrenamiento:** 11.228 fotos (80% de `seg_train`, 351 lotes).
* **Validación:** 2.806 fotos (20% de `seg_train`, "ensayo" honesto dentro de la misma distribución de train).
* **Test:** las 3.000 fotos de `seg_test`.
* **Resultado:** 64.47%, el más alto del proyecto, con 0.9521 de loss.
* **Tiempo por época:** ~26–31 s en CPU (351 lotes/época), el más veloz.

---

## 4. Principales Diferencias Técnicas Aplicadas

### A. Ubicación del Data Augmentation y Rendimiento de Hardware
* **@gabriel1-du** (100% seg_train): declaró las transformaciones (`RandomFlip`, `RandomRotation`, `RandomZoom`) como capas dentro del pipeline de datos (`dataset_train_prep.map(...)`), desacopladas del modelo.
* **@FatZuack** (80/20): mismo enfoque de pipeline con `tf.data` + `AUTOTUNE`.
* **Impacto:** el aumento de datos vive fuera del modelo en ambas versiones; esto permite precarga asíncrona y lotes listos antes de cada paso de gradiente (~2.5x más rápido que declarar las capas dentro del `Sequential`).

### B. Calibración del Dropout (Control de Regularización)
* Ambas versiones finales usaron la arquitectura **512 → 256 → 128** con `BatchNormalization` tras cada capa densa.
* Dropout progresivo: **30% → 25% → 20%** sobre 512 / 256 / 128 neuronas.
* Cambio clave frente al prototipo inicial (Dropout 50/40/30% y 2 capas): apagar el 50% de las 512 neuronas iniciales generaba pérdida prematura de información; bajar a 30% preservó las características del paisaje y elevó la accuracy ~64%.

### C. Hiperparámetros y Callbacks

| Parámetro | 100% seg_train (Gabriel) | 80/20 (Francisco) |
| :--- | :---: | :---: |
| Optimizador | Adam lr=0.0005 | Adam lr=0.0008 |
| Loss | sparse_categorical_crossentropy | sparse_categorical_crossentropy |
| Max épocas | 50 | 40 |
| EarlyStopping | vigilando val_loss (patience 7) | vigilando val_loss (patience 5) |
| ReduceLROnPlateau | factor 0.5, patience 2, min_lr 1e-5 | factor 0.5, patience 3, min_lr 1e-6 |
| Total params | **6,460,550** (trainable 6,458,758) | **6,460,550** (trainable 6,458,758) |

### D. Puntos de Consistencia Compartidos
* Entrada `64 × 64 × 3` (de 67.500 a 12.288 valores).
* Normalización `Rescaling(1/255)` a [0,1].
* Aumento de datos solo en entrenamiento (Flip horizontal, Rotación 0.05, Zoom 0.05).
* Batch de 32, `prefetch` + `AUTOTUNE`, carga por lotes para proteger RAM.

---

## 5. Conclusión Técnica

La estructura de 3 capas (512-256-128) con BatchNormalization y Dropout 30/25/20% es la arquitectura definitiva del proyecto. Las dos estrategias de datos son válidas y alcanzan los KPIs:

* 100% de `seg_train` (`4. Presentacion.ipynb`): **64.13%** de accuracy, 0.63 Macro F1, ~40 s/época.
* Split 80/20 (`3. ...FRANCISCO`): **64.47%** de accuracy, 0.64 Macro F1, ~29 s/época.

El modelo de `@FatZuack` logra el mejor accuracy y el menor tiempo por época; el de `@Gabriel1-du` aprovecha el 100% de los datos. Ambos quedan por debajo del techo de generalización propio de un MLP aplanado, por lo que el trabajo futuro apunta a una **CNN** (convolución + pooling) o *transfer learning* (ResNet/VGG) para superar el 85-90%.