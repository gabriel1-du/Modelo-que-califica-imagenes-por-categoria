# Clasificación de Paisajes con MLP — Informe técnico

> **Asignatura:** Técnicas Avanzadas de Machine Learning I (002D) — TLY1102
> **Entrega:** EP1 — Presentación de proyecto
> **Equipo:** Gabriel Durán · Martín Higuera · Francisco Salazar
> **Fecha:** 2026-09-22
> **Notebook principal:** [`4. Presentacion.ipynb`](notebooks/4.%20Presentacion.ipynb) · **Versión extendida del informe:** [`documentos/INFORME_TECNICO.md`](documentos/INFORME_TECNICO.md)

Se entrenó un **Perceptrón Multicapa (MLP) feedforward** que clasifica imágenes de paisajes en **6 categorías** sobre el dataset **Intel Image Classification**.

| Métrica | Valor |
| :--- | ---: |
| Exactitud en test (`seg_test`, 3.000 img.) | **62,37 %** |
| Loss en test | **1,0155** |
| Macro F1-score | **0,61** |
| Baseline aleatorio (1 de 6) | 16,6 % |
| KPI de la pauta (accuracy > 60 %) | **cumplido** |
| Brecha train/test | **2,17 %** |
| Parámetros | 6.460.550 |
| Tiempo por época (CPU) | ~40 s |
| Épocas entrenadas | 24 de 50 (EarlyStopping, mejor peso en la 17) |

**Reproducción:** abrir `notebooks/4. Presentacion.ipynb` → *Kernel → Restart & Run All*. La sección 0 imprime `Semilla fijada en 42: resultados reproducibles.` No se requiere GPU.

---

## 1. Descripción del problema de negocio

**Cliente potencial:** una empresa de **turismo y viajes** que vende paquetes a distintos tipos de destino (costa, montaña, nieve, ciudad, naturaleza).

**Problema:** cuando un usuario sube una foto, la empresa no sabe **qué tipo de destino** le interesa y le muestra un catálogo genérico. El costo de negocio es la **pérdida de conversión**: recomendaciones fuera de lugar.

**Solución propuesta:** un clasificador que, dada una foto, prediga la categoría de paisaje → *motor de recomendación turístico*. La categoría se traduce en tipo de destino (glaciar → destinos fríos, mar → costeros, edificios → urbano).

**Por qué es un problema real de ML:** hay ~25.000 fotos/día y clasificarlas a mano es inviable. El modelo debe correr en **CPU**, sin GPU, para no encarecer el despliegue.

**Variante propia del proyecto (diferenciación):** además del encuadre de recomendador, se optimiza **macro-F1** en lugar de solo accuracy — en un recomendador, fallar siempre sobre la clase mayoritaria arruina la experiencia de usuario igual que fallar mucho. Y se fija **64×64 en CPU** como restricción de despliegue económico.

---

## 2. Objetivos del proyecto

| # | Objetivo | Criterio de cumplimiento |
| :-- | :--- | :--- |
| **O1** | Clasificar las 6 categorías de paisaje con un MLP denso puro (sin capas convolucionales) | Modelo entrenado y evaluado sobre `seg_test` |
| **O2** | superar el baseline aleatorio (16,6 %) y el objetivo de la pauta (> 60 %) | Exactitud en test > 60 % |
| **O3** | Mantener la evaluación **honesta**: validación separada del examen final, con semilla fija | `seg_test` evaluado una sola vez; semilla 42 documentada |
| **O4** | Garantizar reproducibilidad y trazabilidad de todas las decisiones | Notebook ejecutable de principio a fin + este informe |
| **O5** | Analizar los errores y declarar las limitaciones del MLP para imágenes | Matriz de confusión, ejemplos mal clasificados y sección de ética |

---

## 3. Definición de KPIs que resolverán el problema de negocio

Los KPIs se definen **antes** de entrenar y se evalúan al final:

| KPI | Fórmula / medición | Meta | Resultado | Estado |
| :--- | :--- | ---: | ---: | :-- |
| **Exactitud en test** | `accuracy` sobre `seg_test` | **> 60 %** | **62,37 %** | ✅ |
| **Brecha train/test** | \|acc_train − acc_test\| | **≤ 5 %** | **2,17 %** (60,20 % vs 62,37 %) | ✅ |
| **Macro F1-score** | Promedio del F1 de las 6 clases | **≥ 0,60** | **0,61** | ✅ |
| **Tiempo por época** | segundos reales de entrenamiento | **< 60 s** | **~40 s** | ✅ |
| **Pérdida en test** | `loss` sobre `seg_test` | **< 1,0000** | **1,0155** | ❌ |

**4 de 5 KPIs cumplidos.**

> 📌 **Por qué este valor de 60 %:** con 6 clases el azar rinde 16,6 %, así que 60 % exige aprendizaje real. El macro-F1 se declara como métrica principal porque evita que una clase fácil infle el accuracy global.
>
> ⚠️ **KPI no cumplido:** `loss` quedó en **1,0155**, a un **1,6 % por encima** de la meta de 1,0000. El modelo acierta más de lo que la pauta exige, pero **no con la confianza** que exigía ese umbral. Se reporta tal cual; no se ajusta la meta *a posteriori*.

---

## 4. Descripción de las fuentes de datos utilizadas

**Fuente:** [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) — dataset público de escenarios naturales y urbanos.

| Atributo | Descripción |
| :--- | :--- |
| **Volumen total** | ~25.000 imágenes (17.034 utilizadas: train + test etiquetados) |
| **Resolución nativa** | 150 × 150 píxeles, 3 canales RGB (67.500 valores por imagen) |
| **Número de clases** | **6**: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street` |
| **Formato** | JPG, organizadas en **una subcarpeta por clase** (la etiqueta se infiere del nombre de carpeta) |
| **Estructura** | `seg_train/` (14.034 con etiquetas) · `seg_test/` (3.000 con etiquetas) · `seg_pred/` (7.301 sin etiquetas) |
| **Licencia / uso** | Dataset público con fines académicos; **sin datos personales** |

**Justificación de la selección:** es el dataset propuesto por la pauta, mezcla escenarios naturales y urbanos, está bien etiquetado, tiene un tamaño que permite recorrer el ciclo completo de ML **sin GPU**, y su dificultad real — clases con paletas de color solapadas — permite discutir de forma honesta las limitaciones del MLP.

### División de datos (trazabilidad)

| Conjunto | Origen | Imágenes | Proporción | Lotes | Uso |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Entrenamiento** | 80 % de `seg_train/` | **11.228** | 65,9 % | 351 | ajuste de pesos |
| **Validación** | 20 % de `seg_train/` | **2.806** | 16,5 % | 88 | monitoreo por época |
| **Test (examen final)** | 100 % de `seg_test/` | **3.000** | 17,6 % | 94 | **una sola vez**, al final |

> 🔑 **Semilla:** `tf.keras.utils.set_random_seed(42)` (sección 0 del notebook). Fija tanto los pesos iniciales como el particionado 80/20, de modo que **cualquier ejecución reproduce exactamente estas tres particiones**.

---

## 5. Preparación y análisis exploratorio de los datos (EDA)

### 5.1 Distribución y calidad

| Clase | Imágenes (train) | Clase | Imágenes (train) |
| :--- | ---: | :--- | ---: |
| buildings | 2.191 | mountain | 2.512 |
| forest | 2.271 | sea | 2.274 |
| glacier | 2.404 | street | 2.382 |

- **Balance:** rango 2.191–2.512 → **sin desbalanceo relevante**, no se requiere balanceo de clases.
- **Limpieza:** se recorrieron las carpetas descartando archivos con extensión distinta de `.jpg/.jpeg/.png` (0 archivos inválidos detectados), sobre una **copia de respaldo** para no dañar la fuente original.
- **Rutas:** verificación de que `seg_train`, `seg_test` y `seg_pred` existen y son accesibles.

### 5.2 Resolución y canales

- La resolución **nativa** es **150×150×3 = 67.500 valores** por imagen. El `dtype` es `float32`, adecuado para el entrenamiento.
- Se redimensiona a **64×64×3 = 12.288 valores**: −81,8 % de dimensionalidad, lo que mitiga la maldición de la dimensionalidad y permite entrenar en CPU.

### 5.3 Patrones visuales y muestra de imágenes

Se generó una **grilla de 3 fotografías al azar por clase (6×3)** para verificar visualmente las etiquetas:

| Clase | Patrón dominante observado |
| :--- | :--- |
| `buildings` | Líneas rectas y rectángulos; predominancia de grises |
| `forest` | **Verde dominante** en múltiples tonalidades |
| `glacier` | Blancos, azules y celestes; formas curvas |
| `mountain` | Picos y triángulos; azul/gris/verde |
| `sea` | Azules en varias tonalidades, pocas figuras definidas |
| `street` | Espacios angostos, objetos urbanos, a veces personas |

**Dificultades anticipadas en el EDA** (confirmadas después en la matriz de confusión):

1. `glacier` ↔ `sea`: ambas azuladas/blanquecinas.
2. `buildings` ↔ `street`: ambas grises con estructuras rectas.
3. Variabilidad **dentro** de cada clase (misma categoría con iluminación y encuadre muy distintos).

### 5.4 Variables relevantes

- **Variable objetivo:** la subcarpeta de origen = etiqueta (entero 0–5).
- **Features:** los 12.288 píxeles normalizados tras redimensionar a 64×64.
- **Variable de control:** la semilla (42), que fija el particionado.

### 5.5 Preprocesamiento aplicado

| Paso | Implementación | Justificación |
| :--- | :--- | :--- |
| Copia de respaldo | `shutil.copytree` a `copy/` | No mutar el dataset original |
| Redimensionamiento | `image_size=(64,64)` | −81,8 % dimensionalidad; habilita CPU |
| Normalización | `Rescaling(1/255)` | Rango [0,1] → acelera y estabiliza Adam |
| Etiquetado | `class_names` orden alfabético | 0→buildings … 5→street, determinista |
| Aumento de datos | `RandomFlip` + `RandomRotation` + `RandomZoom` | Regularización; **solo en train** para no falsear la evaluación |
| Lotes | `batch_size=32` con `tf.data` + `AUTOTUNE` | Protege la RAM; pipeline asíncrono |

> 📓 Trazabilidad completa en la **sección 5** del notebook.

---

## 6. Metodología utilizada (CRISP-DM)

| Fase | Qué se hizo | Sección del notebook |
| :--- | :--- | :--- |
| **1. Comprensión del negocio** | Objetivo: clasificador de paisajes para recomendador de destinos, corriendo en CPU. Definición de KPIs. | 1 y 2 |
| **2. Comprensión de los datos** | Exploración: 17.034 imágenes, 6 clases, resolución nativa 150×150×3, distribución balanceada, grilla de muestras. | 3 y 4 |
| **3. Preparación de los datos** | Copia de respaldo, limpieza de inválidos, redimensión a 64×64, normalización 1/255, aumento de datos solo en train, lotes `tf.data`. | 5 |
| **4. Modelado** | MLP embudo 512→256→128, `Adam(0.0005)`, `sparse_categorical_crossentropy`, BatchNorm + Dropout. | 6 y 7 |
| **5. Evaluación** | Exactitud, loss, macro-F1, reporte por clase, matriz de confusión y ejemplos mal clasificados. | 8 |
| **6. Despliegue / documentación** | Conclusiones, limitaciones, impacto ético, bitácora de experimentos e informe. | 9 y 10 |

El ciclo **no es lineal**: el colapso a ~18 % del primer experimento devolvió el trabajo a las fases 3 y 4 (rediseño de preprocesamiento y arquitectura). Detalle en [`documentos/EXPERIMENT_LOG.md`](documentos/EXPERIMENT_LOG.md).

---

## 7. Diseño del MLP

**Canalización de uso:**

1. **Entrada (`Flatten`):** imágenes 64×64 RGB → vector de **12.288** características.
2. **Capa oculta 1:** `Dense(512, ReLU)` → `BatchNormalization` → `Dropout(0.30)`.
3. **Capa oculta 2:** `Dense(256, ReLU)` → `BatchNormalization` → `Dropout(0.25)`.
4. **Capa oculta 3:** `Dense(128, ReLU)` → `BatchNormalization` → `Dropout(0.20)`.
5. **Salida:** `Dense(6, Softmax)` — una neurona por clase.

**Hiperparámetros y sus razones:**

| Hiperparámetro | Valor | Por qué |
| :--- | :--- | :--- |
| Optimizador | `Adam(lr = 0.0005)` | Convergencia rápida con paso conservador; menor que el default 0,001 para no saltarse el mínimo en batches ruidosos |
| Función de pérdida | `sparse_categorical_crossentropy` | Etiquetas como enteros 0–5 (no one-hot) |
| Épocas máximas | 50 | Tope de seguridad; el corte lo decide `EarlyStopping` |
| Batch size | 32 | Balance entre estabilidad del gradiente y velocidad en CPU |
| `EarlyStopping` | `patience=7`, `restore_best_weights=True` | Corta al estancarse y **devuelve los mejores pesos**, no los últimos |
| `ReduceLROnPlateau` | `factor=0.5`, `patience=2` | Refina el paso cuando la meseta se estanca |

**Parámetros totales: 6.460.550** (~6,46 M) — 6.291.968 en la primera capa, 131.328 en la segunda, 32.896 en la tercera y 774 en la salida.

> Justificación detallada de cada decisión en la **sección 6** del notebook y en [`documentos/INFORME_TECNICO.md`](documentos/INFORME_TECNICO.md).

---

## 8. Resultados y análisis de errores

### 8.1 Entrenamiento y validación

- **24 de 50 épocas**; `EarlyStopping` cortó en la **época 17** (mejor `val_loss` 0,8012 / `val_accuracy` 0,7124) y restauró esos pesos.
- **Sin sobreajuste evidente:** brecha train/test de solo **2,17 %** (60,20 % vs 62,37 %).
- **Convergencia estable:** la curva de `val_loss` baja de forma sostenida hasta estancarse; no hay rebote que indique memorización.

### 8.2 Evaluación sobre `seg_test` (3.000 imágenes)

| Métrica | Valor |
| :--- | ---: |
| Accuracy | **0,6237** |
| Loss | **1,0155** |
| Macro F1 | **0,61** |
| Weighted F1 | **0,61** |

### 8.3 Reporte por clase (Precision · Recall · F1)

| Clase | Precision | Recall | F1 | Soporte |
| :--- | ---: | ---: | ---: | ---: |
| buildings | 0,57 | 0,38 | 0,45 | 437 |
| forest | 0,76 | **0,80** | **0,78** | 474 |
| glacier | 0,59 | 0,69 | 0,64 | 553 |
| mountain | 0,60 | 0,66 | 0,63 | 525 |
| sea | 0,56 | **0,41** | **0,47** | 510 |
| street | 0,63 | 0,77 | 0,69 | 501 |
| **macro avg** | **0,62** | **0,62** | **0,61** | 3000 |

**Lectura:**

- **Mejor clase: `forest` (F1 = 0,78).** Es casi verde puro, un color dominante que casi no se confunde con nada.
- **Peores: `buildings` (F1 = 0,45, recall 0,38) y `sea` (F1 = 0,47, recall 0,41).** Esos recalls tan bajos indican que el modelo **pierde** esas fotos y las reasigna a sus vecinas visuales: `sea` → `glacier`, `buildings` → `street`.
- **Asimetría observable:** `glacier` tiene precisión baja (0,59) pero recall alto (0,69): **absorbe** fotos que en realidad eran `sea`. Igual hace `street` (precisión 0,63, recall 0,77) con las de `buildings`. No es un error aleatorio: es el modelo confundiendo **pares de clases con la misma paleta de color**.

### 8.4 Matriz de confusión

**Confusiones principales:** `glacier`↔`sea` y `buildings`↔`street` — exactamente las dos anticipadas en el EDA.

### 8.5 Causa raíz: limitaciones del MLP

`Flatten` **destruye la estructura espacial 2D**: el MLP ve 12.288 números sueltos, sin saber qué píxeles son vecinos. No puede distinguir una costa de un glaciar por su **forma**, solo por su **firma de color**. De ahí el techo práctico de ~62 % y los recalls bajos de `sea` y `buildings`.

---

## 9. Impacto ético

**Riesgos identificados**

- **Sesgo geográfico/cultural:** el dataset se recopiló mayoritariamente en contextos europeos/norteños. Un "bosque" o una "calle" de otra región puede no parecerse a los ejemplos de entrenamiento → el rendimiento bajará en otras geografías. Es un sesgo de muestreo, no deliberado, pero debe declararse.
- **Riesgo de uso indebido:** un clasificador de paisajes podría usarse para *inferir ubicación*. El modelo predice **categoría**, no lugar exacto; igualmente no debe usarse para rastrear personas.
- **Expectativa exagerada:** con ~62 % de exactitud, **2 de cada 5 fotos se clasifica mal**. Presentarlo como "IA que entiende escenas" sería deshonesto.

**Mitigaciones aplicadas**

- Se declara explícitamente el **límite de rendimiento** y las confusiones conocidas.
- Dataset público con fines **académicos**; sin datos personales ni personas identificables como objetivo.
- Aumento de datos **solo en train** para no falsear la evaluación.
- **Semilla y split documentados** para que cualquier revisor verifique los resultados.

**Conclusión ética:** impacto positivo principal **educativo y de eficiencia**. Riesgos **bajos-medios**, identificados y documentados. No hay uso en decisiones de alto impacto (salud, crédito, empleo, justicia).

---

## 10. Conclusiones

- **Desempeño alcanzado:** 62,37 % de exactitud y macro-F1 0,61 sobre `seg_test` (baseline aleatorio 16,6 %), con brecha train/test de solo 2,17 %. **4 de 5 KPIs cumplidos.**
- **Decisiones de diseño que sostienen el resultado:** partición honesta (validación separada del test final, semilla 42), `BatchNorm` + `Dropout` graduado en las tres capas ocultas, `EarlyStopping` con paciencia 7 y restauración del mejor peso, y aumento de datos solo en train.
- **Fortalezas:** entrenamiento estable en CPU (~40 s/época), sin sobreajuste evidente, y un análisis de errores que identifica de forma consistente los pares de clases confundidos.
- **Limitaciones:** `Flatten` elimina la estructura espacial 2D, lo que techo el modelo en ~62 % y explica los recalls bajos de `buildings` (0,38) y `sea` (0,41).
- **Trabajo futuro:** migrar a una **CNN**, que preserva la estructura espacial con filtros locales compartidos y *pooling*; se estima que superaría el 85 %.

---

## 11. Comparación con otras configuraciones del equipo

Se entrenaron varias variantes con la misma arquitectura para comparar protocolos de evaluación:

| Notebook | Estrategia de datos | lr | Épocas | Test Acc | Test Loss | Macro F1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `1. NotebookMLPLay3.ipynb` (@Gabriel1-du) | 100 % `seg_train` · **val = `seg_test`** | 0,0005 | 50 | 64,23 % | 0,9444 | 0,64 |
| `2. notebookMLPPat2Lay3.ipynb` (@Gabriel1-du) | 100 % `seg_train` · **val = `seg_test`** | 0,0005 | 50 | 64,13 % | 0,9510 | 0,63 |
| `3. notebookCorregidoMLP3Capas(FRANCISCO).ipynb` (@FatZuack) | 80/20 de `seg_train` | 0,0008 | 40 | **64,47 %** | 0,9521 | **0,64** |
| **`4. Presentacion.ipynb` (entregable)** | **80/20 de `seg_train` + `seg_test` ciego** | **0,0005** | **24** | **62,37 %** | 1,0155 | **0,61** |

**Por qué el entregable reporta un número *menor*:**

Los notebooks 1 y 2 usan `seg_test` **como validación y como test** → `EarlyStopping` y `ReduceLROnPlateau` toman decisiones mirando el examen, y la métrica queda **inflada**. Su valor correcto es solo una cota superior.

`4. Presentacion.ipynb` es el **protocolo más estricto** del proyecto: la validación es un 20 % de `seg_train`, `seg_test` se evalúa **una sola vez**, y todo queda fijado con **semilla 42**. Entrena además con el 80 % (no el 100 %) de `seg_train`. Por eso su exactitud es menor: **es la estimación sin sesgo del desempeño real**.

> **Nota histórica:** la versión anterior de `4. Presentacion.ipynb` — entrenando con el 100 % de `seg_train` y usando `seg_test` como validación **y** test — alcanzó **64,13 %**. Tras corregir el particionado, bajó a **62,37 %** (−1,76 puntos). La caída no es un retroceso técnico: es la **evidencia empírica de que el número anterior estaba inflado** por la validación contaminada y por entrenar con más datos. Se prefiere defender un número **menor pero honesto**.

---

## 12. Estructura del repositorio

```
Modelo-que-califica-imagenes-por-categoria/
├── README.md                        ← este informe
├── notebooks/
│   ├── 4. Presentacion.ipynb        ← notebook reproducible (secciones a–h)
│   └── 1./2./3.                     ← experimentos previos (opcionales)
├── dataset/
│   └── Intel Image Classification/
│       ├── seg_train/               ← 14.034 imágenes con etiquetas
│       ├── seg_test/                ← 3.000 imágenes con etiquetas
│       └── seg_pred/                ← 7.301 imágenes sin etiquetas
├── documentos/
│   ├── INFORME_TECNICO.md           ← informe extendido (17 secciones)
│   ├── EXPERIMENT_LOG.md            ← bitácora de experimentos
│   └── GUION_* / CAMBIOS_*          ← guiones internos de defensa
├── models/                          ← pesos guardados (.keras)
└── images/                          ← gráficos generados
```

**Documentación adicional:**

- [`documentos/INFORME_TECNICO.md`](documentos/INFORME_TECNICO.md) — versión extendida con derivación completa de cada decisión.
- [`documentos/EXPERIMENT_LOG.md`](documentos/EXPERIMENT_LOG.md) — EXP-01 a EXP-04, errores y lecciones aprendidas.

---

## 13. Colaboradores

| Usuario | Rol |
| --- | --- |
| **@FatZuack** (Francisco Salazar) | Autor del notebook `3. notebookCorregidoMLP3Capas(FRANCISCO) (1).ipynb` |
| **@Gabriel1-du** (Gabriel Duran) | Autor de los notebooks `1. NotebookMLPLay3.ipynb` y `2. notebookMLPPat2Lay3.ipynb` |
| **Martín Higuera** | Autor de `4. Presentacion.ipynb`, documentación y guion de presentación |
