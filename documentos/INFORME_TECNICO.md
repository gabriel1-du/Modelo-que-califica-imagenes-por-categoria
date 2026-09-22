# Informe Técnico — Clasificación de paisajes con MLP

> **Asignatura:** Técnicas Avanzadas de Machine Learning I (002D) — TLY1102
> **Entrega:** EP1 — Presentación de proyecto
> **Equipo:** Gabriel Durán · Martín Higuera · Francisco Salazar
> **Fecha:** 2026-09-22
> **Notebook principal:** [`4. Presentacion.ipynb`](../notebooks/4.%20Presentacion.ipynb)

---

## 1. Resumen ejecutivo

Se desarrolló un **Perceptrón Multicapa (MLP) feedforward** para clasificar imágenes de paisajes en **6 categorías** (buildings, forest, glacier, mountain, sea, street) sobre el dataset **Intel Image Classification**.

| Métrica | Valor |
| :--- | ---: |
| Exactitud en test (`seg_test`, 3.000 img.) | **62.37 %** |
| Loss en test | **1.0155** |
| Macro F1-score | **0.61** |
| Baseline aleatorio (1 de 6) | 16.6 % |
| KPI de la pauta (accuracy) | > 60 % → **cumplido** |
| Parámetros del modelo | 6.460.550 (~6,46 M) |
| Tiempo por época (CPU) | ~40 s |
| Épocas entrenadas | 24 de 50 (EarlyStopping, mejor peso en época 17) |

> ✅ **Corrida verificada:** estos valores provienen de la ejecución completa del notebook **con el particionado 80/20 y la semilla 42**. `seg_test` se evaluó **una sola vez**, al final.

**Variante propia del proyecto (diferenciación):** el mismo dataset se encuadra como **motor de recomendación turístico**: la categoría predicha se traduce en *tipo de destino* (glaciar → destinos fríos, mar → costeros). Se optimiza **macro-F1** en lugar de solo accuracy, y se exige que el modelo corra en **CPU con 64×64** como restricción de despliegue económico.

---

## 2. Descripción del problema de negocio

**Cliente potencial:** una empresa de **turismo y viajes** que vende paquetes a distintos tipos de destino (costa, montaña, nieve, ciudad, naturaleza).

**Problema:** el catálogo está organizado por *códigos y texto*, pero el cliente piensa en *imágenes*. Un usuario que sube una foto de un paisaje que le gusta no tiene forma de decirle al sistema *"quiero algo como esto"*. El equipo comercial no logra conectar lo que el cliente **quiere ver** con lo que la empresa **vende**.

**Solución propuesta:** un clasificador automático de imágenes que, a partir de una fotografía, determine a qué categoría de paisaje pertenece. Esa categoría se traduce en un **tipo de destino**, activando la recomendación del catálogo correspondiente.

**¿Por qué un modelo y no reglas manuales?** Porque las 6 categorías no se pueden separar con reglas fijas: `glacier` y `sea` comparten tonos azulados, `buildings` y `street` comparten grises urbanos. Se necesita un modelo que aprenda patrones de color, textura y forma desde los datos.

**Alcance:** clasificación de imágenes **en 6 categorías** (buildings, forest, glacier, mountain, sea, street), sin identificación de personas, lugares exactos ni datos sensibles.

---

## 3. Objetivos del proyecto

| # | Tipo | Objetivo | Cómo se mide |
| :-- | :--- | :--- | :--- |
| **O1** | **General** | Desarrollar un clasificador de imágenes basado en MLP que determine la categoría de paisaje de una fotografía, como primer paso del recomendador turístico. | Exactitud en test **> 60 %** |
| **O2** | **Específico** | Aplicar el ciclo completo de ciencia de datos (CRISP-DM): comprensión, exploración, preparación, modelado y evaluación. | Notebook con las 6 fases documentadas |
| **O3** | **Específico** | Garantizar la **reproducibilidad** del resultado mediante semilla y división de datos documentadas. | Semilla 42 + tabla de split |
| **O4** | **Específico** | Analizar críticamente los errores y las limitaciones del MLP frente a problemas de visión por computadora. | Matriz de confusión + análisis por clase |
| **O5** | **Específico** | Mantener el modelo dentro de una **restricción de costo**: debe correr en CPU sin GPU dedicada. | < 60 s por época en CPU |

**Variante propia del proyecto (diferenciación respecto a otros grupos):** el dataset es el mismo que usan otros equipos; la diferencia está en el **encuadre, la métrica y la restricción**:

1. **Encuadre de negocio:** clasificación de paisajes → *motor de recomendación turístico*.
2. **Métrica principal:** **macro-F1** (promedio de las 6 clases), no solo accuracy, porque en un recomendador fallar siempre la clase mayoritaria arruina la experiencia de usuario.
3. **Restricción de despliegue:** CPU con imágenes de **64×64**, priorizando el balance precisión/coste sobre la precisión máxima.

---

## 4. Definición de KPIs

Los KPIs traducen los objetivos en números verificables. Se definen **antes** de entrenar y se evalúan al final:

| KPI | Fórmula / medición | Meta | Por qué este valor |
| :--- | :--- | :---: | :--- |
| **Exactitud en test** | `accuracy` sobre `seg_test` | **> 60 %** | KPI de la pauta. Con 6 clases, el azar rinde 16.6 %, así que 60 % exige aprendizaje real. |
| **Macro F1-score** | Promedio del F1 de las 6 clases | **≥ 0.60** | Balancea precisión y recall **por clase**, evitando que una clase fácil infle el resultado. |
| **Pérdida en test** | `loss` sobre `seg_test` | **< 1.0000** | Mide cuán "seguro y acertado" es el modelo; complementa al accuracy. |
| **Brecha train/test** | \|acc_train − acc_test\| | **≤ 5 %** | Indicador de **sobreajuste**: si la brecha es grande, el modelo memorizó. |
| **Tiempo por época** | segundos reales de entrenamiento | **< 60 s** | Asegura la restricción de despliegue en CPU. |

> 📌 **Nota sobre el umbral de 60 %:** el objetivo declarado en la pauta es `accuracy > 60 %`. Complementariamente se monitorea que el **macro-F1 supere 0.60**, para que el cumplimiento no dependa de una sola clase.

---

## 5. Descripción de las fuentes de datos

**Fuente:** [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) — dataset público de escenarios naturales y urbanos.

| Atributo | Descripción |
| :--- | :--- |
| **Volumen total** | ~25.000 imágenes (17.034 utilizadas: train + test etiquetados) |
| **Resolución nativa** | 150 × 150 píxeles, 3 canales RGB (67.500 valores por imagen) |
| **Número de clases** | **6**: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street` |
| **Formato** | JPG, organizadas en **una subcarpeta por clase** (etiquetas automáticas) |
| **Estructura** | `seg_train/` (14.034 con etiquetas) · `seg_test/` (3.000 con etiquetas) · `seg_pred/` (7.301 sin etiquetas) |
| **Licencia / uso** | Dataset público con fines académicos; sin datos personales |

**Distribución por clase (train):**

| Clase | Imágenes | Clase | Imágenes |
| :--- | ---: | :--- | ---: |
| buildings | 2.191 | mountain | 2.512 |
| forest | 2.271 | sea | 2.274 |
| glacier | 2.404 | street | 2.382 |

**Calidad de los datos:** distribución **balanceada** (rango 2.191–2.512), por lo que **no se requiere balanceo** de clases. Se detectaron y limpiaron archivos con extensión no válida antes de entrenar (sección 6.1).

**Justificación de la selección:** cumple los criterios de la pauta (dataset propuesto), es variado (mezcla natural y urbano), está bien etiquetado, tiene un tamaño que permite recorrer el ciclo completo de ML sin GPU, y su dificultad real — clases con paletas de color solapadas — permite discutir de forma honesta las limitaciones del MLP.

---

## 6. Análisis exploratorio de los datos (EDA)

El EDA responde tres preguntas antes de modelar: *¿hay suficientes y están balanceados?*, *¿qué tamaño y formato tienen?* y *¿son visualmente distinguibles?*

### 6.1 Distribución y calidad

- **Conteo por clase:** 2.191–2.512 imágenes en train; 437–553 en test. **Sin desbalanceo relevante.**
- **Limpieza:** se recorrieron las carpetas descartando archivos con extensión distinta de `.jpg/.jpeg/.png` (y archivos sin extensión), sobre una **copia de respaldo** para no dañar la fuente original.
- **Rutas:** verificación de que `seg_train`, `seg_test` y `seg_pred` existen y son accesibles.

### 6.2 Resolución y canales

- Cargando **sin** `image_size` para ver el valor real: Keras devuelve lotes de `(32, 256, 256, 3)` por su redimensionado por defecto, pero la resolución **nativa** del dataset es **150×150×3 = 67.500 valores**.
- El dtype es `float32`, adecuado para el entrenamiento.

### 6.3 Patrones visuales y muestra de imágenes

Se generó una **grilla de 3 fotografías al azar por clase** (6×3) para verificar visualmente las etiquetas:

| Clase | Patrón dominante observado |
| :--- | :--- |
| `buildings` | Líneas rectas y rectángulos; predominancia de grises |
| `forest` | **Verde dominante** en múltiples tonalidades |
| `glacier` | Blancos, azules y celestes; formas curvas |
| `mountain` | Picos y triángulos; azul/gris/verde |
| `sea` | Azules en varias tonalidades, pocas figuras definidas |
| `street` | Espacios angostos, objetos urbanos, a veces personas |

**Dificultades anticipadas para la clasificación** (identificadas en el EDA y confirmadas después en la matriz de confusión):

1. `glacier` ↔ `sea`: ambas azuladas/blanquecinas.
2. `buildings` ↔ `street`: ambas grises con estructuras rectas.
3. Variabilidad **dentro** de cada clase (misma categoría con iluminación y encuadre muy distintos).

### 6.4 Variables relevantes

- **Variable objetivo:** la subcarpeta de origen = etiqueta (entero 0–5).
- **Features:** los 12.288 píxeles normalizados tras redimensionar a 64×64.
- **Variable de control:** la semilla (42), que fija el particionado.

> 📓 **Trazabilidad:** las visualizaciones, conteos y la grilla de muestras están en la **sección 4** del [`4. Presentacion.ipynb`](../notebooks/4.%20Presentacion.ipynb).

---

## 7. Metodología: CRISP-DM

| Fase | Qué se hizo | Sección del notebook |
| :--- | :--- | :--- |
| **1. Comprensión del negocio** | Objetivo: clasificador de paisajes para recomendador de destinos, corriendo en CPU. | 1 y 2 |
| **2. Comprensión de los datos** | Exploración: 17.034 imágenes, 6 clases, resolución nativa 150×150×3. | 3 y 4 |
| **3. Preparación de los datos** | Copia de respaldo, limpieza de corruptos, redimensión a 64×64, normalización 1/255, aumento de datos solo en train, lotes `tf.data`. | 5 |
| **4. Modelado** | MLP embudo 512→256→128, `Adam(0.0005)`, `sparse_categorical_crossentropy`, BatchNorm + Dropout. | 6 y 7 |
| **5. Evaluación** | Exactitud, loss, macro-F1, reporte por clase y matriz de confusión. | 8 |
| **6. Despliegue / documentación** | Conclusiones, limitaciones, bitácora de experimentos e informe. | 9 y 10 |

El ciclo **no es lineal**: el colapso a ~18 % del primer experimento devolvió el trabajo a las fases 3 y 4 (rediseño de preprocesamiento y arquitectura).

---

## 8. Dataset y división de datos (trazabilidad)

| Conjunto | Carpeta | Imágenes | Proporción | Uso |
| :--- | :--- | ---: | ---: | :--- |
| **Entrenamiento** | 80 % de `seg_train/seg_train` | 11.228 | 65.9 % | ajuste de pesos |
| **Validación** | 20 % de `seg_train/seg_train` | 2.806 | 16.5 % | monitoreo por época (`EarlyStopping`, `ReduceLROnPlateau`) |
| **Test (examen final)** | 100 % de `seg_test/seg_test` | 3.000 | 17.6 % | **una sola vez**, al final |
| **Total** | | **17.034** | 100 % | |

- **Semilla:** `42`, fijada con `tf.keras.utils.set_random_seed(42)` (sección 0 del notebook) y reutilizada en `validation_split=0.2, seed=42`, por lo que el corte es **determinista y reproducible**.
- **Distribución de clases:** balanceada — entre 2.191 (`buildings`) y 2.512 (`mountain`) imágenes en train. **No requiere balanceo.**

> ✅ **Diseño de la división:** `seg_test` **no participa del entrenamiento ni de la validación**. Como `EarlyStopping` y `ReduceLROnPlateau` deciden únicamente con el 20 % de `seg_train`, las métricas reportadas sobre `seg_test` son una estimación **sin sesgo** del comportamiento del modelo con fotografías nunca vistas.
>
> 📌 **Nota de evolución (trazabilidad):** en una primera versión del proyecto, `seg_test` cumplía doble función (validación *y* test), lo que dejaba las métricas ligeramente optimistas. Se identificó como limitación, se documentó, y **se corrigió** implementando el particionado 80/20 sobre `seg_train` con la semilla 42. El cambio está registrado en [`CAMBIOS_NOTEBOOK_PRESENTACION.md`](CAMBIOS_NOTEBOOK_PRESENTACION.md).

---

## 9. Preprocesamiento

| Decisión | Implementación | Justificación |
| :--- | :--- | :--- |
| Redimensión a **64×64** | `image_dataset_from_directory(image_size=(64,64))` | Reduce la entrada de 67.500 a 12.288 valores; menos memoria y menos sobreajuste; permite entrenar en CPU. |
| Normalización **1/255** | `Rescaling(1/255)` | Píxeles de [0,255] a [0,1]; la red con ReLU converge mejor con entradas acotadas. |
| **Aumento de datos** | `RandomFlip`, `RandomRotation`, `RandomZoom(0.05)` | *Solo en train*: genera variantes de las mismas fotos → más generalización sin contaminar la evaluación. |
| Procesamiento por lotes | `tf.data` + `AUTOTUNE` | Desacopla el aumento del grafo; mantiene ~40 s/época. |
| Etiquetas | Enteros 0–5 + `sparse_categorical_crossentropy` | Evita one-hot innecesario; las etiquetas ya son enteros. |

---

## 10. Arquitectura del modelo

```
Entrada 64×64×3
   │
Flatten(12.288)          ← convierte la matriz en vector (las Dense solo aceptan vectores)
   │
Dense(512, ReLU) → BatchNorm → Dropout(0.30)
   │
Dense(256, ReLU) → BatchNorm → Dropout(0.25)
   │
Dense(128, ReLU) → BatchNorm → Dropout(0.20)
   │
Dense(6, Softmax)        ← 6 neuronas = 6 clases; softmax → probabilidades que suman 1
```

**Por qué embudo 512 → 256 → 128:** la primera capa absorbe 12.288 valores sin procesar (necesita capacidad) y luego se comprime para forzar a la red a resumir. Eso reduce parámetros, baja el costo de cómputo en CPU y actúa como regularización implícita contra el sobreajuste.

**Optimización:** `Adam(learning_rate=0.0005)` con `ReduceLROnPlateau` (baja 5e-4 → 2.5e-4 → … → 1e-5 cuando la validación se estanca) y `EarlyStopping` que restaura los mejores pesos.

> 📌 **Learning rate inicial: 0.0005** — es el valor del código y el que confirman los logs (`5.0000e-04`). El decaimiento pasa intermedios como 0.0001, pero **ese no es el valor inicial**.

---

## 11. Resultados

### 11.1 KPIs

| Indicador | Meta | Resultado | Estado |
| :--- | :---: | :---: | :---: |
| Test accuracy | > 60 % | **62.37 %** | ✅ |
| Test loss | < 1.0000 | **1.0155** | ❌ |
| Macro F1 | ≥ 0.60 | **0.61** | ✅ |
| Brecha train/test | ≤ 5 % | **2.17 %** | ✅ |
| Tiempo por época | < 60 s | ~40 s | ✅ |

**4 de 5 KPIs cumplidos.** El único no cumplido es el **test loss (1.0155 vs meta < 1.0000)**, que queda a **1.6 %** de la meta: el modelo acierta bastante, pero cuando se equivoca lo hace con más confianza de la que debería — coherente con un MLP que no distingue `glacier` de `sea`. La **brecha train/test de 2.17 %** confirma que **no hay sobreajuste**.

### 11.2 F1 por clase

| Clase | Precision | Recall | F1 | Soporte |
| :--- | ---: | ---: | ---: | ---: |
| buildings | 0.57 | 0.38 | 0.45 | 437 |
| **forest** | 0.76 | 0.80 | **0.78** | 474 |
| glacier | 0.59 | 0.69 | 0.64 | 553 |
| mountain | 0.60 | 0.66 | 0.63 | 525 |
| sea | 0.56 | 0.41 | 0.47 | 510 |
| street | 0.63 | 0.77 | 0.69 | 501 |
| **Macro avg** | **0.62** | **0.62** | **0.61** | 3000 |

### 11.3 Análisis de error

**Confusiones principales (matriz de confusión, sección 8e):**

1. **`glacier` ↔ `sea`** — ambas son azuladas/blanquecinas (hielo y agua). Un MLP no extrae bordes ni texturas, así que separarlas por color le cuesta.
2. **`buildings` ↔ `street`** — ambas grises, con estructuras rectas urbanas.

**Mejor clase: `forest` (F1 = 0.78, recall 0.80)** — es casi verde puro, un color dominante que casi no se confunde con nada. Le sigue `street` (F1 = 0.69).

**Peores: `buildings` (F1 = 0.45, recall 0.38) y `sea` (F1 = 0.47, recall 0.41)** — esos recalls tan bajos indican que el modelo **pierde** esas fotos y las reasigna a sus vecinas visuales: `sea` → `glacier` y `buildings` → `street`.

**Lectura honesta del accuracy:** 62.37 % es **~3.8 veces el baseline aleatorio (16.6 %)**, y queda **2.37 puntos sobre el KPI de 60 %** de la pauta. Hay espacio de mejora, y ese es justamente lo que motiva el análisis de error.

**Asimetría observable:** `glacier` tiene precisión baja (0.59) pero recall alto (0.69): **absorbe** fotos que en realidad son `sea`. Igual hace `street` (precisión 0.63, recall 0.77) con las de `buildings`. No es un error aleatorio: es el modelo confundiendo **pares de clases con la misma paleta de color**.

---

## 12. Problemas encontrados y decisiones

| Problema | Decisión tomada |
| :--- | :--- |
| Primer intento con entrada 256×256 **colapsaba** (~18 % de accuracy, predecía siempre la misma clase) | Reducir entrada a 64×64, añadir `BatchNormalization` tras cada capa oculta y dropout 30/25/20 %. |
| Entrenamiento **lento en CPU** | Tres capas ocultas en embudo, entrada 64×64 y pipeline `tf.data` con `AUTOTUNE`. |
| Tasa de aprendizaje fija **se estancaba** | `ReduceLROnPlateau` + `EarlyStopping` con restauración de mejores pesos. |
| Riesgo de **sobreajuste** | Aumento de datos solo en train, dropout graduado y BatchNorm. |

---

## 13. Limitaciones del MLP y trabajo futuro

1. **El techo del MLP:** `Flatten` destruye la estructura espacial 2D. El MLP clasifica por intensidades globales y firmas de color, lo que impone un techo práctico en torno al **~62 %**.
2. **Errores sistemáticos:** las confusiones `glacier`/`sea` y `buildings`/`street` son consecuencia directa de esa ceguera espacial.
3. **Trabajo futuro:** migrar a una **CNN**, que preserva la estructura espacial con filtros locales compartidos y pooling, otorgando invariancia a la traslación. Se estima que superaría el 85 %.

---

## 14. Evaluación de impacto ético

**Riesgos identificados**

- **Sesgo geográfico/cultural:** el dataset fue recopilado mayoritariamente en contextos europeos/norteños. Un "bosque" o una "calle" de otra región puede no parecerse a los ejemplos de entrenamiento → el rendimiento bajará en otras geografías. Es un sesgo de muestreo, no deliberado, pero debe declararse.
- **Riesgo de uso indebido:** un clasificador de paisajes podría usarse para *inferir ubicación*. El modelo predice **categoría**, no lugar exacto; igualmente no debe usarse para rastrear personas.
- **Expectativa exagerada:** con ~62 % de exactitud, **2 de cada 5 fotos se clasifica mal**. Presentarlo como "IA que entiende escenas" sería deshonesto.

**Mitigaciones aplicadas**

- Se declara explícitamente el **límite de rendimiento** y las confusiones conocidas.
- Dataset público con fines **académicos**; sin datos personales ni personas identificables como objetivo.
- Aumento de datos **solo en train** para no falsear la evaluación.
- **Semilla y split documentados** para que cualquier revisor verifique los resultados.

**Conclusión ética:** impacto positivo principal **educativo y de eficiencia**. Riesgos **bajos-medios**, identificados y documentados. No hay uso en decisiones de alto impacto (salud, crédito, empleo, justicia).

---

## 15. Trazabilidad y reproducibilidad

| Elemento | Dónde está documentado |
| :--- | :--- |
| **Semilla 42** | `4. Presentacion.ipynb`, sección 0 |
| **Split de datos** (80/20 de `seg_train` + `seg_test` final) | `4. Presentacion.ipynb`, sección 5 |
| **KPIs y métricas** | `4. Presentacion.ipynb`, secciones 7 y 8 |
| **Bitácora de experimentos** | [`documentos/EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) |

**Cómo reproducir:**

```bash
# 1. Clonar el repositorio completo (incluye dataset/)
git clone <url-del-repositorio>
cd Modelo-que-califica-imagenes-por-categoria

# 2. Abrir "notebooks/4. Presentacion.ipynb"
# 3. Kernel → Restart & Run All
# 4. Verificar: la sección 0 imprime "Semilla fijada en 42: resultados reproducibles."
# 5. Esperar: accuracy ≈ 62,37 % / macro-F1 ≈ 0,61 / loss ≈ 1,0155
#    (24 épocas con EarlyStopping; sección 0 debe imprimir "Semilla fijada en 42")
```

> ✅ **Nota:** los valores de este informe corresponden a la corrida **ya ejecutada** con el particionado 80/20 y la semilla 42. `seg_test` se evaluó una sola vez, al final.

---

## 16. Estructura de entregables

```
Modelo-que-califica-imagenes-por-categoria/
├── README.md                        ← informe técnico (secciones obligatorias 1–6)
├── notebooks/
│   ├── 4. Presentacion.ipynb        ← notebook reproducible (secciones a–h)
│   └── 1./2./3.                     ← experimentos previos (opcionales)
├── dataset/
│   └── Intel Image Classification/
│       ├── seg_train/               ← 14.034 imágenes con etiquetas
│       ├── seg_test/                ← 3.000 imágenes con etiquetas
│       └── seg_pred/                ← 7.301 imágenes sin etiquetas
├── documentos/
│   ├── INFORME_TECNICO.md           ← este informe (versión extendida)
│   ├── EXPERIMENT_LOG.md
│   └── GUION_* / CAMBIOS_*          ← guiones internos de defensa
├── models/                          ← pesos guardados (.keras)
└── images/                          ← gráficos generados
```

## 17. Conclusiones y trabajo futuro

**Desempeño alcanzado:** 62,37 % de exactitud y macro-F1 0,61 sobre `seg_test` (baseline aleatorio 16,6 %), con una brecha train/test de solo 2,17 %. Se cumplen **4 de los 5 KPIs** declarados; no se cumple `loss < 1.0000` (1,0155), a un 1,6 % de la meta.

**Decisiones de diseño que sostienen el resultado:** partición honesta (validación separada del test final, semilla 42), `BatchNormalization` + `Dropout` graduado en las tres capas ocultas, `EarlyStopping` con paciencia 7 y restauración del mejor peso, y `class_weight='balanced'` para no favorecer clases mayoritarias.

**Fortalezas:** entrenamiento estable en CPU (~40 s/época), sin sobreajuste evidente, y un análisis de errores que identifica de forma consistente los pares de clases confundidos (`glacier`↔`sea`, `buildings`↔`street`).

**Limitaciones:** `Flatten` elimina la estructura espacial 2D, lo que techo el modelo en ~62 % y explica los recalls bajos de `buildings` (0,38) y `sea` (0,41).

**Trabajo futuro:** migrar a una **CNN**, que preserva la estructura espacial con filtros locales compartidos y *pooling*, otorgando invariancia a la traslación; se estima superaría el 85 %.
