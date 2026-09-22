# Guion de Presentación — Live Coding con `4. Presentacion.ipynb`

**Asignatura:** Técnicas avanzadas de machine learning - 002D
**Equipo:** Martín Higuera · Gabriel Durán · Francisco Salazar

Formato de la presentación: **código en vivo**. Se corre `4. Presentacion.ipynb` en orden mientras se explica; las celdas marcan el ritmo y cada sección cierra con la métrica o la figura clave. Duración estimada: **~12–15 min**.

> Referencias: la numeración de "celda" corresponde a las celdas del notebook `4. Presentacion.ipynb` (**88 celdas**; la sección 11 guarda y carga el modelo). Los elementos visuales que se mencionan se pueden capturar del propio output del notebook o pegar en las diapositivas del docx.

---

## 🔢 NÚMEROS CLAVE DE ESTA CORRIDA (memorizar)

| Concepto | Valor |
| :--- | ---: |
| Baseline aleatorio (1 de 6) | **16.6 %** |
| KPI de la pauta (accuracy) | **> 60 %** |
| **Accuracy en test** | **62.37 %** ✅ |
| **Macro F1** | **0.61** ✅ |
| **Test loss** | **1.0155** ❌ (meta < 1.0000) |
| **Brecha train/test** | **2.17 %** ✅ (60.20 % vs 62.37 %) |
| Tiempo por época | **~40 s** ✅ |
| Semilla | **42** |
| Learning rate inicial | **0.0005** |
| Split | **11.228 / 2.806 / 3.000** |
| Épocas | **24 de 50** (EarlyStopping; mejores pesos en la 17) |
| Mejor clase | `forest` **F1 0.78** |
| Peores clases | `buildings` **0.45** · `sea` **0.47** |

**Resultado:** **4 de 5 KPIs cumplidos.**

---

## Bloque 0 — Portada, semilla e intro (`~1 min`)

- **Celda 0:** título e integrantes.
- **Celda 1–2 (correr):** `tf.keras.utils.set_random_seed(42)`.
  - **Decir:** *"Fijamos la semilla 42: pesos iniciales, orden de datos y Dropout quedan determinados. Es lo que permite que cualquier revisor reproduzca estos números."*
- **Celda 3:** sección 1 — variante propia del problema (recomendación turístico, macro-F1 como métrica principal, restricción CPU/64×64).
- **Recorrer:** el proyecto entero = EDA → preprocesamiento → arquitectura → entrenamiento → evaluación de errores.

---

## Bloque 1 — Problema de negocio y KPIs (`~1.5 min`)

- **Celdas 4–7.** Contexto, importancia y **definición de KPIs (celda 7)**.
- **Puntos a reforzar:**
  - Fotos 150×150×3 = 67.500 números → se bajarán a 64×64 (12.288) por viabilidad en CPU/RAM.
  - Usos: turismo, medio ambiente, redes sociales, ciudades inteligentes.
  - **KPIs (5, fijados ANTES de entrenar):** accuracy > 60 % · |acc_train − acc_test| ≤ 5 % · macro F1 ≥ 0.60 · **test loss < 1.0000** · tiempo por época < 60 s.

---

## Bloque 2 — Dataset (`~1 min`)

- **Celdas 8–13.** Selección y descripción (Intel Image Classification, 6 clases, ~25.000 imágenes).
- **Dato clave:** 14.034 (train) / 3.000 (test) / 7.301 (pred sin etiqueta).
- **Aclarar ya aquí:** de las 14.034 hacemos **80/20** → 11.228 train / 2.806 val; las 3.000 de `seg_test` quedan como examen final.

---

## Bloque 3 — EDA (`~1.5 min`)

- **Celda 16 (correr):** conteo por clase → **celda 17** el resultado.
  - Clases balanceadas (~2.200–2.500 por clase); **no hace falta balancear**.
- **Celda 19 (correr):** resolución y canales → **celda 20** el resultado.
  - Output `(32, 256, 256, 3)`: es el default de `image_dataset_from_directory`; **la resolución nativa es 150×150**.
- **Celda 23 (correr):** grilla visual 6 clases × 3 fotos → **celda 24** lo que se observa.
- **Celda 25:** variabilidad (glacier≈sea, mountain≈forest).

---

## Bloque 4 — Preprocesamiento (`~2 min`)

- **Celda 28 / 30 / 32 (correr):** copia con `shutil` + limpieza (0 archivos inválidos).
- **Celda 34 (correr):** redimensionamiento a 64×64 → `(32, 64, 64, 3)`.
  - Justificación: 67.500 → 12.288 entradas; primera capa pasa de ~34,6 M a ~6,29 M de parámetros.
- **Celda 36 (correr):** normalización `Rescaling(1/255)` (píxel [85, 125, 184] → [0.33, 0.49, 0.72]).
- **Celda 39 (correr):** etiquetado automático (0→buildings … 5→street).
- **Celda 41 (correr): 🔑 EL SPLIT —** `validation_split=0.2, seed=42` sobre `seg_train`.
  - **Decir:** *"11.228 fotos entrenan, 2.806 validan, y las 3.000 de `seg_test` **no se tocan** hasta el final. Antes `seg_test` hacía de validación **y** test, lo que inflaba la métrica."*
  - Output esperado: `Train 351 lotes · Val 88 lotes · Test 94 lotes`.
- **Celdas 43–44 (correr):** pipeline con aumento de datos (Flip/Rotación/Zoom) **solo** en train.
- **Elemento visual:** pipeline de datos y formas de lote.

---

## Bloque 5 — Arquitectura del MLP (`~2 min`)

- **Celdas 46–51.** Teoría: MLP fully connected, Flatten, activaciones, pérdida (lr **0.0005** en celda 50).
- **Celda 52 (correr):** definición + `model.summary()`.
  - Arquitectura: **Flatten(12.288) → 512 ReLU → BN → Dropout 30 % → 256 ReLU → BN → Dropout 25 % → 128 ReLU → BN → Dropout 20 % → 6 Softmax**.
  - **Total params: 6.460.550** (trainable 6.458.758 · non-trainable 1.792).
  - Primeras capas: 6.291.968 + 131.328 + 32.896 + salida 774.
  - Compilado: `Adam lr=0.0005`, `sparse_categorical_crossentropy`.
- **Celda 53 (correr):** genera `diagrama_arquitectura_mlp.png` (requiere graphviz; si falla, mostrar el `summary`).
- **Celda 54:** lectura del `summary`.
- **Elemento visual:** `model.summary()` + diagrama de arquitectura.

---

## Bloque 6 — Entrenamiento (`~2.5 min`)

- **Celda 57 (correr): 🔑 `modelo.fit`** con los dos callbacks:
  - `EarlyStopping` (monitorea `val_loss`, patience 7, restaura mejores pesos).
  - `ReduceLROnPlateau` (factor 0.5, patience 2, min_lr 1e-5) — la lr baja de 0.0005 hasta 1e-5.
  - `validation_data=dataset_val_prep` → **el test NO se toca**.
  - **Mensaje importante:** con Dropout es normal que el accuracy de train sea menor que el de validación; curva cercana = generaliza.
- **Celda 60 (correr):** curvas de Accuracy y Loss (train vs validation).
- **Datos para el cierre (celda 58/61):**
  - **24 épocas** de las 50 programadas. `EarlyStopping` detectó el mejor punto en la **época 17**: `val_accuracy` **0.7124**, `val_loss` **0.8012**; tras 7 épocas sin mejora se detuvo y **restauró los pesos de la 17**.
  - Mejor época (17): `val_accuracy` **0.7124** vs acc train **0.6020** → val > train (Dropout/aumento activos solo en train) = **no hay sobreajuste**.
  - **Brecha train/test: 2.17 %** → **KPI de overfitting cumplido**.

---

## Bloque 7 — Resultados y análisis de errores (`~2.5 min`)

- **Celda 67 (correr):** evaluación en test + reporte de clasificación (incluye la subsección `c. Recall`).
  - **Accuracy en test: 0.6237 (62.37 %)** → KPI > 60 % **cumplido**.
  - **Test loss: 1.0155** → meta < 1.0000 **NO cumplido** (a 1.6 % de la meta). *Decirlo sin miedo: es el KPI que no llegamos y lo declaramos.*
  - Macro F1 **0.61**, Weighted **0.61** → KPI ≥ 0.60 **cumplido**.
  - Tabla por clase (test 3.000):

    | Clase | Prec | Rec | F1 |
    |---|---|---|---|
    | buildings | 0.57 | 0.38 | 0.45 |
    | forest | 0.76 | 0.80 | **0.78** |
    | glacier | 0.59 | 0.69 | 0.64 |
    | mountain | 0.60 | 0.66 | 0.63 |
    | sea | 0.56 | 0.41 | **0.47** |
    | street | 0.63 | 0.77 | 0.69 |

- **Celda 70 (correr):** matriz de confusión (glacier↔sea y street↔buildings) → **celda 71** interpretación.
- **Celda 74 (correr):** ejemplos bien/mal clasificados → **celda 75**.
- **Celda 76:** mejor/peor clase.
- **Cierre del bloque:** `forest` es la clase más fuerte (F1 **0.78**); `buildings` (**0.45**, recall 0.38) y `sea` (**0.47**, recall 0.41) son las débiles por similitud visual con `glacier` y `street`.

---

## Bloque 8 — Problemas, decisiones y conclusión (`~1.5 min`)

- **Celdas 77–80:** problemas y decisiones (colapso inicial ~18 % con entrada grande + dropout alto; BatchNorm + reducción a 64×64 + Dropout 30/25/20 + lr 0.0005 lo resolvieron).
- **Celda 81:** metodología **CRISP-DM**.
- **Celda 82:** evaluación de **impacto ético** (sesgo geográfico, límite de rendimiento declarado).
- **Celda 83–84:** conclusión.
- **Cierre:**
  - **KPIs: 4 de 5 cumplidos** — accuracy **62.37 %** (>60 %) · brecha **2.17 %** (≤5 %) · macro F1 **0.61** (≥0.60) · **~40 s/época** (<60 s) · **test loss 1.0155** (❌).
  - **Comparación honesta:** la versión anterior (con `seg_test` como validación y test) daba **64.13 %**. Al separar la validación del examen, baja a **62.37 %**. Esa caída de **1.76 puntos** es la evidencia de que la métrica anterior estaba **inflada**; el número nuevo es menor pero **honesto**.
  - **Limitación del MLP:** `Flatten` destruye la estructura espacial 2D → techo de generalización ~62 %.
  - **Trabajo futuro:** CNN (convolución + pooling) o transfer learning (ResNet/VGG) para superar el 85–90 %.

---

## Notas para el modo en vivo

1. **Antes de empezar:**
   - Verificar que `dataset/Intel Image Classification` y la copia `copy/Intel Image Classification` existen; si la copia no está, la celda 28 la crea (puede tardar unos segundos).
   - Si TensorFlow no detecta GPU, aparece un warning: **no es un error**, se entrena en CPU.
2. **Si el entrenamiento (celda 57) es largo en el momento de la demo:** mostrar el output ya guardado en el notebook (conserva las 24 épocas completas) o correr solo unas pocas épocas y explicar que el cuadro completo está en el documento. **~25 min** de entrenamiento en total.
3. **Para el diagrama (celda 53):** requiere Graphviz; si no está instalado, usar el `model.summary()` como elemento visual de arquitectura.
4. **Contingencia de red:** no descargar el dataset durante la clase; todo debe estar local.
5. **Tiempo:** respetar los bloques; el grueso de la evaluación ocurre en los bloques 5–7.

---

## ⚠️ Errores ya cometidos — no repetir

| Nunca digas | Di siempre |
| :--- | :--- |
| "la semilla es 52" | semilla **42** (celda 1) |
| "el baseline es 60 %" | baseline **16.6 %**; el 60 % es el **KPI** |
| "atarreoceros" / clases inventadas | las 6 clases: `buildings, forest, glacier, mountain, sea, street` |
| "64.13 %" como resultado actual | **62.37 %** (64.13 % era la corrida con validación contaminada) |
| "0.0001 de learning rate" | **0.0005** inicial; 1e-4 era un escalón de `ReduceLROnPlateau` |
