# Guion de Presentación — Live Coding con `4. Presentacion.ipynb`

**Asignatura:** Técnicas avanzadas de machine learning - 002D
**Equipo:** Martín Higuera · Gabriel Durán · Francisco Salazar

Formato de la presentación: **código en vivo**. Se corre `4. Presentacion.ipynb` en orden mientras se explica; las celdas marcan el ritmo y cada sección cierra con la métrica o la figura clave. Duración estimada: **~12–15 min**.

> Referencias: la numeración de "celda" corresponde a las celdas del notebook `4. Presentacion.ipynb`. Los elementos visuales que se mencionan se pueden capturar del propio output del notebook o pegar en las diapositivas del docx.

---

## Bloque 0 — Portada e intro (`~1 min`)

- **Celdas 0–4.** Leer títulos e integrantes.
- **Qué decir:** el proyecto consiste en entrenar un Perceptrón Multicapa (MLP) en TensorFlow para clasificar paisajes en 6 categorías; recorreremos el ciclo completo: EDA → preprocesamiento → arquitectura → entrenamiento → evaluación de errores.

---

## Bloque 1 — Problema de negocio y KPIs (`~1.5 min`)

- **Celdas 2–4.** Contexto, importancia y KPIs.
- **Puntos a reforzar:**
  - Fotos 150×150×3 = 67.500 números → se bajarán a 64×64 (12.288) por viabilidad en CPU/RAM.
  - Usos: turismo, medio ambiente, redes sociales, ciudades inteligentes.
  - **KPIs:** Accuracy > 60% · |Acc_train − Acc_test| ≤ 5% · Macro F1 ≥ 0.60 · tiempo por época < 60 s.

---

## Bloque 2 — Dataset (`~1 min`)

- **Celdas 5–10.** Selección y descripción (Intel Image Classification, 6 clases, ~25.000 imágenes).
- **Dato clave:** 14.034 (train) / 3.000 (test) / 7.301 (pred sin etiqueta).

---

## Bloque 3 — EDA (`~1.5 min`)

- **Celda 13 (correr):** conteo por clase.
  - Resultado: clases balanceadas (train ~2.200–2.500 por clase); **no hace falta balancear**.
- **Celda 16 (correr):** resolución y canales.
  - Output `(32, 256, 256, 3)`: es el default de `image_dataset_from_directory`; **la resolución nativa es 150×150** (se aclara en la celda 17).
- **Celdas 18–19:** patrones visuales y variabilidad (glacier≈sea, mountain≈forest).

---

## Bloque 4 — Preprocesamiento (`~2 min`)

- **Celdas 22/24/26 (correr):** copia con `shutil` + limpieza (0 archivos inválidos).
- **Celda 28 (correr):** redimensionamiento a 64×64 → `(32, 64, 64, 3)`.
  - Justificación: 67.500 → 12.288 entradas; primera capa pasa de ~34,6 M a ~6,29 M de parámetros.
- **Celda 30 (correr):** normalización `Rescaling(1/255)` (píxel [85, 125, 184] → [0.33, 0.49, 0.72]).
- **Celda 33 (correr):** etiquetado automático (0→buildings … 5→street).
- **Celdas 35/37/38 (correr):** división **100% `seg_train` para train** (439 lotes) y **100% `seg_test` como validación/test** (94 lotes), batch 32; aumento de datos (Flip/Rotación/Zoom) **solo** en train.
- **Elemento visual:** pipeline de datos (normalización + aumento) y formas de lote.

---

## Bloque 5 — Arquitectura del MLP (`~2 min`)

- **Celdas 40–45.** Teoría: MLP fully connected, Flatten, activaciones, pérdida.
- **Celda 46 (correr):** definición + `model.summary()`.
  - Arquitectura: **Flatten(12.288) → 512 ReLU → BN → Dropout 30% → 256 ReLU → BN → Dropout 25% → 128 ReLU → BN → Dropout 20% → 6 Softmax**.
  - **Total params: 6.460.550** (trainable 6.458.758 · non-trainable 1.792).
  - Primeras capas: 6.291.968 + 131.328 + 32.896 + salida 774.
  - Compilado: `Adam lr=0.0005`, `sparse_categorical_crossentropy`.
- **Celda 47 (correr):** genera `diagrama_arquitectura_mlp.png` (requiere graphviz; si falla, mostrar el `summary`).
- **Elemento visual:** `model.summary()` + diagrama de arquitectura.

---

## Bloque 6 — Entrenamiento (`~2.5 min`)

- **Celdas 49–52.**
- **Celda 51 (correr):** `model.fit` con los dos callbacks:
  - `EarlyStopping` (monitorea `val_loss`, patience 7, restaura mejores pesos).
  - `ReduceLROnPlateau` (factor 0.5, patience 2, min_lr 1e-5) — la lr baja de 0.0005 hasta 1e-5.
  - **Mensaje importante:** con Dropout es normal que el accuracy de train sea menor que el de validación; curva cercana = generaliza.
- **Celda 54 (correr):** curvas de Accuracy y Loss (train vs validation).
- **Datos para el cierre:** mejor `val_accuracy` 0.6430 (época 43), mejor `val_loss` 0.9510 (época 46); la brecha |acc_train − acc_val| se mantiene ≤ 5% → **KPI de overfitting cumplido**.

---

## Bloque 7 — Resultados y análisis de errores (`~2.5 min`)

- **Celdas 56–60.**
- **Celda 60 (correr):** evaluación en test + reporte de clasificación.
  - **Accuracy en test: 0.6413 (64.13%)** → KPI > 60% cumplido.
  - Macro F1 **0.63**, Weighted **0.64** → KPI ≥ 0.60 cumplido.
  - Tabla por clase (test 3.000):

    | Clase | Prec | Rec | F1 |
    |---|---|---|---|
    | buildings | 0.56 | 0.44 | 0.49 |
    | forest | 0.78 | 0.81 | **0.79** |
    | glacier | 0.63 | 0.66 | 0.64 |
    | mountain | 0.61 | 0.70 | 0.65 |
    | sea | 0.61 | 0.48 | 0.54 |
    | street | 0.65 | 0.74 | 0.69 |

- **Celda 63 (correr):** matriz de confusión (glacier↔sea y street↔buildings).
- **Celda 67 (correr):** ejemplos bien/mal clasificados.
- **Cierre del bloque:** `forest` es la clase más fuerte (F1 0.79); `buildings` (0.49) y `sea` (0.54) son las débiles por similitud visual con `street` y `glacier`.

---

## Bloque 8 — Problemas, decisiones y conclusión (`~1.5 min`)

- **Celdas 70–75.** Leer problemas y decisiones (colapso inicial ~18% con 150×150 + dropout alto; BatchNorm + Reducción a 64×64 + Dropout 30/25/20 + lr 0.0005 lo resolvieron).
- **Cierre:**
  - **KPIs cumplidos:** 64.13% (>60%) · brecha ≤5% · Macro F1 0.63 (≥0.60) · ~40 s/época (<60 s).
  - **Limitación del MLP:** Flatten destruye la estructura espacial 2D → techo de generalización bajo.
  - **Trabajo futuro:** CNN (convolución + pooling) o transfer learning (ResNet/VGG) para superar el 85–90%.

---

## Notas para el modo en vivo

1. **Antes de empezar:**
   - Verificar que `dataset/Intel Image Classification` y la copia `copy/Intel Image Classification` existen; si la copia no está, la celda 22 la crea (puede tardar unos segundos).
   - Si TensorFlow no detecta GPU, aparece un warning: **no es un error**, se entrena en CPU.
2. **Si el entrenamiento (celda 51) es largo en el momento de la demo:** mostrar el output ya guardado en el notebook (la celda conserva las 50 épocas) o correr solo unas pocas épocas y explicar que el cuadro completo está en el documento.
3. **Para el diagrama (celda 47):** requiere Graphviz; si no está instalado, usar el `model.summary()` como elemento visual de arquitectura.
4. **Contingencia de red:** no descargar el dataset durante la clase; todo debe estar local.
5. **Tiempo:** respetar los bloques; el grueso de la evaluación ocurre en los bloques 5–7.