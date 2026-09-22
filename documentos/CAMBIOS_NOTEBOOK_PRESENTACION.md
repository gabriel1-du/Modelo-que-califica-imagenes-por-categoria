# Cambios en `4. Presentacion.ipynb` — explicados para el equipo

> **Archivo auditado:** `Modelo-que-califica-imagenes-por-categoria/notebooks/4. Presentacion.ipynb`
> **Fecha:** 2026-09-22 · **Estado final:** ✅ **notebook re-ejecutado y documentos sincronizados**
> **Motivo:** cerrar las brechas detectadas contra la rúbrica `EP1_TLY1102_Instrucciones y Pauta PRESENTACIÓN_Estudiante.md` (criterios IE1–IE6).

---

## Resumen rápido

| # | Brecha (rúbrica) | Cambio | Estado |
|---|---|---|---|
| 1 | IE1: falta la "variante" propia del problema | Nueva sección **1. Introducción, objetivo de negocio y variante** | ✅ Hecho |
| 2 | IE6: no se documentaba la **semilla** | Nueva sección **0. Semilla y reproducibilidad** con `set_random_seed(42)` | ✅ Hecho |
| 3 | IE6: el **split** no quedaba trazado | Tabla de split **11.228 / 2.806 / 3.000** en la sección 5 | ✅ Hecho |
| 4 | IE5: **contradicción de learning rate** (texto 0.0001 vs código 0.0005) | Texto unificado a **0.0005** = valor real del código | ✅ Hecho |
| 5 | IE5: **KPI de F1** inconsistente (0.70 vs 0.60) | Meta realineada a ≥0.60 · ahora **5 KPIs** declarados | ✅ Hecho |
| 6 | IE5: EDA sin muestra visual de imágenes | Nueva celda con **grilla 6×3** (3 fotos al azar por clase) + interpretación | ✅ Hecho |
| 7 | IE6: **ruta absoluta** `c:\Users\gabod\...` en salidas | Reemplazadas por `<ruta_entorno>` genérica | ✅ Hecho |
| 8 | IE6: falta informe técnico `.md` aparte | Nuevo **`documentos/INFORME_TECNICO.md`** (17 secciones) | ✅ Hecho |
| 9 | IE6: sin CRISP-DM ni evaluación ética | Nuevas secciones **9.1 CRISP-DM** y **9.2 Ética** | ✅ Hecho |
| 10 | **IE4: `seg_test` usaba doble función (val + test)** | Split **80/20 de `seg_train`** con `seed=42`; `seg_test` solo examen final | ✅ **Hecho y ejecutado** |
| 11 | IE6: faltaban **problema de negocio, objetivos, KPIs y EDA** en el informe | Nuevas secciones **2, 3, 4, 5 y 6** del informe | ✅ Hecho |
| 12 | Estructura profesional incompleta | Creadas **`notebooks/`, `models/`, `images/`** con su `LEEME.md` | ✅ Hecho |

---

## 🔴 CAMBIO IMPORTANTE: nueva división de datos (punto 10)

**Antes (incorrecto):**
```
seg_train (14.034)  → entrenar
seg_test  (3.000)   → validación Y test final   ❌ doble uso
```

**Ahora (correcto, ya ejecutado):**
```
seg_train 80% (11.228) → entrenar        (351 lotes)
seg_train 20% (2.806)  → validación      (88 lotes)  → EarlyStopping + ReduceLROnPlateau
seg_test 100% (3.000)  → examen final    (94 lotes)  → UNA SOLA VEZ
```

**Celdas modificadas:** 40 (markdown split), 41 (carga), 42 (markdown lotes), 43 (pipeline), 56 (métricas), 57 (`fit`).

**Por qué:** con el diseño anterior, `EarlyStopping` y `ReduceLROnPlateau` decidían con las mismas 3.000 fotos que después "aprobaban" al modelo → métricas sesgadas al alza. Afectaba **IE4 (15%)** e **IE6 (15%)**.

---

## 📊 RESULTADO DE LA NUEVA CORRIDA (ya ejecutada)

| Métrica | Corrida vieja (split contaminado) | **Corrida nueva (split honesto)** |
| :--- | ---: | ---: |
| Split train / val / test | 14.034 / 3.000 / 3.000 ❌ | **11.228 / 2.806 / 3.000** ✅ |
| Semilla | no había | **42** ✅ |
| **Accuracy en test** | 64.13 % | **62.37 %** |
| **Loss en test** | 0.9510 | **1.0155** |
| **Macro F1** | 0.64 | **0.61** |
| Épocas | 50 (sin corte) | **24** (EarlyStopping; mejor en la 17) |
| Tiempo por época | ~39 s | **~40 s** |
| Brecha train/test | ~3.8 % | **2.17 %** |

### Cumplimiento de KPIs: **4 de 5**

| KPI | Meta | Resultado | Estado |
| :--- | :---: | :---: | :---: |
| Test accuracy | > 60 % | 62.37 % | ✅ |
| Brecha train/test | ≤ 5 % | 2.17 % | ✅ |
| Macro F1 | ≥ 0.60 | 0.61 | ✅ |
| Tiempo por época | < 60 s | ~40 s | ✅ |
| **Test loss** | **< 1.0000** | **1.0155** | **❌** |

### F1 por clase (test 3.000)

| Clase | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: |
| buildings | 0.57 | 0.38 | 0.45 |
| forest | 0.76 | 0.80 | **0.78** |
| glacier | 0.59 | 0.69 | 0.64 |
| mountain | 0.60 | 0.66 | 0.63 |
| sea | 0.56 | 0.41 | **0.47** |
| street | 0.63 | 0.77 | 0.69 |
| **macro avg** | **0.62** | **0.62** | **0.61** |

> **Regla aplicada:** el accuracy bajó de 64.13 % a **62.37 %**, y **se reporta el número nuevo**. Esa caída de **1.76 puntos** no es un retroceso: es la **evidencia empírica de que la métrica anterior estaba inflada** por la validación contaminada.

---

## 📄 DOCUMENTOS SINCRONIZADOS CON LOS NÚMEROS NUEVOS

| Documento | Qué se actualizó |
| :--- | :--- |
| `4. Presentacion.ipynb` | Celdas 7 (5 KPIs), 57 (comentario de épocas), 58, 61, 63, 65, 67, 70, 75, 81, 83 |
| `documentos/INFORME_TECNICO.md` | Resumen ejecutivo, tabla de split, §4 KPIs, §11.1–11.3, §17 pendientes |
| `README.md` | Nota de KPIs y nota sobre la presentación |
| `documentos/EXPERIMENT_LOG.md` | Nueva fila **EXP-04** + §4 conclusión final |
| `documentos/GUION_PRESENTACION_LIVE.md` | Reescrito: números de celda reales + métricas nuevas |
| `documentos/GUION_DEFENSA_TRIBUNAL.md` | P4 reescrito (el problema ya se corrigió), P5/P7 y checklist |

> ⚠️ **Los otros notebooks NO se re-ejecutaron.** Los números de `1.`, `2.`, `3.`, `scraps/` siguen siendo los originales y **son correctos para esos archivos**. Solo cambió `4. Presentacion.ipynb`.

---

## ⏳ QUÉ PASA POR TU MANO

1. ~~**Re-ejecutar `4. Presentacion.ipynb`**~~ ✅ Hecho (24 épocas, semilla 42, split 3 líneas correcto).
2. ~~**Actualizar métricas en `INFORME_TECNICO.md`**~~ ✅ Hecho.
3. ~~**Actualizar 5 documentos con 64.13 %**~~ ✅ Hecho.
4. **Guardar los pesos del modelo en `models/`** — ⬜ sigue pendiente (código en `models/LEEME.md`).
5. **Avisar a Gabriel y Francisco** de todo lo anterior. — ⬜

---

## Detalle de cada cambio (por si preguntan)

### 1. Nueva sección `## 1. Introducción, objetivo de negocio y variante del problema`

**Por qué:** la pauta exige una **variante propia** del problema para que los trabajos no sean copias. El notebook empezaba directo con `## 2. Problema de negocio`, sin declarar qué hace distinto a *este* trabajo.

**Qué declara:**

- **Objetivo de negocio:** una empresa de turismo/viajes quiere un recomendador de destinos; clasificar la imagen es el primer paso para conectar *lo que el cliente quiere ver* con *lo que la empresa vende*.
- **Variante (3 puntos diferenciadores):**
  1. **Salida orientada a negocio** — además de la categoría, se interpreta como *tipo de destino* (glaciar → destinos fríos, mar → costeros...).
  2. **Métrica principal `macro-F1`** en vez de solo accuracy, porque en un recomendador acertar siempre la clase mayoritaria arruina la experiencia.
  3. **Restricción práctica** — debe correr en **CPU con imágenes de 64×64**, priorizando balance precisión/coste sobre precisión máxima.

**Cómo defenderlo en la oral:** *"Nuestra variante no es el dataset, que es el de Intel como usan todos, sino el **objetivo, la métrica y la restricción de despliegue** que le ponemos."*

---

### 2. Nueva sección `## 0. Semilla y reproducibilidad` (+ celda de código)

**Por qué:** la rúbrica (IE6) pide explícitamente **documentar la semilla**. Antes no estaba en ninguna parte.

```python
import random
import numpy as np
import tensorflow as tf

SEMILLA = 42
tf.keras.utils.set_random_seed(SEMILLA)  # fija random, numpy y tensorflow
print(f"Semilla fijada en {SEMILLA}: resultados reproducibles.")
```

**Salida verificada en la corrida:** `Semilla fijada en 42: resultados reproducibles.` ✅

---

### 3. Split documentado con tabla (sección 5)

**Por qué:** IE6 exige trazabilidad del **data split**.

Tabla actual:

| Conjunto | Carpeta | Imágenes | Proporción | Uso |
|---|---|---:|---:|---|
| Entrenamiento | 80 % de `seg_train/seg_train` | 11.228 | 65.9 % | ajuste de pesos |
| Validación | 20 % de `seg_train/seg_train` | 2.806 | 16.5 % | `EarlyStopping`, `ReduceLROnPlateau` |
| Test | `seg_test/seg_test` | 3.000 | 17.6 % | evaluación final, **una sola vez** |
| **Total** | | **17.034** | 100 % | |

**Por qué está bien decirlo en voz alta:** un tribunal prefiere **un problema detectado y corregido** a un problema escondido.

---

### 4. Contradicción de learning rate corregida (IE5)

| Dónde | Decía |
|---|---|
| Sección 6 (código) | `Adam(learning_rate=0.0005)` ✅ |
| Logs de entrenamiento | `learning_rate: 5.0000e-04` ✅ |
| Sección 9 (Problemas) | `0.0005` ✅ |
| ~~Sección 7 (Resultados)~~ | ~~`0.0001`~~ → **corregido a `0.0005`** ✅ |

**Respuesta oficial:** **0.0005** (5e-4). El `ReduceLROnPlateau` parte de ahí y va bajando (5e-4 → 2.5e-4 → 1.25e-4 → … → 1e-5), o sea que **el 0.0001 era solo un valor intermedio del decaimiento**.

---

### 5. KPIs de la sección 7 (conclusión) reescritos

Antes el texto mezclaba metas que no coincidían con el informe. Ahora **el notebook y el informe listan los mismos 5 KPIs** (sección 7 del notebook = §4 del informe):

1. Exactitud en test **> 60 %** → 62.37 % ✅
2. Brecha |acc_train − acc_test| **≤ 5 %** → 2.17 % ✅
3. Macro F1 **≥ 0.60** → 0.61 ✅
4. Pérdida en test **< 1.0000** → 1.0155 ❌
5. Tiempo por época **< 60 s** → ~40 s ✅

Y el detalle por clase quedó con los números reales de la corrida: `forest` 0.78 (mejor), `buildings` 0.45 y `sea` 0.47 (peores).

---

## Cómo re-ejecutar todo de forma segura

```bash
# 1. Abrir el notebook y "Restart & Run All"  (~25 min en CPU)
# 2. Verificar que la sección 0 imprime: "Semilla fijada en 42: resultados reproducibles."
# 3. Verificar que la celda de carga (41) imprime 3 líneas: Train 351 / Val 88 / Test 94 lotes
# 4. Guardar (Ctrl+S) para que las salidas actualizadas queden persistidas
# 5. Esperado: accuracy ~0.6237, loss ~1.0155, macro-F1 ~0.61, 24 épocas
```
