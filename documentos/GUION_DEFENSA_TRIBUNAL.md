# Guion de Defensa ante el Tribunal — Preguntas clave y respuestas

> **Preparado:** 2026-09-22 · **Para:** equipo del proyecto (Gabriel, Martín, Francisco)
> **Objetivo:** tener listas las respuestas a lo que el tribunal pregunta seguido.
> **Archivo complementario:** [`CAMBIOS_NOTEBOOK_PRESENTACION.md`](CAMBIOS_NOTEBOOK_PRESENTACION.md) (qué cambió en el notebook y por qué).

---

## ⚠️ REGLA DE ORO ANTES DE EMPEZAR

**NO INVENTES. Si no lo sabés, di:**

> *"Lo tengo documentado en la sección X del notebook, déjenme confirmarlo."*

En este entrenamiento aparecieron **3 respuestas inventadas** y las 3 son errores graves en una defensa:

| Lo que se inventó | La verdad |
| :--- | :--- |
| "La semilla es **52**" | **No había semilla.** Ahora es **42** (sección 0). |
| "Los glaciares se confunden con **atardeceres**" | **No existe esa clase.** Son `glacier↔sea` y `buildings↔street`. |
| "El baseline está en **60%**" | El baseline aleatorio es **16.6%** (1 de 6). El 60% es el **KPI** de la pauta. |

**Todo lo que preguntó el tribunal ya está escrito en el notebook.** Leer > improvisar.

---

## 🔢 LOS 6 NÚMEROS QUE HAY QUE SABER DE MEMORIA

| Concepto | Valor |
| :--- | ---: |
| Baseline aleatorio | **16.6 %** |
| KPI de la pauta (accuracy) | **> 60 %** |
| **Resultado en test** | **62.37 %** |
| **Macro F1** | **0.61** |
| **Semilla** | **42** |
| **Learning rate inicial** | **0.0005** |

**KPIs: 4 de 5 cumplidos** (el test loss quedó en 1.0155 contra una meta de < 1.0000).

---

## 📋 ÍNDICE DE PREGUNTAS

1. [IE1 — ¿Cuál es la variante propia del problema?](#p1)
2. [IE2 — ¿Por qué este preprocesamiento?](#p2)
3. [IE3 — Diseño de la MLP](#p3)
4. [IE4/IE6 — Validación, test y semilla](#p4)
5. [IE5 — Resultados y análisis de error](#p5)
6. [IE6 — Reproducibilidad y trazabilidad](#p6)
7. [Preguntas trampa frecuentes](#p7)

---

<a name="p1"></a>
## P1 · IE1 — "¿Cuál es la variante propia del problema?"

**Lo que FALLÓ:** responder vago tipo *"nuestra variante es que definimos los objetivos del negocio"* — el tribunal escucha eso y pregunta *"¿y qué hace distinto a su trabajo?"*.

**✅ Respuesta correcta (ya está escrita en la sección 1 del notebook):**

> *"El dataset es el mismo Intel que usan otros grupos — ahí no está la diferencia. Nuestra variante está en **tres cosas propias**:*
> 1. ***Encuadramos la clasificación como motor de recomendación turístico**: la categoría predicha se traduce en tipo de destino (glaciar → destinos fríos, mar → costeros), conectando lo que el cliente quiere ver con lo que la empresa vende.*
> 2. ***Optimizamos macro-F1** y no solo accuracy, porque en un recomendador acertar siempre la clase mayoritaria arruina la experiencia de usuario.*
> 3. ***Restricción de despliegue**: debe correr en CPU con imágenes 64×64, priorizando balance precisión/coste sobre precisión máxima."*

**Truco:** decir explícitamente *"no es el dataset"* desarma la pregunta más común del tribunal.

---

<a name="p2"></a>
## P2 · IE2 — "¿Por qué este preprocesamiento?"

**Lo que FALLÓ:** justificar con frases vagas (*"nos parecieron las ideales"*) y no saber qué hacía el aumento de datos.

**✅ Respuesta — 3 decisiones, 3 justificaciones:**

| Decisión | Respuesta |
| :--- | :--- |
| **¿Por qué 64×64 y no 150×150?** | La entrada baja de 67.500 a **12.288 valores**: menos memoria, menos sobreajuste y permite entrenar en CPU. Es una restricción de negocio, no un capricho. |
| **¿Por qué normalizar 1/255?** | Lleva los píxeles de [0,255] a [0,1]. La red con ReLU **converge mejor** con entradas chicas y acotadas. |
| **¿Qué hace el aumento de datos?** | `RandomFlip`, `RandomRotation`, `RandomZoom(0.05)` generan **variantes de las mismas fotos** → el modelo generaliza en vez de memorizar. **Se aplica SOLO al train**: si lo aplicaras también a validación, estarías falseando la evaluación. |

**Truco:** toda decisión de preprocesamiento necesita **¿por qué?**. *"Nos pareció bien"* = pierde IE2.

---

<a name="p3"></a>
## P3 · IE3 — "Expliquen la arquitectura"

**✅ Las 3 preguntas clásicas:**

### 1. ¿Por qué `Flatten` si la imagen es 64×64×3?
> *"Las capas `Dense` **solo aceptan vectores**. `Flatten` convierte el tensor 64×64×3 en un vector de **12.288** valores. A cambio **destruye la información espacial** — y esa es exactamente la causa del techo de rendimiento que después mencionamos en conclusiones."*

(Esa colectada demuestra que entendés la limitación, no solo la arquitectura.)

### 2. ¿Por qué 6 neuronas y qué activación?
> *"**6 porque hay 6 clases** — siempre: neuronas de salida = número de clases. Lleva **softmax**, no ReLU, para que la salida sea una **distribución de probabilidades que suma 1** (ej. mar 72%, montaña 15%), entrenándose con `sparse_categorical_crossentropy` porque las etiquetas son enteros del 0 al 5."*

### 3. ¿Por qué 512 → 256 → 128 (cada vez más chicas)?
> *"Es un **embudo**. La primera capa recibe 12.288 valores sin procesar y necesita capacidad; después vamos comprimiendo para **forzar a la red a resumir**. Eso reduce parámetros, baja el costo de cómputo en CPU y funciona como **regularización implícita** contra el sobreajuste. Entre medias, `BatchNorm` estabiliza las activaciones y `Dropout` 30/25/20% apaga neuronas al azar."*

**Ojo:** la salida **no** lleva ReLU. Es el error más común al responder esto.

---

<a name="p4"></a>
## P4 · IE4/IE6 — "¿Cómo dividieron los datos? ¿Y la semilla?"

### 4.1 🔑 ¿Cómo está hecho el split? (ya se corrigió)

> *"Dividimos **`seg_train` en 80/20 con `validation_split=0.2, seed=42`**: **11.228 fotos entrenan** y **2.806 validan**. Las **3.000 fotos de `seg_test` se separaron por completo** y se evaluaron **una sola vez, al final**.*
>
> *Esto importa porque **antes `seg_test` cumplía doble función** — servía a la vez de validación y de test, así que el `EarlyStopping` y el `ReduceLROnPlateau` decidían mirando el propio examen. Lo detectamos, lo corregimos y lo documentamos en la sección 5 del notebook y en `CAMBIOS_NOTEBOOK_PRESENTACION.md`."*

**Pregunta de seguimiento casi segura: "¿y qué pasó con la métrica?"**

> *"Bajó de **64.13 %** a **62.37 %**. Y **tenía que bajar**: la caída de 1.76 puntos es la evidencia empírica de que la métrica anterior estaba inflada por la validación contaminada. Preferimos defender un número **menor pero honesto**."*

**Por qué funciona:** un problema **detectado, corregido y documentado** suma mucho más que un número inflado. Este es el mejor argumento de trazabilidad (IE6) que tiene el equipo.

### 4.2 ¿Cuál es la semilla y para qué sirve?
> *"**Semilla 42**, fijada con `tf.keras.utils.set_random_seed(42)` en la sección 0. Fija pesos iniciales, orden de las imágenes y el Dropout, para que **cualquiera que corra el notebook obtenga los mismos resultados**. Sin ella, cada corrida da métricas distintas y no se pueden comparar experimentos."*

### 4.3 ¿Con qué learning rate entrenaron?
> *"**0.0005** — es lo que dice el código (`Adam(learning_rate=0.0005)`) y lo confirman los logs (`5.0000e-04`). El `ReduceLROnPlateau` **parte de ahí y va bajando**: 5e-4 → 2.5e-4 → 1.25e-4 → … → 1e-5. Si en algún texto viejo aparece 0.0001, era **un escalón intermedio del decaimiento**, no la tasa inicial."*

**📌 Mandato:** el **código manda** sobre el texto. Si texto y código discrepan, el valor real es el del código.

---

<a name="p5"></a>
## P5 · IE5 — "¿Qué errores comete el modelo?"

### 5.1 ¿Qué clases se confunden más?
> *"Dos pares: **`glacier` ↔ `sea`** (ambas azuladas/blanquecinas: hielo y agua) y **`buildings` ↔ `street`** (ambas grises, estructuras rectas). Un MLP no extrae bordes ni texturas, así que separarlas por color le cuesta."*

**Dato de refuerzo (sale del reporte):** `buildings` tiene el recall más bajo (**0.38**) y `sea` el segundo (**0.41**). En sentido contrario, `glacier` tiene recall **0.69** pero precisión **0.59**: **absorbe** fotos que en realidad eran `sea`. Lo mismo hace `street` (recall 0.77, precisión 0.63) con las de `buildings`.

❌ **Prohibido:** inventar clases que no existen. Tus 6 clases son:
```
buildings, forest, glacier, mountain, sea, street
```

### 5.2 ¿Qué clase va mejor y por qué?
> *"`forest`, con **F1 = 0.78** y recall 0.80. Es casi **verde puro** — un color dominante que casi no se confunde con nada, y el color dominante es justo lo que un MLP sí sabe capturar. Le sigue `street` (0.69)."*

**Y la peor:**
> *`buildings` con **F1 = 0.45** y `sea` con **0.47**. Son las dos que comparten paleta con otra clase: mar con glaciar, edificios con calle."*

### 5.3 ¿El accuracy está sobre el baseline?
> *"**62.37 %** contra un baseline aleatorio de **16.6 %** (1 de 6), o sea **~3.8 veces mejor que adivinar**. Sobre el **KPI de la pauta (60 %)** quedamos **2.37 puntos arriba**, y eso es lo que motiva el análisis de error: sabemos que hay espacio."*

📌 **No confundir:**
| Concepto | Valor |
| :--- | ---: |
| Baseline aleatorio | **16.6 %** |
| KPI de la pauta | **60 %** |
| Nuestro resultado | **62.37 %** |

### 5.4 ❓ "¿Cuál de sus KPIs NO cumplieron?"
> *"El de **pérdida en test**: la meta era `< 1.0000` y quedó en **1.0155** — a 1.6 % de la meta. Significa que el modelo acierta bastante, pero cuando se equivoca lo hace con **más confianza de la que debería**, coherente con un MLP que no distingue `glacier` de `sea`. Lo reportamos tal cual en el notebook y en el informe; los otros **4 KPIs sí se cumplen**."*

⚠️ **Esta pregunta va a llegar.** Decirla **antes** de que la hagan ya ganó puntos en cualquier defensa: un equipo que anuncia su propio fallo se percibe como honesto.

---

<a name="p6"></a>
## P6 · IE6 — "¿Es reproducible? ¿Qué documentaron?"

### 6.1 Si un compañero corre el notebook, ¿sale lo mismo?
**No respondas solo "porque le faltan los datos"** — es cierto pero es el segundo problema:

> *"Con el notebook solo **no**: le faltan el dataset y las instrucciones del README. Ya está **fijada la semilla 42** en la sección 0, así que con la carpeta completa debería reproducir resultados **casi idénticos**."*

**Blindaje extra:** *"casi idénticos, pueden variar mínimamente según hardware y versión de TensorFlow."*

### 6.2 Tres cosas documentadas para trazabilidad

| # | Qué | Dónde |
| :--- | :--- | :--- |
| 1 | **Semilla 42** | Sección **0** del notebook |
| 2 | **Split 80/20**: 11.228 train / 2.806 val / 3.000 test (test ciego, una sola evaluación) | Sección **5** (tabla) |
| 3 | **KPIs y métricas reales**: 62.37 % accuracy, macro-F1 0.61, F1 por clase | Secciones **7** y **8** |

**+1 fuera del notebook (la más fuerte):**
- 📄 `documentos/EXPERIMENT_LOG.md` — **bitácora de experimentos**: qué probaste, qué falló y por qué cambiaste (incluye EXP-01 a EXP-04).

---

<a name="p7"></a>
## 🪤 P7 — Preguntas trampa frecuentes

### "¿Por qué no usaron una CNN?"
> *"Lo dejamos como trabajo futuro. El MLP fue el objetivo pedagógico de la asignatura. El `Flatten` destruye la estructura espacial, lo cual impone un techo ~62%. Una CNN preserva esa estructura con filtros locales y pooling, y estimamos que superaría el 85%."*

→ Nunca digas *"porque no sabíamos"*. Es una **limitación teórica declarada** (sección 10 del notebook).

### "¿Este 62.37% es honesto?"
> *"Sí, y es más honesto que la versión anterior del proyecto. Antes `seg_test` servía de validación **y** de test, lo que inflaba la métrica a 64.13%. Corregimos el particionado: ahora la validación es un 20% de `seg_train` y el test se evalúa **una sola vez**. Precisamente por eso la exactitud **bajó 1.76 puntos** — que baje es la prueba de que antes estaba inflada. Está documentado en la sección 5 y en `CAMBIOS_NOTEBOOK_PRESENTACION.md`."*

### "¿Cuántos parámetros tiene?"
> *"6.460.550 (~6,46 millones), de los cuales 6.29M están en la primera capa oculta, porque recibe los 12.288 píxeles aplanados."*

### "¿Cómo evitan el sobreajuste?"
> *"Tres mecanismos: **Dropout graduado** 30/25/20%, **BatchNormalization** tras cada capa oculta, y **aumento de datos solo en el train**. Además el embudo 512→256→128 comprime la representación. Resultado: la **brecha train/test quedó en 2.17 %** (60.20 % vs 62.37 %), muy por debajo del KPI de 5 %."*

**¿Y por qué el `val_accuracy` (71.24 %) es MAYOR que el de train (60.20 %)?**
> *"Porque **durante el entrenamiento el Dropout apaga neuronas al azar** y las fotos vienen aumentadas, así que el entrenamiento es deliberadamente más difícil. En validación no hay ni dropout ni aumento. Es lo **contrario** al sobreajuste, que sería train mucho más alto que val."*

### "¿Qué problema tuvieron?"
> *"El primer intento con entrada grande **colapsaba**: la red aprendía a predecir siempre la misma clase y se quedaba clavada en ~18%. Lo solucionamos bajando a 64×64, añadiendo BatchNorm y ajustando el dropout. Está en la sección 9 y en `EXPERIMENT_LOG.md`."*

→ **Este tipo de pregunta es regalada.** Un problema bien contado demuestra experiencia real.

### "¿Por qué paró antes de tiempo el entrenamiento?"
> *"Corrieron **24 de las 50 épocas**. El `EarlyStopping` encontró el mejor `val_loss` (**0.8012**) en la **época 17** y, como después no mejoró durante 7 épocas, se detuvo y **restauró los pesos de la época 17**. Entrenar más habría sido sobreentrenar por inercia."*

### "¿Es ético?"
> *"Sí, y lo evaluamos formalmente en la sección 9.2. Riesgo principal: **sesgo geográfico** — el dataset es mayoritariamente europeo, así que baterá en fotos de otras regiones. También declaramos que **~2 de cada 5 fotos se clasifican mal** (y que `sea` y `buildings` tienen recall por debajo de 0.45), para no exagerar el rendimiento. No hay datos personales ni usos de alto impacto."*

---

## 🎯 CHECKLIST 10 MINUTOS ANTES DE LA DEFENSA

- [ ] Leer **sección 1** (variante) en voz alta una vez
- [ ] Repasar las **6 clases**: `buildings, forest, glacier, mountain, sea, street`
- [ ] Memorizar los **3 números**: baseline **16.6 %** · KPI **60 %** · resultado **62.37 %**
- [ ] Saber el **LR = 0.0005** y que el código manda
- [ ] Saber la **semilla = 42** (sección 0)
- [ ] Saber el **split**: **11.228 / 2.806 / 3.000** y que el test se toca **una sola vez**
- [ ] Saber los **2 pares de confusión**: `glacier/sea` y `buildings/street`
- [ ] Tener presente el **F1 de forest = 0.78** y el peor (**buildings 0.45**)
- [ ] Tener preparada la respuesta del **KPI que NO se cumple** (test loss 1.0155)
- [ ] Tener preparada la respuesta **"¿por qué bajó de 64.13 a 62.37?"**
- [ ] Recordar la **regla de oro**: si no sabés, di "está documentado en la sección X"

---

## 📌 MAPA DE SECCIONES DEL NOTEBOOK (para "miren la sección X")

| Sección | Contenido |
| :--- | :--- |
| **0** | Semilla 42 y reproducibilidad |
| **1** | Variante del problema y objetivo de negocio |
| **2** | Problema de negocio (contexto, importancia, **KPIs**) |
| **3** | Selección y descripción del dataset |
| **4** | EDA: distribución, resolución, patrones, **muestra visual**, variabilidad |
| **5** | Preprocesamiento + **tabla de split 80/20** y por qué `seg_test` queda ciego |
| **6** | Diseño de la MLP |
| **7** | Entrenamiento y curvas (24 épocas, EarlyStopping en la 17) |
| **8** | Métricas, interpretación, matriz de confusión, ejemplos |
| **9** | Problemas y decisiones · **9.1 CRISP-DM** · **9.2 Ética** |
| **10** | Conclusión, **cumplimiento de KPIs (4 de 5)** y limitaciones del MLP |
