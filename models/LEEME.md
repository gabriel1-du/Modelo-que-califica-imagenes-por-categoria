# models/

Aquí se guardan los **pesos entrenados** para no tener que re-entrenar cada vez.

Para generar el archivo, agregar al final del notebook:

```python
modelo.save("models/mlp_paisajes_64x64.keras")
print("Modelo guardado en models/mlp_paisajes_64x64.keras")
```

Y para cargarlo:

```python
import tensorflow as tf
modelo_cargado = tf.keras.models.load_model("models/mlp_paisajes_64x64.keras")
```
