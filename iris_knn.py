
# Se reconfigura la salida estándar a UTF-8 para que los acentos y símbolos
# se impriman correctamente en la consola de Windows.
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


iris = datasets.load_iris()

X = iris.data          # Características (features)
y = iris.target        # Etiquetas (labels)

print("=" * 70)
print("DATASET IRIS")
print("=" * 70)
print(f"Número de muestras (filas)         : {X.shape[0]}")
print(f"Número de características (columnas): {X.shape[1]}")
print(f"Características                    : {iris.feature_names}")
print(f"Clases                             : {[str(nombre) for nombre in iris.target_names]}")
print()


# -----------------------------------------------------------------------------
# PASO 3: Dividir los datos en entrenamiento y prueba
# -----------------------------------------------------------------------------
# test_size=0.3  -> el 30% de los datos se reserva para evaluar el modelo.
# random_state=42 -> semilla fija: la división es siempre la misma, lo que
#                    garantiza resultados reproducibles.
# stratify=y     -> mantiene la misma proporción de cada especie en ambos
#                    conjuntos (opción recomendada en clasificación).
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("=" * 70)
print("DIVISIÓN DE LOS DATOS")
print("=" * 70)
print(f"Muestras de entrenamiento: {X_train.shape[0]} (70%)")
print(f"Muestras de prueba      : {X_test.shape[0]} (30%)")
print()


# -----------------------------------------------------------------------------
# PASO 4: Normalizar (escalar) las características
# -----------------------------------------------------------------------------
# StandardScaler transforma cada característica con la fórmula z = (x - media) / std.
# IMPORTANTE: el escalador se ajusta SOLO con los datos de entrenamiento
# (fit_transform) y luego se aplica a los de prueba (transform). Así se evita
# la "fuga de información" (data leakage) del conjunto de prueba.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # Ajusta y transforma el train
X_test_scaled = scaler.transform(X_test)        # Solo transforma el test

print("=" * 70)
print("NORMALIZACIÓN CON StandardScaler")
print("=" * 70)
print("Media aproximada de cada característica en el conjunto de entrenamiento:")
print([round(float(valor), 3) for valor in scaler.mean_])
print("Desviación estándar de cada característica en el conjunto de entrenamiento:")
print([round(float(valor), 3) for valor in scaler.scale_])
print()


# -----------------------------------------------------------------------------
# PASO 5: Crear y entrenar el modelo k-NN
# -----------------------------------------------------------------------------
# n_neighbors=5 -> cada predicción se decide por mayoría entre los 5 vecinos
#                  más cercanos del punto a clasificar.
modelo = KNeighborsClassifier(n_neighbors=5)

# fit() entrena el modelo: k-NN simplemente "memoriza" los datos escalados.
modelo.fit(X_train_scaled, y_train)

print("=" * 70)
print("MODELO K-NEAREST NEIGHBORS")
print("=" * 70)
print(f"Modelo entrenado con k = {modelo.n_neighbors} vecinos.")
print()


# -----------------------------------------------------------------------------
# PASO 6: Realizar predicciones sobre el conjunto de prueba
# -----------------------------------------------------------------------------
# predict() devuelve la clase predicha (0, 1 o 2) para cada flor de prueba.
y_pred = modelo.predict(X_test_scaled)


# -----------------------------------------------------------------------------
# PASO 7: Evaluar el modelo y mostrar los resultados
# -----------------------------------------------------------------------------
# accuracy_score      -> proporción de predicciones correctas (0 a 1).
# confusion_matrix    -> filas = clase real, columnas = clase predicha.
# classification_report -> precisión, recall, f1-score y support por clase.
accuracy = accuracy_score(y_test, y_pred)
matriz_confusion = confusion_matrix(y_test, y_pred)
reporte = classification_report(y_test, y_pred, target_names=iris.target_names)

print("=" * 70)
print("EVALUACIÓN DEL MODELO")
print("=" * 70)
print(f"Exactitud (accuracy): {accuracy:.4f}  ->  {accuracy * 100:.2f}%")
print()

print("Matriz de confusión:")
print("(filas = clase real, columnas = clase predicha)")
print(matriz_confusion)
print()

print("Informe de clasificación:")
print(reporte)

# Comparación breve entre valores reales y predichos para las primeras flores.
print("-" * 70)
print("Primeras 10 predicciones del conjunto de prueba:")
print("-" * 70)
print(f"{'#':<4}{'Real':<12}{'Predicho':<12}{'¿Acierto?':<10}")
for i in range(10):
    nombre_real = str(iris.target_names[y_test[i]])
    nombre_pred = str(iris.target_names[y_pred[i]])
    acierto = "Sí" if y_test[i] == y_pred[i] else "No"
    print(f"{i + 1:<4}{nombre_real:<12}{nombre_pred:<12}{acierto:<10}")
print("=" * 70)
