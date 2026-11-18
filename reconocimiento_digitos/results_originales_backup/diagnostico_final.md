# Diagnóstico final

Modelo congelado: **Sequential** · umbral de baja confianza: 0.7

| Prueba | Exactitud | Observaciones |
|---|---|---|
| Conjunto de prueba (360 imágenes) | 98.33 % | Evaluación única, no usada para entrenar ni elegir |
| 30 imágenes originales (semilla fija) | 100.00 % | Subconjunto de prueba |
| 30 imágenes con ruido gaussiano N(0, 0.15) | 96.67 % | Diferencia: 3.33 puntos porcentuales |
| 20 dibujos propios | PENDIENTE (el estudiante no ha completado los 20 dibujos) | Prueba incompleta (0 de 20 registros, faltan 20) |
| Avisos de baja confianza | 20 dibujos: 0 · 30 imágenes: 0 | Probabilidad máxima < 0.7 |

## ¿Por qué pueden existir diferencias?

- **Datos de prueba**: provienen del mismo conjunto y de la misma escala que el
  entrenamiento; el modelo ya "vio" imágenes de ese estilo.
- **30 imágenes con ruido**: el ruido cambia el valor de cada píxel y borra o inventa
  trazos, de ahí la caída de exactitud.
- **30 imágenes originales**: al ser solo 30, cada acierto o error pesa 3.33 puntos
  porcentuales; pequeñas diferencias frente al conjunto completo son normales.
- **Dibujos propios**: cambian el estilo (grosor, proporción, posición). Aunque el
  preprocesamiento los recorta, invierte y centra, el trazo del ratón no es idéntico a un
  dígito escrito a mano de 8x8. Además son solo 20, así que un error mueve 5 puntos.

## Propuesta de mejora

Aplicar **aumento de datos** (rotaciones pequeñas, desplazamientos y cambios de grosor) al
conjunto de entrenamiento para que la red vea trazos parecidos a los del ratón, y **calibrar
las probabilidades** para que el umbral de 0.70 tenga un significado estadístico real.
