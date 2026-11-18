"""Diagnóstico final: compara todas las pruebas sobre el modelo ya congelado.

Uso:  python -m utils.informe_final

Lee (no modifica) los resultados ya generados:
    results/metricas_prueba.json      -> exactitud en el conjunto de prueba
    results/prueba_ruido.json         -> 30 imágenes originales y con ruido
    results/prueba_20_dibujos.csv     -> 20 dibujos hechos por el estudiante

Genera:
    results/diagnostico_final.json
    results/diagnostico_final.md

Si la prueba de 20 dibujos todavía no está realizada, el informe lo indica y NO
inventa cifras.
"""

from __future__ import annotations

import json
import sys

from utils import RESULTS_DIR, UMBRAL_CONFIANZA
from utils import history as hist


def construir_diagnostico() -> dict:
    metricas = json.loads((RESULTS_DIR / "metricas_prueba.json").read_text(encoding="utf-8"))
    ruido = json.loads((RESULTS_DIR / "prueba_ruido.json").read_text(encoding="utf-8"))
    prueba20 = hist.resumen_prueba_20()

    advertencias_ruido = sum(
        1
        for fila in ruido["detalle"]
        if max(fila["prob_max_original"], fila["prob_max_con_ruido"]) < UMBRAL_CONFIANZA
    )

    return {
        "modelo": metricas["modelo_seleccionado"],
        "umbral_baja_confianza": UMBRAL_CONFIANZA,
        "exactitud_conjunto_prueba": metricas["modelo_final_en_prueba"]["accuracy"],
        "tamano_conjunto_prueba": metricas["tamano_prueba"],
        "exactitud_20_dibujos_propios": prueba20["accuracy"],
        "prueba_20_dibujos_realizada": prueba20["realizada"],
        "prueba_20_registrados": prueba20["registrados"],
        "prueba_20_faltantes": prueba20["faltan"],
        "prueba_20_correctos": prueba20["correctos"],
        "prueba_20_incorrectos": prueba20["incorrectos"],
        "exactitud_30_imagenes_original": ruido["exactitud_original"],
        "exactitud_30_imagenes_con_ruido": ruido["exactitud_con_ruido"],
        "diferencia_accuracy_ruido": ruido["diferencia_accuracy"],
        "advertencias_baja_confianza_20_dibujos": prueba20["advertencias_baja_confianza"],
        "advertencias_baja_confianza_en_30_imagenes": advertencias_ruido,
    }


def texto_markdown(d: dict) -> str:
    linea_20 = (
        f"{d['exactitud_20_dibujos_propios'] * 100:.2f} %"
        if d["exactitud_20_dibujos_propios"] is not None and d["prueba_20_registrados"] > 0
        else "PENDIENTE (el estudiante no ha completado los 20 dibujos)"
    )
    estado_20 = (
        f"completa ({d['prueba_20_registrados']} registros)"
        if d["prueba_20_dibujos_realizada"]
        else f"incompleta ({d['prueba_20_registrados']} de 20 registros, faltan {d['prueba_20_faltantes']})"
    )
    return f"""# Diagnóstico final

Modelo congelado: **{d['modelo']}** · umbral de baja confianza: {d['umbral_baja_confianza']}

| Prueba | Exactitud | Observaciones |
|---|---|---|
| Conjunto de prueba ({d['tamano_conjunto_prueba']} imágenes) | {d['exactitud_conjunto_prueba'] * 100:.2f} % | Evaluación única, no usada para entrenar ni elegir |
| 30 imágenes originales (semilla fija) | {d['exactitud_30_imagenes_original'] * 100:.2f} % | Subconjunto de prueba |
| 30 imágenes con ruido gaussiano N(0, 0.15) | {d['exactitud_30_imagenes_con_ruido'] * 100:.2f} % | Diferencia: {d['diferencia_accuracy_ruido'] * 100:.2f} puntos porcentuales |
| 20 dibujos propios | {linea_20} | Prueba {estado_20} |
| Avisos de baja confianza | 20 dibujos: {d['advertencias_baja_confianza_20_dibujos']} · 30 imágenes: {d['advertencias_baja_confianza_en_30_imagenes']} | Probabilidad máxima < {d['umbral_baja_confianza']} |

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
"""


def main() -> int:
    if not (RESULTS_DIR / "metricas_prueba.json").exists() or not (RESULTS_DIR / "prueba_ruido.json").exists():
        print("Faltan resultados. Ejecuta primero: python train.py")
        return 1
    diagnostico = construir_diagnostico()
    hist.guardar_json(RESULTS_DIR / "diagnostico_final.json", diagnostico)
    (RESULTS_DIR / "diagnostico_final.md").write_text(texto_markdown(diagnostico), encoding="utf-8")
    if not diagnostico["prueba_20_dibujos_realizada"]:
        print(
            "AVISO: la prueba de 20 dibujos propios todavía no está completa "
            f"({diagnostico['prueba_20_registrados']} de 20). El informe no calcula su exactitud."
        )
    print("Informe generado en results/diagnostico_final.md y .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
