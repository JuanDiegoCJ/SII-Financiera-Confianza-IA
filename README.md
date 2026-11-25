# Transformación operativa y prevención de fraudes – Financiera Confianza Perú

Trabajo individual del curso **Sistemas de Información Integrados** (UPAO, 2026).
Estudiante: Carranza Jacinto Juan Diego.

## Programas

| Archivo | Caso | Técnica |
|---|---|---|
| `datos_financiera.py` | Datos sintéticos de operaciones de crédito | Generador con reglas de fraude |
| `01_ml_random_forest_fraude.py` | Ejemplo de Machine Learning / Componente 2 | Random Forest (scikit-learn) |
| `02_dl_red_neuronal_profunda.py` | Ejemplo de Deep Learning | Red neuronal profunda 13-64-32-16-1 en NumPy |
| `03_mineria_procesos.py` | Componente 1 | Descubrimiento (DFG), conformidad y mejora |
| `04_chatbot_pln.py` | Componente 3 | PLN: TF-IDF + regresión logística + SQLite |
| `05_prediccion_elecciones_2026.py` | Predicción de la segunda vuelta 2026 | Sentimiento dirigido + Naive Bayes + post-estratificación |

## Ejecución

```bash
pip install -r requirements.txt
python datos_financiera.py
python 01_ml_random_forest_fraude.py
python 02_dl_red_neuronal_profunda.py
python 03_mineria_procesos.py          # requiere Graphviz (comando dot)
python 04_chatbot_pln.py               # --chat para modo interactivo
python 05_prediccion_elecciones_2026.py  # usa X API v2 si existe X_BEARER_TOKEN
```

Las salidas (métricas, gráficos y registros de consola) quedan en `resultados/`.
Todos los datos de clientes son ficticios.
