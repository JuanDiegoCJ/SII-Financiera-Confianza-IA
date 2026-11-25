"""
01_ml_random_forest_fraude.py
-----------------------------
EJEMPLO DE MACHINE LEARNING: Random Forest para detectar fraude en desembolsos.

Corresponde al Componente 2 del informe (modelo de ML embebido en el ERP).
Un Random Forest es un conjunto (ensemble) de árboles de decisión entrenados sobre
muestras bootstrap y subconjuntos aleatorios de variables; la predicción final es el
promedio de las probabilidades de los árboles. Es aprendizaje supervisado clásico:
las variables de entrada se diseñan a mano (ingeniería de características).

Entrada : 13 variables de la operación (ver datos_financiera.VARIABLES)
Salida  : probabilidad de fraude, nivel de riesgo y acción recomendada para el ERP

Ejecución:
    python 01_ml_random_forest_fraude.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report)
from datos_financiera import generar_operaciones, VARIABLES

UMBRAL_REVISION = 0.30   # a partir de aquí la operación pasa a revisión manual
UMBRAL_BLOQUEO = 0.70    # a partir de aquí el desembolso se retiene


def nivel_riesgo(p: float) -> dict:
    """Traduce la probabilidad a la salida que consume el ERP."""
    if p >= UMBRAL_BLOQUEO:
        return {"nivel": "ALTO", "accion": "RETENER_DESEMBOLSO_Y_ALERTAR"}
    if p >= UMBRAL_REVISION:
        return {"nivel": "MEDIO", "accion": "REVISION_ANALISTA_RIESGOS"}
    return {"nivel": "BAJO", "accion": "APROBAR_FLUJO_NORMAL"}


def main():
    # 1. Datos
    df = generar_operaciones(n=20000, semilla=42)
    X, y = df[VARIABLES], df["fraude"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=7)

    # 2. Entrenamiento (class_weight compensa el desbalance: el fraude es raro)
    modelo = RandomForestClassifier(n_estimators=300, max_depth=10,
                                    min_samples_leaf=5, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=7)
    modelo.fit(X_tr, y_tr)

    # 3. Evaluación
    p_te = modelo.predict_proba(X_te)[:, 1]
    pred = (p_te >= UMBRAL_REVISION).astype(int)
    auc, ap = roc_auc_score(y_te, p_te), average_precision_score(y_te, p_te)
    cm = confusion_matrix(y_te, pred)
    print("=== Random Forest – detección de fraude ===")
    print(f"Operaciones de prueba: {len(y_te)} | fraudes reales: {int(y_te.sum())}")
    print(f"ROC-AUC: {auc:.3f} | PR-AUC: {ap:.3f}")
    print("Matriz de confusión (umbral 0.30):\n", cm)
    print(classification_report(y_te, pred, digits=3, target_names=["legítima", "fraude"]))

    # 4. Importancia de variables (explicabilidad para auditoría)
    imp = sorted(zip(VARIABLES, modelo.feature_importances_), key=lambda t: t[1])
    plt.figure(figsize=(7, 4.2))
    plt.barh([v for v, _ in imp], [i for _, i in imp], color="#3b6ea5")
    plt.xlabel("Importancia (reducción media de impureza)")
    plt.title("Random Forest: variables que más explican el fraude")
    plt.tight_layout(); plt.savefig("resultados/rf_importancia.png", dpi=200)

    # 5. Persistencia del modelo y ejemplo de respuesta para el ERP
    joblib.dump(modelo, "resultados/modelo_rf_fraude.joblib")
    nueva = {"monto_desembolso": 15000, "plazo_meses": 12, "antiguedad_cliente_meses": 2,
             "creditos_previos": 0, "score_buro": 480, "dias_apertura_cuenta": 3,
             "cambio_telefono_30d": 1, "abono_cuenta_tercero": 1, "hora_operacion": 23,
             "distancia_agencia_km": 35, "intentos_login_fallidos": 3,
             "dispositivo_nuevo": 1, "canal_app": 1}
    p = float(modelo.predict_proba(pd.DataFrame([nueva])[VARIABLES])[0, 1])
    respuesta = {"id_operacion": "OP-DEMO-001", "probabilidad_fraude": round(p, 3),
                 **nivel_riesgo(p), "modelo": "RF-v1"}
    print("Respuesta enviada al ERP:", json.dumps(respuesta, ensure_ascii=False))
    with open("resultados/metricas_rf.json", "w") as f:
        json.dump({"roc_auc": auc, "pr_auc": ap, "matriz": cm.tolist(),
                   "importancias": dict(imp), "ejemplo": respuesta}, f, indent=2)


if __name__ == "__main__":
    main()
