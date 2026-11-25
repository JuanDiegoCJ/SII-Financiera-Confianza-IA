"""
datos_financiera.py
-------------------
Generador de datos sintéticos de operaciones de crédito de "Financiera Confianza Perú".

Los datos son ficticios. Se generan con reglas de negocio que reproducen patrones
conocidos de fraude en microfinanzas (suplantación de identidad, desembolso a cuentas
de terceros, cambios recientes de teléfono, operaciones fuera de horario, etc.), de modo
que los modelos de Machine Learning y Deep Learning puedan entrenarse sin exponer
datos personales reales (Ley N.° 29733).

Uso:
    from datos_financiera import generar_operaciones
    df = generar_operaciones(n=20000, semilla=42)
"""
import numpy as np
import pandas as pd

VARIABLES = [
    "monto_desembolso",          # soles
    "plazo_meses",
    "antiguedad_cliente_meses",
    "creditos_previos",
    "score_buro",                # 0 a 1000
    "dias_apertura_cuenta",      # días desde que se abrió la cuenta de abono
    "cambio_telefono_30d",       # 1 si el cliente cambió de celular en los últimos 30 días
    "abono_cuenta_tercero",      # 1 si el desembolso va a una cuenta que no es del titular
    "hora_operacion",            # 0 a 23
    "distancia_agencia_km",      # distancia entre el domicilio y la agencia/punto de la operación
    "intentos_login_fallidos",   # en las últimas 24 h (canal digital)
    "dispositivo_nuevo",         # 1 si el dispositivo no se había visto antes
    "canal_app",                 # 1 si la solicitud entró por la app móvil
]


def generar_operaciones(n: int = 20000, semilla: int = 42) -> pd.DataFrame:
    """Devuelve un DataFrame con n operaciones y la etiqueta 'fraude' (0/1)."""
    rng = np.random.default_rng(semilla)

    antiguedad = rng.gamma(2.0, 18, n).round()
    creditos_previos = rng.poisson(np.clip(antiguedad / 14, 0.2, None))
    score = np.clip(rng.normal(620, 110, n) + creditos_previos * 8, 250, 950).round()
    monto = np.round(np.exp(rng.normal(8.4, 0.7, n)), -1)          # mediana aprox. S/ 4 400
    plazo = rng.choice([6, 9, 12, 18, 24, 36], n, p=[.12, .15, .30, .18, .17, .08])
    canal_app = rng.binomial(1, 0.35, n)
    dias_cuenta = rng.gamma(1.6, 300, n).round()
    cambio_tel = rng.binomial(1, 0.06, n)
    tercero = rng.binomial(1, 0.04, n)
    hora = np.clip(rng.normal(12.5, 3.2, n), 0, 23).round()
    distancia = np.round(rng.gamma(1.5, 4, n), 1)
    login_fallidos = rng.poisson(0.3, n) * canal_app
    disp_nuevo = rng.binomial(1, 0.08, n) * canal_app

    # Puntaje latente de fraude (reglas inspiradas en tipologías del sector)
    z = (-4.6
         + 2.6 * cambio_tel
         + 3.0 * tercero
         + 2.0 * (dias_cuenta < 15)
         + 1.7 * ((hora < 6) | (hora > 21))
         + 0.07 * np.minimum(distancia, 40)
         + 0.8 * np.minimum(login_fallidos, 5)
         + 2.0 * disp_nuevo
         + 0.8 * (monto > 12000)
         - 0.025 * np.minimum(antiguedad, 60)
         - 0.004 * (score - 600)
         + 0.6 * (creditos_previos == 0))
    prob = 1 / (1 + np.exp(-z))
    fraude = rng.binomial(1, prob)

    df = pd.DataFrame({
        "id_operacion": [f"OP{100000 + i}" for i in range(n)],
        "monto_desembolso": monto, "plazo_meses": plazo,
        "antiguedad_cliente_meses": antiguedad, "creditos_previos": creditos_previos,
        "score_buro": score, "dias_apertura_cuenta": dias_cuenta,
        "cambio_telefono_30d": cambio_tel, "abono_cuenta_tercero": tercero,
        "hora_operacion": hora, "distancia_agencia_km": distancia,
        "intentos_login_fallidos": login_fallidos, "dispositivo_nuevo": disp_nuevo,
        "canal_app": canal_app, "fraude": fraude,
    })
    return df


if __name__ == "__main__":
    d = generar_operaciones()
    d.to_csv("datos/operaciones_credito.csv", index=False)
    print(d.head())
    print(f"Operaciones: {len(d)} | tasa de fraude: {d.fraude.mean():.2%}")
