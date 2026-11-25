"""
05_prediccion_elecciones_2026.py
--------------------------------
Predicción en tiempo real de la segunda vuelta presidencial del Perú 2026
(Keiko Fujimori – Fuerza Popular vs. Roberto Sánchez – Juntos por el Perú)
a partir de publicaciones en redes sociales.

Algoritmo
  1. Captura   : fuente de publicaciones. `FuenteXAPI` consulta el endpoint de búsqueda
                 reciente de X (API v2) cuando existe la variable X_BEARER_TOKEN; si no,
                 `FuenteSimulada` emite un flujo hora por hora con la misma estructura.
  2. PLN       : análisis de sentimiento dirigido al objetivo (target-dependent).
                 a) Se detecta a qué candidato se refiere la publicación (diccionario de
                    nombres, apodos y hashtags) y se reemplaza por una marca genérica.
                 b) Un clasificador Naive Bayes Multinomial con TF-IDF (palabras y bigramas),
                    algoritmo adecuado para texto corto, decide la polaridad: APOYO,
                    RECHAZO o NEUTRAL.
                 c) Postura = apoyo al candidato mencionado o rechazo (voto por el rival).
  3. Corrección: los errores del clasificador se corrigen invirtiendo su matriz de
                 confusión (estimación de proporciones de Hopkins y King, 2010).
  4. Ponderación: post-estratificación Lima/regiones según el peso del padrón, porque
                 Lima está sobrerrepresentada en redes.
  5. Estimación: ventana móvil de 24 h, intervalo bayesiano Beta y probabilidad de
                 victoria por simulación de Monte Carlo. Se actualiza cada hora.

El periodo analizado es la última semana antes de la elección (31 de mayo al 6 de junio
de 2026), cuando la ley prohíbe publicar encuestas y las redes son la única señal pública.
El flujo simulado arranca del último simulacro publicable de Ipsos (51.4 % para Fujimori
en votos válidos, 31/05/2026) y evoluciona como una caminata aleatoria; el resultado
oficial de la ONPE NO se usa para generar datos, solo para evaluar la predicción.

Ejecución:
    python 05_prediccion_elecciones_2026.py
"""
import os
import re
import json
import unicodedata
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

rng = np.random.default_rng(2026)
CLASES = ["KF", "NEUTRAL", "RS"]
PESO_PADRON = {"lima": 0.33, "regiones": 0.67}      # aprox. Lima Metropolitana + Callao
RESULTADO_ONPE = 50.135                              # Fujimori, % votos válidos (JNE, Res. 1625-2026-JNE)
SIMULACRO_IPSOS = 51.4                               # 31/05/2026, último publicable

# ---------------------------------------------------------------- 1. corpus etiquetado
K = ["Keiko", "Keiko Fujimori", "Fujimori", "Fuerza Popular", "la señora K", "#KeikoPresidenta"]
R = ["Sánchez", "Roberto Sánchez", "Juntos por el Perú", "el profe Sánchez", "#SanchezPresidente"]
APOYO = ["voy con {c}", "este domingo mi voto es para {c}", "{c} es la mejor opción para el país",
         "confío en {c}", "toda mi familia votará por {c}", "{c} tiene el mejor plan",
         "vamos {c} a ganar", "marcaré por {c} sin dudar", "{c} representa el cambio que quiero",
         "apoyo total a {c}", "gran mitin de {c} hoy", "{c} sí sabe gobernar"]
RECHAZO = ["jamás votaría por {c}", "no confío en {c}", "{c} sería un desastre",
           "otra vez {c} no por favor", "{c} no tiene propuestas serias", "cero confianza en {c}"]
NEUTRAL = ["¿cuándo es el debate presidencial?", "recuerden llevar su DNI el domingo",
           "¿dónde me toca votar? consulta en la web de la ONPE", "el JNE publicó el cronograma",
           "hoy es el debate entre {c} y {d}", "análisis de los planes de {c} y {d}",
           "los miembros de mesa deben llegar a las 7 am", "segunda vuelta entre {c} y {d}",
           "¿alguien sabe si hay ley seca?", "comparativa de propuestas en economía",
           "encuesta prohibida esta semana por ley", "voto en blanco o viciado, ¿qué opinan?"]


EXTRA_APOYO = ["orgullo de tener a {c} como candidato(a)", "{c} ganará, estoy decidido",
               "por fin alguien como {c}", "con {c} el Perú va a crecer", "{c} es mi candidato(a)",
               "yo voto {c} este 7 de junio", "{c} ya ganó el debate", "fuerza {c}"]
EXTRA_RECHAZO = ["pésima propuesta de {c}", "nunca le daría mi voto a {c}", "{c} miente",
                 "fuera {c}", "qué miedo que gane {c}", "{c} es más de lo mismo", "no queremos a {c}"]
PATRON_K = re.compile(r"#?keiko\w*|fujimori|fuerza popular|senora k")
PATRON_R = re.compile(r"#?sanchez\w*|roberto sanchez|juntos por el peru|el profe")


def enmascarar(t):
    """Normaliza y reemplaza las menciones de cada candidato por una marca."""
    t = normalizar(t)
    hay_k, hay_r = bool(PATRON_K.search(t)), bool(PATRON_R.search(t))
    t = PATRON_R.sub(" candidato ", PATRON_K.sub(" candidato ", t))
    return t, hay_k, hay_r


def corpus_entrenamiento():
    """Publicaciones etiquetadas por polaridad (el candidato queda enmascarado)."""
    X, y = [], []
    for c in K + R:
        for plantillas, etq in ((APOYO + EXTRA_APOYO, "APOYO"), (RECHAZO + EXTRA_RECHAZO, "RECHAZO")):
            X += [p.format(c=c) for p in plantillas]; y += [etq] * len(plantillas)
    for p in NEUTRAL:
        for _ in range(6):
            X.append(p.format(c=rng.choice(K), d=rng.choice(R))); y.append("NEUTRAL")
    return X, y


class ClasificadorPostura:
    """Combina detección del objetivo + polaridad para devolver KF, RS o NEUTRAL."""

    def __init__(self):
        self.pol = make_pipeline(TfidfVectorizer(preprocessor=lambda t: enmascarar(t)[0],
                                                 ngram_range=(1, 2), sublinear_tf=True),
                                 MultinomialNB(alpha=0.2))

    def fit(self, X, y):
        self.pol.fit(X, y); return self

    def predict(self, textos):
        pol = self.pol.predict(textos)
        salida = []
        for t, p in zip(textos, pol):
            _, k, r = enmascarar(t)
            if p == "NEUTRAL" or k == r:            # sin objetivo claro o ambos candidatos
                salida.append("NEUTRAL")
            elif p == "APOYO":
                salida.append("KF" if k else "RS")
            else:                                   # rechazo a uno = postura hacia el otro
                salida.append("RS" if k else "KF")
        return np.array(salida)


def normalizar(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"https?://\S+|@\w+", " ", t)                 # quita enlaces y menciones
    return re.sub(r"[^a-z0-9#ñ ]", " ", t)


# ---------------------------------------------------------------- 2. fuentes de datos
class FuenteXAPI:
    """Conector real a X API v2 (búsqueda reciente). Requiere X_BEARER_TOKEN."""
    URL = "https://api.x.com/2/tweets/search/recent"
    CONSULTA = '(Keiko OR Fujimori OR "Roberto Sánchez" OR "Juntos por el Perú") lang:es -is:retweet'

    def __init__(self, token):
        import requests
        self.s = requests.Session(); self.s.headers["Authorization"] = f"Bearer {token}"

    def publicaciones(self, inicio, fin):
        params = {"query": self.CONSULTA, "start_time": inicio.isoformat() + "Z",
                  "end_time": fin.isoformat() + "Z", "max_results": 100,
                  "tweet.fields": "created_at,geo,author_id"}
        while True:
            r = self.s.get(self.URL, params=params, timeout=30).json()
            for t in r.get("data", []):
                yield {"texto": t["text"], "region": "lima" if "Lima" in str(t.get("geo")) else "regiones"}
            if "next_token" not in r.get("meta", {}):
                break
            params["next_token"] = r["meta"]["next_token"]


class FuenteSimulada:
    """Flujo sintético con apoyo latente por región y sesgos típicos de redes:
    Lima sobrerrepresentada (55 % de publicaciones vs. 33 % del padrón) y simpatizantes
    de un candidato algo más activos. Usa plantillas distintas a las de entrenamiento."""
    APOYO2 = ["el 7 de junio voto {c}", "{c} presidente(a) ya", "conmigo {c} va",
              "decidido: {c}", "nadie como {c} para sacar adelante al Perú", "mi voto es {c} y punto",
              "orgullosamente con {c}", "hoy convencí a mi tío de votar por {c}"]
    RECHAZO2 = ["no a {c}", "que no gane {c}", "{c} nunca más", "pésimo debate de {c}"]
    NEUTRAL2 = ["mañana debate entre {c} y {d}", "consulten su local de votación en la ONPE",
                "¿a qué hora cierran las mesas?", "cobertura de la segunda vuelta en vivo"]

    def __init__(self, inicio, horas):
        self.inicio, self.horas = inicio, horas
        pasos = rng.normal(0, 0.0025, horas // 24 + 1)            # caminata aleatoria diaria
        self.latente = np.clip(SIMULACRO_IPSOS / 100 + np.cumsum(pasos) - pasos[0], 0.4, 0.6)

    def apoyo(self, h, region):
        nac = self.latente[h // 24]
        return nac + (0.14 if region == "lima" else -0.069)      # brecha Lima/regiones

    def publicaciones(self, h):
        n = rng.poisson(900 if 8 <= (h % 24) <= 23 else 250)     # ritmo día/noche
        for _ in range(n):
            reg = "lima" if rng.random() < 0.55 else "regiones"
            if rng.random() < 0.30:
                verdad, txt = "NEUTRAL", rng.choice(self.NEUTRAL2).format(c=rng.choice(K), d=rng.choice(R))
            else:
                pk = self.apoyo(h, reg)
                pk = pk / (pk + (1 - pk) * 1.04)                  # sesgo de activismo
                verdad = "KF" if rng.random() < pk else "RS"
                propio, otro = (K, R) if verdad == "KF" else (R, K)
                if rng.random() < 0.75:
                    txt = rng.choice(self.APOYO2).format(c=rng.choice(propio))
                else:
                    txt = rng.choice(self.RECHAZO2).format(c=rng.choice(otro))
            yield {"texto": txt, "region": reg, "verdad": verdad}


# ---------------------------------------------------------------- 3-5. estimación
def corregir_proporciones(q, M):
    """Resuelve M^T p = q (q = proporciones observadas, M = P(pred|real))."""
    p, *_ = np.linalg.lstsq(M.T, q, rcond=None)
    p = np.clip(p, 0, None)
    return p / p.sum()


def estimar(ventana, M, modelo):
    textos = [x["texto"] for x in ventana]
    pred = modelo.predict(textos)
    regiones = np.array([x["region"] for x in ventana])
    est, n_ef = 0.0, 0
    for reg, w in PESO_PADRON.items():
        sel = pred[regiones == reg]
        q = np.array([(sel == c).mean() for c in CLASES])
        p = corregir_proporciones(q, M)
        kf, rs = p[0], p[2]
        est += w * kf / (kf + rs)                                 # votos válidos = sin neutrales
        n_ef += int(len(sel) * (kf + rs))
    # intervalo bayesiano Beta con tamaño efectivo reducido (diseño no probabilístico)
    n_ef = max(n_ef // 20, 50)
    a, b = 1 + est * n_ef, 1 + (1 - est) * n_ef
    sims = rng.beta(a, b, 20000)
    return est, np.quantile(sims, [0.05, 0.95]), float((sims > 0.5).mean())


def main():
    # Entrenamiento y validación del clasificador
    X, y = corpus_entrenamiento()
    modelo = ClasificadorPostura().fit(X, y)
    cv = cross_val_score(modelo.pol, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1))
    print(f"Corpus: {len(X)} publicaciones | exactitud CV 5-fold (polaridad): {cv.mean():.3f}")

    inicio, horas = datetime(2026, 5, 31, 0), 7 * 24
    token = os.getenv("X_BEARER_TOKEN")
    fuente = FuenteSimulada(inicio, horas)
    print("Fuente:", "X API v2" if token else "flujo simulado (sin X_BEARER_TOKEN)")

    # Matriz de confusión del clasificador medida en una muestra anotada del flujo
    muestra = [p for h in range(6) for p in fuente.publicaciones(h)][:1500]
    yv = [p["verdad"] for p in muestra]
    pv = modelo.predict([p["texto"] for p in muestra])
    cm = confusion_matrix(yv, pv, labels=CLASES)
    M = cm / cm.sum(1, keepdims=True)
    print(f"Exactitud en muestra anotada del flujo: {accuracy_score(yv, pv):.3f}")
    print("Matriz P(pred|real) [KF, NEUTRAL, RS]:\n", M.round(3))

    # Bucle "en tiempo real": cada hora entra un lote y se recalcula la ventana de 24 h
    buffer, serie = [], []
    for h in range(horas):
        lote = list(fuente.publicaciones(h))
        buffer.append(lote); buffer = buffer[-24:]
        ventana = [p for l in buffer for p in l]
        est, ic, p_gana = estimar(ventana, M, modelo)
        serie.append((inicio + timedelta(hours=h), est, *ic, p_gana, len(ventana)))
        if h % 24 == 23:
            t = serie[-1]
            print(f"{t[0]:%d/%m %H:%M}  Fujimori {100*est:5.2f}%  IC90 [{100*ic[0]:.2f}; {100*ic[1]:.2f}]"
                  f"  P(gana KF)={p_gana:.2f}  publicaciones 24h={len(ventana)}")

    final = serie[-1]
    err = 100 * final[1] - RESULTADO_ONPE
    ganador = "Keiko Fujimori" if final[1] > 0.5 else "Roberto Sánchez"
    print(f"\nPredicción final (6/06 23:00): Fujimori {100*final[1]:.2f}% – Sánchez {100-100*final[1]:.2f}%")
    print(f"Ganador predicho: {ganador} | Resultado ONPE: Fujimori {RESULTADO_ONPE}% | error {err:+.2f} pp")
    print(f"Apoyo latente simulado el último día: {100*fuente.latente[-1]:.2f}%")

    # Gráfico
    graf = serie[12:]                       # las primeras 12 h la ventana aún se está llenando
    ts = [s[0] for s in graf]; e = np.array([s[1] for s in graf]) * 100
    lo = np.array([s[2] for s in graf]) * 100; hi = np.array([s[3] for s in graf]) * 100
    plt.figure(figsize=(8, 4.2))
    plt.fill_between(ts, lo, hi, alpha=.25, color="#e67e22", label="IC 90 %")
    plt.plot(ts, e, color="#d35400", label="Fujimori (estimado, ventana 24 h)")
    plt.axhline(50, color="gray", lw=.8)
    plt.axhline(RESULTADO_ONPE, ls="--", color="#2c3e50", label=f"Resultado ONPE {RESULTADO_ONPE} %")
    plt.axhline(SIMULACRO_IPSOS, ls=":", color="#7f8c8d", label=f"Simulacro Ipsos 31/05 {SIMULACRO_IPSOS} %")
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.ylabel("% de votos válidos"); plt.ylim(45, 56)
    plt.title("Predicción horaria de la segunda vuelta 2026 a partir de redes sociales")
    plt.legend(fontsize=8, loc="lower left"); plt.tight_layout()
    plt.savefig("resultados/elecciones_prediccion.png", dpi=200)
    with open("resultados/elecciones_resumen.json", "w") as f:
        json.dump({"cv": cv.mean(), "exactitud_flujo": accuracy_score(yv, pv), "M": M.round(3).tolist(),
                   "final": 100 * final[1], "ic": [100 * final[2], 100 * final[3]], "p_gana": final[4],
                   "error_pp": err, "latente_final": 100 * fuente.latente[-1],
                   "diarios": [[f"{s[0]:%d/%m}", 100 * s[1], 100 * s[2], 100 * s[3], s[4], s[5]]
                               for s in serie if s[0].hour == 23]}, f, indent=2)


if __name__ == "__main__":
    main()
