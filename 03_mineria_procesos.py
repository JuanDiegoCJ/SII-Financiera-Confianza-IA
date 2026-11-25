"""
03_mineria_procesos.py
----------------------
COMPONENTE 1: Minería de procesos sobre el ciclo de crédito (solicitud -> desembolso).

Aplica las tres técnicas de la minería de procesos (van der Aalst, 2016):
  1. Descubrimiento : construye el grafo de seguimiento directo (Directly-Follows Graph)
                      a partir del log de eventos que exporta el ERP/CRM.
  2. Conformidad    : compara cada caso con el modelo de referencia aprobado por la
                      Gerencia de Riesgos y calcula su fitness y sus desviaciones.
  3. Mejora         : mide tiempos de espera entre actividades (cuellos de botella) y
                      cruza las desviaciones con los casos que terminaron en fraude.

El log es sintético (formato XES simplificado: caso, actividad, marca de tiempo, recurso).

Ejecución:
    python 03_mineria_procesos.py
"""
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import subprocess

rng = np.random.default_rng(11)

# Modelo de referencia (camino feliz y transiciones permitidas)
REFERENCIA = ["Registrar solicitud", "Validar identidad (RENIEC/biometría)",
              "Consultar buró de crédito", "Evaluar en campo", "Aprobar en comité",
              "Firmar contrato", "Desembolsar"]
PERMITIDAS = set(zip(REFERENCIA[:-1], REFERENCIA[1:])) | {
    ("Evaluar en campo", "Solicitar documentos"), ("Solicitar documentos", "Evaluar en campo"),
    ("Aprobar en comité", "Rechazar"), ("Consultar buró de crédito", "Rechazar")}
OBLIGATORIAS = {"Validar identidad (RENIEC/biometría)", "Aprobar en comité"}

# Duración media (horas) de espera antes de cada actividad
ESPERA_H = {"Validar identidad (RENIEC/biometría)": 2, "Consultar buró de crédito": 1,
            "Evaluar en campo": 30, "Solicitar documentos": 6, "Aprobar en comité": 20,
            "Firmar contrato": 8, "Desembolsar": 4, "Rechazar": 3}
ASESORES = [f"ASE{i:02d}" for i in range(1, 21)]


def generar_log(n_casos=3000):
    filas, fraude = [], {}
    for c in range(n_casos):
        caso = f"SOL-{20260001 + c}"
        asesor = rng.choice(ASESORES)
        u = rng.random()
        if u < 0.70:                                            # camino feliz
            traza = REFERENCIA.copy()
        elif u < 0.82:                                          # reproceso por documentos
            traza = REFERENCIA[:4] + ["Solicitar documentos", "Evaluar en campo"] + REFERENCIA[4:]
        elif u < 0.90:                                          # rechazo
            traza = REFERENCIA[:5] + ["Rechazar"]
        elif u < 0.95:                                          # desviación: sin comité
            traza = [a for a in REFERENCIA if a != "Aprobar en comité"]
        else:                                                   # desviación: sin validar identidad
            traza = [a for a in REFERENCIA if a != "Validar identidad (RENIEC/biometría)"]
        # dos asesores concentran desviaciones (patrón que la minería debe revelar)
        if asesor in ("ASE07", "ASE15") and rng.random() < 0.25:
            traza = [a for a in REFERENCIA if a not in OBLIGATORIAS]
        t = datetime(2026, 3, 1, 8) + timedelta(hours=float(rng.uniform(0, 24 * 120)))
        for act in traza:
            if act != traza[0]:
                factor = 2.2 if act == "Aprobar en comité" and t.weekday() >= 4 else 1.0
                t += timedelta(hours=float(rng.exponential(ESPERA_H[act] * factor)))
            filas.append((caso, act, t, asesor))
        salto = OBLIGATORIAS - set(traza)
        fraude[caso] = int(rng.random() < (0.22 if salto else 0.012)) if "Desembolsar" in traza else 0
    log = pd.DataFrame(filas, columns=["caso", "actividad", "timestamp", "recurso"])
    return log, fraude


def descubrir(log):
    """Grafo de seguimiento directo: frecuencias y tiempo medio entre actividades."""
    freq, tiempos, variantes = Counter(), defaultdict(list), Counter()
    for caso, g in log.sort_values("timestamp").groupby("caso"):
        acts, ts = g.actividad.tolist(), g.timestamp.tolist()
        variantes[tuple(acts)] += 1
        freq[("INICIO", acts[0])] += 1; freq[(acts[-1], "FIN")] += 1
        for a, b, t1, t2 in zip(acts, acts[1:], ts, ts[1:]):
            freq[(a, b)] += 1
            tiempos[(a, b)].append((t2 - t1).total_seconds() / 3600)
    return freq, {k: float(np.mean(v)) for k, v in tiempos.items()}, variantes


def dibujar_dfg(freq, tmed, ruta):
    """Escribe el grafo en lenguaje DOT y lo renderiza con Graphviz (comando dot)."""
    mx = max(freq.values())
    lineas = ['digraph DFG {', 'rankdir=LR; dpi=200; nodesep=0.35; ranksep=0.45;',
              'node [shape=box style="rounded,filled" fillcolor="#eef3fa" fontname=Helvetica fontsize=11];',
              'edge [fontname=Helvetica fontsize=9];',
              'INICIO [shape=circle fillcolor="#d5f5e3" label=""];',
              'FIN [shape=doublecircle fillcolor="#fadbd8" label=""];']
    import textwrap
    for act in {x for par in freq for x in par} - {"INICIO", "FIN"}:
        etiqueta_nodo = "\\n".join(textwrap.wrap(act, 14, break_long_words=False))     # salto de línea en DOT
        lineas.append(f'"{act}" [label="{etiqueta_nodo}"];')
    for (a, b), f in freq.items():
        desviada = a != "INICIO" and b != "FIN" and (a, b) not in PERMITIDAS
        etiqueta = f"{f}" + (f"\\n{tmed[(a, b)]:.1f} h" if (a, b) in tmed else "")
        color = "#c0392b" if desviada else "#34495e"
        lineas.append(f'"{a}" -> "{b}" [label="{etiqueta}" penwidth={0.6 + 3.5 * f / mx:.2f} '
                      f'color="{color}" fontcolor="{color}"];')
    lineas.append("}")
    with open(ruta + ".dot", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))
    subprocess.run(["dot", "-Tpng", ruta + ".dot", "-o", ruta + ".png"], check=True)


def conformidad(log):
    """Fitness por caso = transiciones permitidas / transiciones observadas,
    penalizado por cada actividad obligatoria omitida."""
    res = []
    for caso, g in log.sort_values("timestamp").groupby("caso"):
        acts = g.actividad.tolist()
        trans = list(zip(acts, acts[1:]))
        ok = sum(t in PERMITIDAS for t in trans)
        omitidas = sorted(OBLIGATORIAS - set(acts)) if "Desembolsar" in acts else []
        fitness = ok / len(trans) * (1 - 0.25 * len(omitidas))
        res.append({"caso": caso, "recurso": g.recurso.iloc[0], "fitness": round(fitness, 3),
                    "omitidas": "; ".join(omitidas), "conforme": not omitidas and ok == len(trans)})
    return pd.DataFrame(res)


def main():
    log, fraude = generar_log()
    log.to_csv("datos/log_eventos_credito.csv", index=False)
    print(f"Log: {log.caso.nunique()} casos, {len(log)} eventos, {log.actividad.nunique()} actividades")

    # 1. Descubrimiento
    freq, tmed, variantes = descubrir(log)
    dibujar_dfg(freq, tmed, "resultados/dfg_proceso_credito")
    print("\nVariantes más frecuentes:")
    for v, n in variantes.most_common(5):
        print(f"  {n:5d} casos ({n / len(fraude):.1%}): " + " > ".join(v))

    # 2. Conformidad
    conf = conformidad(log)
    conf["fraude"] = conf.caso.map(fraude)
    print(f"\nFitness promedio del log: {conf.fitness.mean():.3f}")
    print(f"Casos conformes: {conf.conforme.mean():.1%}")
    dev = conf[conf.omitidas != ""]
    print(f"Casos desembolsados con controles omitidos: {len(dev)}")
    por_asesor = (conf.assign(desviado=conf.omitidas != "").groupby("recurso")["desviado"]
                  .mean().sort_values(ascending=False).head(5))
    print("Asesores con mayor tasa de desviación:\n", por_asesor.round(3).to_string())

    # 3. Mejora: cuellos de botella y relación desviación-fraude
    cuellos = sorted(((k, v) for k, v in tmed.items() if freq[k] > 100), key=lambda t: -t[1])[:3]
    print("\nCuellos de botella (espera media):")
    for (a, b), h in cuellos:
        print(f"  {a} -> {b}: {h:.1f} h")
    tasa_dev = dev.fraude.mean(); tasa_ok = conf[conf.omitidas == ""].fraude.mean()
    print(f"\nTasa de fraude con controles omitidos: {tasa_dev:.1%} | sin omisiones: {tasa_ok:.1%}")
    duracion = log.groupby("caso").timestamp.agg(lambda s: (s.max() - s.min()).total_seconds() / 86400)
    print(f"Tiempo de ciclo medio: {duracion.mean():.1f} días | P90: {duracion.quantile(.9):.1f} días")

    with open("resultados/mineria_resumen.json", "w") as f:
        json.dump({"casos": int(log.caso.nunique()), "eventos": len(log),
                   "fitness": float(conf.fitness.mean()), "conformes": float(conf.conforme.mean()),
                   "desviados": int(len(dev)), "fraude_desviados": float(tasa_dev),
                   "fraude_conformes": float(tasa_ok), "ciclo_medio_dias": float(duracion.mean()),
                   "ciclo_p90": float(duracion.quantile(.9)),
                   "cuellos": [[a, b, h] for (a, b), h in cuellos],
                   "top_asesores": por_asesor.round(3).to_dict(),
                   "variantes": [[" > ".join(v), n] for v, n in variantes.most_common(5)]},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
