"""
02_dl_red_neuronal_profunda.py
------------------------------
EJEMPLO DE DEEP LEARNING: red neuronal profunda (perceptrón multicapa) programada
desde cero con NumPy para el mismo problema de fraude en desembolsos.

Diferencia con el ejemplo de ML (Random Forest):
  * El modelo aprende representaciones internas en varias capas ocultas
    (13 -> 64 -> 32 -> 16 -> 1); cada capa combina las salidas de la anterior.
  * Se entrena con retropropagación del error (backpropagation) y el optimizador Adam.
  * Usa regularización propia del deep learning: dropout y parada temprana.

Se implementa sin TensorFlow/PyTorch para que se vea cada paso del algoritmo:
propagación hacia adelante, función de pérdida, gradientes y actualización de pesos.
La misma arquitectura en Keras sería:
    Dense(64,'relu') -> Dropout(0.2) -> Dense(32,'relu') -> Dropout(0.2)
    -> Dense(16,'relu') -> Dense(1,'sigmoid')

Ejecución:
    python 02_dl_red_neuronal_profunda.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from datos_financiera import generar_operaciones, VARIABLES

rng = np.random.default_rng(7)


class RedProfunda:
    """Perceptrón multicapa con ReLU, dropout, salida sigmoide y optimizador Adam."""

    def __init__(self, capas, dropout=0.2, lr=1e-3):
        self.W, self.b = [], []
        for n_in, n_out in zip(capas[:-1], capas[1:]):
            self.W.append(rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)))  # inicialización He
            self.b.append(np.zeros(n_out))
        self.dropout, self.lr = dropout, lr
        self.m = [np.zeros_like(p) for p in self.W + self.b]   # momentos de Adam
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    @staticmethod
    def _sigmoide(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def adelante(self, X, entrenando=False):
        """Propagación hacia adelante; guarda activaciones para la retropropagación."""
        self.cache = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            if i < len(self.W) - 1:                 # capas ocultas
                a = np.maximum(z, 0)                 # ReLU
                if entrenando and self.dropout > 0:  # dropout invertido
                    mask = (rng.random(a.shape) > self.dropout) / (1 - self.dropout)
                    a = a * mask
                    self.cache.append(("mask", mask))
                else:
                    self.cache.append(("mask", None))
            else:
                a = self._sigmoide(z)                # salida: probabilidad de fraude
            self.cache.append(a)
        return a.ravel()

    def atras(self, y, peso_pos):
        """Retropropagación con entropía cruzada binaria ponderada."""
        acts = [self.cache[0]] + [c for c in self.cache[1:] if not isinstance(c, tuple)]
        masks = [c[1] for c in self.cache[1:] if isinstance(c, tuple)]
        p = acts[-1].ravel()
        w = np.where(y == 1, peso_pos, 1.0)
        delta = ((p - y) * w / len(y))[:, None]     # dL/dz de la capa de salida
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(0)
            if i > 0:
                delta = delta @ self.W[i].T
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1]
                delta = delta * (acts[i] > 0)        # derivada de ReLU
        self._adam(gW + gb)

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = self.W + self.b
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)


def perdida(y, p, peso_pos):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    w = np.where(y == 1, peso_pos, 1.0)
    return float(np.mean(-w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def main():
    df = generar_operaciones(n=20000, semilla=42)
    X, y = df[VARIABLES].values.astype(float), df["fraude"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=7)
    esc = StandardScaler().fit(X_tr)                     # las redes necesitan datos escalados
    X_tr, X_va, X_te = esc.transform(X_tr), esc.transform(X_va), esc.transform(X_te)

    peso_pos = (1 - y_tr.mean()) / y_tr.mean()           # compensa el desbalance de clases
    red = RedProfunda([len(VARIABLES), 64, 32, 16, 1], dropout=0.2, lr=1e-3)
    hist_tr, hist_va, mejor, paciencia, mejores_pesos = [], [], np.inf, 0, None

    for epoca in range(1, 201):
        idx = rng.permutation(len(X_tr))
        for i in range(0, len(idx), 256):                # mini-lotes de 256
            lote = idx[i:i + 256]
            red.adelante(X_tr[lote], entrenando=True)
            red.atras(y_tr[lote], peso_pos)
        l_tr = perdida(y_tr, red.adelante(X_tr), peso_pos)
        l_va = perdida(y_va, red.adelante(X_va), peso_pos)
        hist_tr.append(l_tr); hist_va.append(l_va)
        if l_va < mejor - 1e-4:
            mejor, paciencia = l_va, 0
            mejores_pesos = ([w.copy() for w in red.W], [b.copy() for b in red.b], epoca)
        else:
            paciencia += 1
            if paciencia >= 15:                          # parada temprana
                break
    red.W, red.b, ep_best = mejores_pesos

    p_te = red.adelante(X_te)
    # Con pérdida ponderada la salida se desplaza hacia arriba; se fija el umbral que
    # alcanza un recall similar al del Random Forest para compararlos con justicia.
    umbral = float(np.quantile(p_te, 1 - 0.18))
    pred = (p_te >= umbral).astype(int)
    auc, ap = roc_auc_score(y_te, p_te), average_precision_score(y_te, p_te)
    cm = confusion_matrix(y_te, pred)
    print("=== Red neuronal profunda (NumPy) – detección de fraude ===")
    print(f"Arquitectura: 13-64-32-16-1 | épocas: {epoca} (mejor: {ep_best})")
    print(f"ROC-AUC: {auc:.3f} | PR-AUC: {ap:.3f} | umbral: {umbral:.3f}")
    print("Matriz de confusión:\n", cm)
    rec = cm[1, 1] / cm[1].sum(); prec = cm[1, 1] / cm[:, 1].sum()
    print(f"Recall fraude: {rec:.3f} | Precisión fraude: {prec:.3f}")

    plt.figure(figsize=(7, 4))
    plt.plot(hist_tr, label="entrenamiento"); plt.plot(hist_va, label="validación")
    plt.axvline(ep_best - 1, ls="--", c="gray", label="mejor época")
    plt.xlabel("Época"); plt.ylabel("Entropía cruzada ponderada")
    plt.title("Curva de aprendizaje de la red profunda"); plt.legend(); plt.tight_layout()
    plt.savefig("resultados/dl_curva_aprendizaje.png", dpi=200)
    with open("resultados/metricas_dl.json", "w") as f:
        json.dump({"roc_auc": auc, "pr_auc": ap, "matriz": cm.tolist(), "epocas": epoca,
                   "mejor_epoca": ep_best, "recall": rec, "precision": prec}, f, indent=2)


if __name__ == "__main__":
    main()
