"""
04_chatbot_pln.py
-----------------
COMPONENTE 3: Asistente conversacional (chatbot) con PLN integrado a la base de datos.

Arquitectura:
  canal (WhatsApp / web / app)  ->  procesar_mensaje()  ->  clasificador de intención (PLN)
  -> extracción de entidades (regex) -> consulta SQL a la réplica de lectura del core/CRM
  -> respuesta en lenguaje natural  (+ registro de la conversación para auditoría)

PLN: normalización del texto (minúsculas, sin tildes), vectorización TF-IDF de
n-gramas de caracteres (tolera errores de tipeo frecuentes en mensajería) y un
clasificador de regresión logística. Si la confianza es menor a 0.45 se deriva a un
asesor humano. Las consultas de datos personales exigen validar DNI y un código OTP.

Ejecución:
    python 04_chatbot_pln.py          # corre una conversación de demostración
    python 04_chatbot_pln.py --chat   # modo interactivo en consola
"""
import re
import sys
import sqlite3
import unicodedata
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

UMBRAL_CONFIANZA = 0.45

# ---------------------------------------------------------------- datos de entrenamiento
EJEMPLOS = {
    "saludo": ["hola", "buenos dias", "buenas tardes", "hola confianza", "que tal", "buenas noches"],
    "consultar_cuota": ["cuanto debo pagar este mes", "cual es mi proxima cuota", "cuando vence mi cuota",
                        "fecha de pago de mi credito", "monto de mi cuota", "cuanto pago y cuando",
                        "quiero saber mi cuota", "mi cuota de este mes"],
    "consultar_saldo": ["cuanto me falta pagar", "saldo de mi prestamo", "deuda total de mi credito",
                        "cuanto debo en total", "saldo pendiente", "quiero cancelar todo cuanto es"],
    "estado_solicitud": ["como va mi solicitud", "ya aprobaron mi credito", "estado de mi solicitud",
                         "mi prestamo fue aprobado", "en que va mi tramite", "cuando me desembolsan"],
    "reportar_fraude": ["me robaron el celular", "no reconozco un desembolso", "alguien saco un credito a mi nombre",
                        "creo que me estafaron", "hay un prestamo que yo no pedi", "me llamaron pidiendo mi clave",
                        "suplantaron mi identidad", "fraude"],
    "agencias": ["donde queda la agencia mas cercana", "horario de atencion", "direccion de la agencia",
                 "agencias en trujillo", "a que hora abren"],
    "hablar_asesor": ["quiero hablar con una persona", "comunicame con un asesor", "necesito un humano",
                      "asesor por favor", "no me ayudas quiero un agente"],
    "despedida": ["gracias", "chau", "eso es todo", "muchas gracias adios", "hasta luego"],
}


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]", " ", t)


def entrenar_clasificador():
    X = [normalizar(f) for frases in EJEMPLOS.values() for f in frases]
    y = [i for i, frases in EJEMPLOS.items() for _ in frases]
    modelo = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
                           LogisticRegression(max_iter=2000, C=8))
    return modelo.fit(X, y)


# ---------------------------------------------------------------- base de datos (réplica)
def crear_bd(ruta=":memory:"):
    bd = sqlite3.connect(ruta)
    bd.executescript("""
    CREATE TABLE cliente(id INTEGER PRIMARY KEY, dni TEXT UNIQUE, nombre TEXT, celular TEXT, otp TEXT);
    CREATE TABLE credito(id INTEGER PRIMARY KEY, id_cliente INT, producto TEXT, saldo REAL, estado TEXT);
    CREATE TABLE cuota(id_credito INT, nro INT, vencimiento TEXT, monto REAL, pagada INT);
    CREATE TABLE solicitud(nro TEXT PRIMARY KEY, id_cliente INT, estado TEXT, actualizado TEXT);
    CREATE TABLE agencia(ciudad TEXT, direccion TEXT, horario TEXT);
    CREATE TABLE ticket_fraude(id INTEGER PRIMARY KEY AUTOINCREMENT, id_cliente INT, canal TEXT,
                               detalle TEXT, fecha TEXT, estado TEXT);
    CREATE TABLE conversacion(fecha TEXT, canal TEXT, usuario TEXT, mensaje TEXT,
                              intencion TEXT, confianza REAL, respuesta TEXT);
    INSERT INTO cliente VALUES (1,'45879632','María Quispe','987654321','4821'),
                               (2,'70125896','Luis Rojas','912345678','1937');
    INSERT INTO credito VALUES (10,1,'Crédito Emprendedor',6840.50,'VIGENTE'),
                               (11,2,'Palabra de Mujer',1200.00,'VIGENTE');
    INSERT INTO cuota VALUES (10,7,'2026-12-05',612.30,0),(10,8,'2027-01-05',612.30,0),
                             (11,3,'2026-11-30',215.00,0);
    INSERT INTO solicitud VALUES ('SOL-20262871',1,'EN EVALUACIÓN DE CAMPO','2026-11-23');
    INSERT INTO agencia VALUES ('trujillo','Av. España 1234, Trujillo','L-V 9:00-18:00, S 9:00-13:00'),
                               ('lima','Av. Javier Prado 456, San Isidro','L-V 9:00-18:00');
    """)
    return bd


# ---------------------------------------------------------------- motor de diálogo
class Asistente:
    PRIVADAS = {"consultar_cuota", "consultar_saldo", "estado_solicitud"}

    def __init__(self, bd):
        self.bd, self.nlu, self.sesiones = bd, entrenar_clasificador(), {}

    def _sesion(self, usuario):
        return self.sesiones.setdefault(usuario, {"cliente": None, "pendiente": None, "paso": None})

    def procesar_mensaje(self, canal: str, usuario: str, texto: str) -> str:
        s = self._sesion(usuario)
        if s["paso"]:                                    # flujo de autenticación en curso
            resp, intencion, conf = self._autenticar(s, texto), "autenticacion", 1.0
        else:
            proba = self.nlu.predict_proba([normalizar(texto)])[0]
            k = proba.argmax(); intencion, conf = self.nlu.classes_[k], float(proba[k])
            if conf < UMBRAL_CONFIANZA:
                intencion, resp = "no_entendido", ("No estoy seguro de haber entendido. "
                                                   "¿Deseas que te comunique con un asesor?")
            elif intencion in self.PRIVADAS and not s["cliente"]:
                s["pendiente"], s["paso"] = intencion, "dni"
                resp = "Para proteger tus datos necesito validar tu identidad. Escribe tu DNI."
            else:
                resp = self._responder(intencion, s, texto, canal)
        self.bd.execute("INSERT INTO conversacion VALUES (?,?,?,?,?,?,?)",
                        (datetime.now().isoformat(timespec="seconds"), canal, usuario,
                         texto, intencion, round(conf, 3), resp))
        return resp

    def _autenticar(self, s, texto):
        if s["paso"] == "dni":
            dni = re.search(r"\b\d{8}\b", texto)
            fila = dni and self.bd.execute("SELECT id, celular FROM cliente WHERE dni=?",
                                           (dni.group(),)).fetchone()
            if not fila:
                return "No encuentro ese DNI. Verifícalo, por favor (8 dígitos)."
            s["candidato"], s["paso"] = fila[0], "otp"
            return f"Te envié un código por SMS al celular terminado en {fila[1][-3:]}. Escríbelo aquí."
        otp = self.bd.execute("SELECT otp FROM cliente WHERE id=?", (s["candidato"],)).fetchone()[0]
        if texto.strip() != otp:
            return "El código no coincide. Inténtalo de nuevo."
        s["cliente"], s["paso"] = s.pop("candidato"), None
        return "Identidad validada. " + self._responder(s.pop("pendiente"), s, "", "")

    def _responder(self, intencion, s, texto, canal):
        c, q = s["cliente"], self.bd.execute
        if intencion == "saludo":
            return "¡Hola! Soy el asistente de Financiera Confianza. ¿En qué te ayudo?"
        if intencion == "consultar_cuota":
            r = q("""SELECT cu.nro, cu.vencimiento, cu.monto, cr.producto FROM cuota cu
                     JOIN credito cr ON cr.id=cu.id_credito
                     WHERE cr.id_cliente=? AND cu.pagada=0 ORDER BY cu.vencimiento LIMIT 1""", (c,)).fetchone()
            return f"Tu cuota N.° {r[0]} de {r[3]} es de S/ {r[2]:,.2f} y vence el {r[1]}."
        if intencion == "consultar_saldo":
            r = q("SELECT producto, saldo FROM credito WHERE id_cliente=? AND estado='VIGENTE'", (c,)).fetchone()
            return f"El saldo de capital de tu {r[0]} es S/ {r[1]:,.2f}."
        if intencion == "estado_solicitud":
            r = q("SELECT nro, estado, actualizado FROM solicitud WHERE id_cliente=?", (c,)).fetchone()
            return (f"Tu solicitud {r[0]} está: {r[1]} (actualizado el {r[2]})." if r
                    else "No tienes solicitudes en trámite.")
        if intencion == "reportar_fraude":
            q("INSERT INTO ticket_fraude(id_cliente,canal,detalle,fecha,estado) VALUES (?,?,?,?,?)",
              (c, canal, texto, datetime.now().isoformat(timespec="seconds"), "ABIERTO"))
            nro = q("SELECT max(id) FROM ticket_fraude").fetchone()[0]
            return (f"Registré tu reporte con el ticket F-{nro:05d} y lo derivé a la Unidad de Prevención "
                    "de Fraudes, que te llamará en los próximos minutos. Nunca te pediremos tu clave.")
        if intencion == "agencias":
            ciudad = next((x for x in ("trujillo", "lima") if x in normalizar(texto)), "trujillo")
            r = q("SELECT direccion, horario FROM agencia WHERE ciudad=?", (ciudad,)).fetchone()
            return f"Agencia {ciudad.title()}: {r[0]}. Horario: {r[1]}."
        if intencion == "hablar_asesor":
            return "Te comunico con un asesor. El tiempo de espera estimado es de 3 minutos."
        return "¡Gracias por escribirnos! Que tengas un buen día."


def demo():
    bot = Asistente(crear_bd())
    guion = [("whatsapp", "51987654321", "Hola buenas tardes"),
             ("whatsapp", "51987654321", "cuanto debo pagr este mes?"),
             ("whatsapp", "51987654321", "mi dni es 45879632"),
             ("whatsapp", "51987654321", "4821"),
             ("whatsapp", "51987654321", "y como va mi solicitud?"),
             ("whatsapp", "51987654321", "no reconozco un desembolso que me llegó"),
             ("web", "visitante-88", "a que hora abren en trujillo"),
             ("web", "visitante-88", "el clima de mañana"),
             ("whatsapp", "51987654321", "gracias")]
    for canal, usuario, texto in guion:
        print(f"[{canal}] Usuario: {texto}")
        print(f"[{canal}] Bot    : {bot.procesar_mensaje(canal, usuario, texto)}\n")
    print("Registro de auditoría:")
    for fila in bot.bd.execute("SELECT canal, intencion, confianza FROM conversacion"):
        print("  ", fila)


if __name__ == "__main__":
    if "--chat" in sys.argv:
        bot = Asistente(crear_bd())
        while (t := input("Tú: ")).strip().lower() not in ("salir", "exit"):
            print("Bot:", bot.procesar_mensaje("consola", "local", t))
    else:
        demo()
