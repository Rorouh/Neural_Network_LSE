# src/synth_engine.py
import json, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
from .rule_engine import translate_rule_based

random.seed(1234)

# -------------------------------
# Pequeño banco léxico/morfológico
# -------------------------------

@dataclass
class Noun:
    surface: str
    gender: str  # 'm' o 'f'
    number: str = "sg"  # 'sg' o 'pl'

NOUNS: List[Noun] = [
    Noun("libro", "m"), Noun("café", "m"),
    Noun("película", "f"), Noun("casa", "f"),
    Noun("zapatos", "m", "pl"), Noun("manzanas", "f", "pl"),
    Noun("tarea", "f"), Noun("agua", "f"), Noun("pan", "m"),
]

# Nombres propios variados (con y sin tilde) para entrenar el copiado + '#'
NAMES: List[str] = [
    "Ana","Miguel","Carlos","Lucía","José","María","Raúl","Íñigo","Óscar","Sofía",
    "Álvaro","Martín","Paula","Julián","Nuria","Camila","Noé","Irene","Ariadna","Aarón",
    "Juan","Elena","Pérez","García","López","Rubén","Tomás","Marcos"
]

PLACES: List[str] = [
    "en la tienda", "en casa", "en Madrid", "en el parque", "a la escuela", "en el trabajo","a Madrid", ""
]

TIMES: List[str] = [
    "Ayer", "Hoy", "Mañana", "Ahora", "Luego", "A las 9", "A las 7", "Por la noche", ""
]

SUBJECTS_1S: List[str] = ["Yo"]
SUBJECTS_2S: List[str] = ["Tú"]
SUBJECTS_3S: List[str] = ["Ana", "Miguel", "Lucía", "Carlos", "María", "Raúl"]

# Determinantes acordados
def det(typ: str, n: Noun) -> str:
    # typ: 'def', 'ind', 'zero'
    if typ == "zero":
        return ""
    table = {
        ("def", "m", "sg"): "el",
        ("def", "f", "sg"): "la",
        ("def", "m", "pl"): "los",
        ("def", "f", "pl"): "las",
        ("ind", "m", "sg"): "un",
        ("ind", "f", "sg"): "una",
        ("ind", "m", "pl"): "unos",
        ("ind", "f", "pl"): "unas",
    }
    return table[(typ, n.gender, n.number)]

# Verbitos con formas mínimas necesarias (+ jugar / llamarse)
VERBS: Dict[str, Dict[str, str]] = {
    "comprar": {"1s_pret":"compré","3s_pret":"compró","1s_pres":"compro","3s_pres":"compra","inf":"comprar"},
    "ver":     {"1s_pret":"vi","3s_pret":"vio","1s_pres":"veo","3s_pres":"ve","inf":"ver"},
    "querer":  {"1s_pres":"quiero","3s_pres":"quiere","1s_pret":"quise","3s_pret":"quiso","inf":"querer"},
    "ir":      {"1s_pres":"voy","3s_pres":"va","1s_pret":"fui","3s_pret":"fue","imp_2s":"ve","inf":"ir"},
    "vivir":   {"1s_pres":"vivo","2s_pres":"vives","3s_pres":"vive","1s_pret":"viví","3s_pret":"vivió","inf":"vivir"},
    "dar":     {"imp_2s":"da","inf":"dar"},
    "venir":   {"imp_2s":"ven","3s_pres":"viene","inf":"venir"},
    "hacer":   {"imp_2s":"haz","inf":"hacer"},
    "poner":   {"imp_2s":"pon","inf":"poner"},
    "decir":   {"imp_2s":"di", "inf":"decir"},
    "llover":  {"subj_3s":"llueva","inf":"llover"},
    # Reflexivo para WH de "llamarse"
    "llamar":  {"1s_pres_ref":"me llamo","2s_pres_ref":"te llamas","3s_pres_ref":"se llama","inf":"llamar"},
    # Nuevo: jugar (formas básicas)
    "jugar":   {"1s_pres":"juego","2s_pres":"juegas","3s_pres":"juega","inf":"jugar"},
}

WH_BANK = [
    ("Cómo",  "¿Cómo te llamas?", lambda: "¿Cómo te llamas?"),
    ("Dónde", "¿Dónde vives?",     lambda: "¿Dónde vives?"),
    ("Quién", "¿Quién viene?",     lambda: "¿Quién viene?"),
    ("Qué",   "¿Qué quieres?",     lambda: "¿Qué quieres?"),
]

# -------------------------------
# Utilidades
# -------------------------------

def _clean_spaces(s: str) -> str:
    s = s.replace(" ,", ",").replace(" .", ".")
    s = " ".join(s.split())
    # arreglos pequeños
    s = s.replace(" al la ", " a la ").replace(" al el ", " al ")
    return s

def _maybe(p: float) -> bool:
    return random.random() < p

def _choose(lst):
    return random.choice(lst)

# -------------------------------
# Generadores de oraciones
# -------------------------------

def gen_time_place_stmt() -> str:
    """
    Ej.: "Ayer compré un libro en la tienda."
    Elegimos sujeto (1s o 3s), tiempo y lugar; conjugamos acorde.
    """
    n = _choose(NOUNS)
    dtyp = _choose(["def","ind","zero"])
    det_str = det(dtyp, n)
    obj = (det_str + " " + n.surface).strip()

    time = _choose(TIMES)  # puede ser ""
    place = _choose(PLACES)

    # sujeto
    if _maybe(0.6):
        S = _choose(SUBJECTS_1S)
        v = _choose(["comprar","ver","ir","vivir"])
        # si hay "Ayer", favorece pretérito
        if time.lower() == "ayer":
            form = VERBS[v].get("1s_pret") or VERBS[v].get("1s_pres")
        else:
            form = VERBS[v].get("1s_pres") or VERBS[v].get("1s_pret")
    else:
        S = _choose(SUBJECTS_3S)
        v = _choose(["comprar","ver","ir","vivir"])
        form = VERBS[v].get("3s_pret") if time.lower() == "ayer" else VERBS[v].get("3s_pres")

    pieces = []
    if time: pieces.append(time)
    pieces.append(S)
    pieces.append(form) 
    if obj and v not in ("ir","vivir"): # para "ir", "vivir" no siempre meter objeto
        pieces.append(obj)
    if place:
        pieces.append(place)

    sent = " ".join(pieces) + "."
    return _clean_spaces(sent)

def gen_ir_ser_ambiguity() -> str:
    # 50% IR con 'a + lugar', 50% SER con predicativo
    if _maybe(0.5):
        name = _choose(NAMES)
        place = "a Madrid"
        return _clean_spaces(f"Ayer {name} fue {place}.")
    else:
        name = _choose(NAMES)
        pred = _choose(["doctor", "doctora", "profesor", "actriz"])
        return _clean_spaces(f"Ayer {name} fue {pred}.")

def gen_ve_ambiguity() -> str:
    # Imperativo de IR: "Ve a casa ahora."
    # Vs declarativa de VER: "Carlos ve la película en Madrid."
    if _maybe(0.5):
        return "Ve a casa ahora."
    else:
        name = _choose(SUBJECTS_3S)
        obj = "la película"
        place = "en Madrid"
        return _clean_spaces(f"Hoy {name} ve {obj} {place}.")


def gen_negation_stmt() -> str:
    """
    Ej.: "Yo no quiero café."
    """
    S = "Yo" if _maybe(0.7) else _choose(SUBJECTS_3S)
    n = _choose(NOUNS)
    obj = (det("ind", n) + " " + n.surface).strip()
    # usar "querer" presente
    vform = VERBS["querer"]["1s_pres"] if S == "Yo" else VERBS["querer"]["3s_pres"]
    sent = f"{S} no {vform} {obj}."
    return _clean_spaces(sent)

def gen_wh_question() -> str:
    """
    Varias WH con formas frecuentes:
    - ¿Cómo te llamas?
    - ¿Dónde vives?
    - ¿Quién viene?
    - ¿Qué quieres?
    """
    tag, example, build = _choose(WH_BANK)
    return build()

def gen_imperative() -> str:
    """
    Imperativos típicos:
    - "Ven aquí ahora mismo."
    - "Dámelo, por favor."
    - Alguno irregular: "Pon eso ahí.", "Haz la tarea."
    """
    opt = random.randint(0, 3)
    if opt == 0:
        return "Ven aquí ahora mismo."
    elif opt == 1:
        # enclítico clásico con acento
        return "Dámelo, por favor."
    elif opt == 2:
        return "Haz la tarea."
    else:
        return "Pon eso ahí."

def gen_ojala() -> str:
    """
    Oraciones desiderativas con 'Ojalá' + subjuntivo 3a pers. + (opcional) tiempo:
    """
    choices = [
        "Ojalá llueva mañana.",
        "Ojalá venga Ana.",
        "Ojalá haya café.",
        "Ojalá pueda venir Miguel mañana.",
    ]
    return _clean_spaces(_choose(choices))

def gen_jugar_futbol() -> str:
    """
    Cobertura de 'jugar al fútbol' con sujetos 1s/2s/3s y lugar/tiempo opcional.
    """
    time = _choose(TIMES)
    place = _choose(PLACES)
    who_type = random.choice(["1s","2s","3s"])
    if who_type == "1s":
        S = "Yo"
        vform = VERBS["jugar"]["1s_pres"]  # juego
    elif who_type == "2s":
        S = "Tú"
        vform = VERBS["jugar"]["2s_pres"]  # juegas
    else:
        S = _choose(NAMES)
        vform = VERBS["jugar"]["3s_pres"]  # juega

    pieces = []
    if time: pieces.append(time)
    # 30% quitamos el sujeto explícito en 1s para variar (e.g., "Juego al fútbol.")
    if not (who_type == "1s" and _maybe(0.3)):
        pieces.append(S)
    pieces.append(vform)
    pieces.append("al fútbol")
    if place: pieces.append(place)
    return _clean_spaces(" ".join(pieces) + ".")

def gen_llamarse() -> str:
    """
    Frases de 'llamarse' (afirmativa y 2a persona típica).
    """
    if _maybe(0.5):
        # afirmativa
        name = _choose(NAMES)
        return _clean_spaces(f"Yo me llamo {name}.")
    else:
        # variante sin 'Yo'
        name = _choose(NAMES)
        return _clean_spaces(f"Me llamo {name}.")

# -------------------------------
# Bucle de generación
# -------------------------------

GENS: List[Tuple[str, Callable[[], str]]] = [
    ("time_place", gen_time_place_stmt),
    ("negation",   gen_negation_stmt),
    ("wh",         gen_wh_question),
    ("imperative", gen_imperative),
    ("ojala",      gen_ojala),
    ("jugar",      gen_jugar_futbol),
    ("llamarse",   gen_llamarse),
    ("ir_ser_amb", gen_ir_ser_ambiguity),   # <-- nuevo
    ("ve_amb",     gen_ve_ambiguity),       # <-- nuevo
]

def sanity_ok(src: str, tgt: str) -> bool:
    """
    Filtros suaves para garantizar que la traducción sigue las reglas clave:
    - Ojalá -> contiene 'OJALA'
    - Pregunta WH -> WH al final
    - Imperativo 'Ven...' -> sujeto TU y verbo VENIR
    - Dámelo -> aparece A MI/ESO o similar por clíticos (si está, mejor)
    - Jugar al fútbol -> verbo JUGAR→inf 'JUGAR' en target (reglas lo llevan a infinitivo)
    - Llamarse -> verbo LLAMAR y (si hay nombre) #NOMBRE en target (reglas + léxico)
    """
    s = src.strip()
    t = tgt.strip()
    low = s.lower()

    if "ojalá" in low and "OJALA" not in t:
        return False
    if "¿" in s or "?" in s:
        ok = any(t.endswith(x) for x in [" COMO", " DONDE", " QUIEN", " QUE", " CUANDO", " PORQUE", "¿", "?"])
        if not ok:
            return False
    if low.startswith("ven "):
        if not ("TU" in t and "VENIR" in t):
            return False
    if "dámelo" in low:
        if not (("A MI" in t) or ("ESO" in t)):
            return False
    if "fútbol" in low and "JUGAR" not in t:
        return False
    if "me llamo" in low or "yo me llamo" in low:
        if "LLAMAR" not in t:
            return False
    return True

def generate_split(n: int) -> List[Dict[str, str]]:
    out = []
    per = max(1, n // len(GENS))
    for name, fn in GENS:
        for _ in range(per):
            tries = 0
            while True:
                tries += 1
                src = fn()
                tgt = translate_rule_based(src)
                if sanity_ok(src, tgt) or tries > 5:
                    out.append({"src": src, "tgt": tgt, "tpl": name})
                    break
    while len(out) < n:
        name, fn = _choose(GENS)
        tries = 0
        while True:
            tries += 1
            src = fn()
            tgt = translate_rule_based(src)
            if sanity_ok(src, tgt) or tries > 5:
                out.append({"src": src, "tgt": tgt, "tpl": name})
                break
    return out

def write_jsonl(rows: List[Dict[str, str]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_all(n_train=3000, n_dev=400, n_test=400):
    train = generate_split(n_train)
    dev   = generate_split(n_dev)
    test  = generate_split(n_test)
    write_jsonl(train, "data/synthetic/train.jsonl")
    write_jsonl(dev,   "data/synthetic/dev.jsonl")
    write_jsonl(test,  "data/synthetic/test.jsonl")

if __name__ == "__main__":
    generate_all()
