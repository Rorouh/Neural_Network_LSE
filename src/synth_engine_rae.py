# src/synth_engine_rae.py
import json, random, os, re, glob, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional
from .rule_engine import translate_rule_based
import random

random.seed(1234)

# -------------------------------
# Fallback mínimo (por si no hay RAE)
# -------------------------------
@dataclass
class Noun:
    surface: str
    gender: str  # 'm' o 'f'
    number: str = "sg"  # 'sg' o 'pl'

FALLBACK_NOUNS: List[Noun] = [
    Noun("libro","m"), Noun("café","m"),
    Noun("película","f"), Noun("casa","f"),
    Noun("zapatos","m","pl"), Noun("manzanas","f","pl"),
]
FALLBACK_PLACES: List[str] = [
    "en la tienda","en casa","en Madrid","en el parque","a la escuela","en el trabajo",""
]
FALLBACK_TIMES: List[str] = ["Ayer","Hoy","Mañana","Ahora","Luego","A las 9",""]
SUBJECTS_1S: List[str] = ["Yo"]
SUBJECTS_3S: List[str] = ["Ana","Miguel","Lucía","Carlos"]

RAE_FLAG_RE = re.compile(r"/[^/\s]+$")  # todo lo que va tras la primera "/" hasta fin de token

def _strip_rae_flags(w: str) -> str:
    return RAE_FLAG_RE.sub("", w).strip()

_TOKEN_OK = re.compile(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:[ \-][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)*$")


def det(typ: str, n: Noun) -> str:
    if typ == "zero": return ""
    table = {
        ("def","m","sg"):"el",  ("def","f","sg"):"la",
        ("def","m","pl"):"los", ("def","f","pl"):"las",
        ("ind","m","sg"):"un",  ("ind","f","sg"):"una",
        ("ind","m","pl"):"unos",("ind","f","pl"):"unas",
    }
    return table[(typ, n.gender, n.number)]

VERBS: Dict[str, Dict[str,str]] = {
    "comprar":{"1s_pret":"compré","3s_pret":"compró","1s_pres":"compro","3s_pres":"compra","inf":"comprar"},
    "ver":{"1s_pret":"vi","3s_pret":"vio","1s_pres":"veo","3s_pres":"ve","inf":"ver"},
    "querer":{"1s_pres":"quiero","3s_pres":"quiere","1s_pret":"quise","3s_pret":"quiso","inf":"querer"},
    "ir":{"1s_pres":"voy","3s_pres":"va","1s_pret":"fui","3s_pret":"fue","imp_2s":"ve","inf":"ir"},
    "ser":{"1s_pret":"fui","3s_pret":"fue","1s_pres":"soy","3s_pres":"es","inf":"ser"},
    "estar":{"1s_pres":"estoy","3s_pres":"está","1s_pret":"estuve","3s_pret":"estuvo","inf":"estar"},
    "vivir":{"1s_pres":"vivo","3s_pres":"vive","1s_pret":"viví","3s_pret":"vivió","inf":"vivir"},
    "dar":{"imp_2s":"da","inf":"dar"},
    "venir":{"imp_2s":"ven","inf":"venir"},
    "hacer":{"imp_2s":"haz","inf":"hacer"},
    "poner":{"imp_2s":"pon","inf":"poner"},
    "decir":{"imp_2s":"di","inf":"decir"},
    "llover":{"subj_3s":"llueva","inf":"llover"},
    "llamar":{"1s_pres_ref":"me llamo","2s_pres_ref":"te llamas","3s_pres_ref":"se llama","inf":"llamar"},
    "jugar":{"1s_pres":"juego","3s_pres":"juega","inf":"jugar"},
}

# -------------------------------
# RAE: localización por defecto + fallback
# -------------------------------
RAE_BASE_CANDIDATES = [
    # tu ruta actual:
    os.path.join("RAE","rla-es","ortografia","palabras","RAE","l10n","es_ES"),
    # alternativa clásica por si la mueves:
    os.path.join("extern","rla-es","ortografia","palabras","RAE","l10n","es_ES"),
]

def _norm_line(s: str) -> Optional[str]:
    s = s.strip()
    if not s: return None
    if s.startswith("#") or s.startswith("//"): return None
    return s

def _pick_existing_dir(user_dir: Optional[str]) -> Optional[str]:
    if user_dir and os.path.isdir(user_dir): return user_dir
    for d in RAE_BASE_CANDIDATES:
        if os.path.isdir(d): return d
    return None


def load_rla_es_wordlists(base_dir: Optional[str]) -> Dict[str, List[str]]:
    out = {"nouns":[], "adjs":[], "verbs":[], "advs":[], "names":[], "toponyms":[]}
    if not base_dir: return out
    files = glob.glob(os.path.join(base_dir, "*.txt"))
    for fp in files:
        name = os.path.basename(fp).lower()
        bucket = None
        if any(k in name for k in ["sustant","nombres_comunes","nombres-comunes","sustantivo"]):
            bucket = "nouns"
        elif "adjet" in name:
            bucket = "adjs"
        elif "verbo" in name or "verbos" in name:
            bucket = "verbs"
        elif "adverb" in name:
            bucket = "advs"
        elif any(k in name for k in ["nombres_propios","nombres-propios","apellidos","nombres"]):
            bucket = "names"
        elif any(k in name for k in ["topon","gentil","lugares","ciudades","provincias"]):
            bucket = "toponyms"
        if bucket is None: 
            continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("//"):
                    continue
                s = _strip_rae_flags(s)
                if not _TOKEN_OK.match(s):
                    continue
                out[bucket].append(s)
    # dedup simple (case-insensible)
    for k,lst in out.items():
        seen, res = set(), []
        for x in lst:
            xl = x.lower()
            if xl not in seen:
                seen.add(xl); res.append(x)
        out[k] = res
    return out

def take_sample(lst: List[str], k: int) -> List[str]:
    return random.sample(lst, min(k, len(lst))) if lst else []

def seed_from_rla_es(base_dir: Optional[str], max_nouns=2000, max_places=1500) -> Tuple[List[Noun], List[str], List[str]]:
    wl = load_rla_es_wordlists(base_dir)
    if not any(wl.values()):
        return FALLBACK_NOUNS, FALLBACK_PLACES, FALLBACK_TIMES

    nouns: List[Noun] = []
    for w in take_sample(wl["nouns"], max_nouns):
        base = _strip_rae_flags(w)                 # <<--- LIMPIA FLAGS
        if not base: 
            continue
        g = "f" if base.endswith("a") else "m"     # heurística simple
        num = "pl" if base.endswith("s") and not base.endswith("és") else "sg"
        nouns.append(Noun(base, g, num))

    places = []
    seed_places = wl["toponyms"] or wl["names"]
    for city in take_sample(seed_places, max_places):
        base = _strip_rae_flags(city)              # <<--- LIMPIA FLAGS
        if base:
            places.append(f"en {base}")
    places += ["a Madrid","a la escuela","a casa","a la tienda","en el trabajo","en el parque",""]

    times = FALLBACK_TIMES
    return nouns or FALLBACK_NOUNS, places or FALLBACK_PLACES, times

# estos se rellenan en main() tras parsear args
NOUNS: List[Noun] = []
PLACES: List[str] = []
TIMES: List[str] = []

# -------------------------------
# Utils
# -------------------------------
def _clean_spaces(s: str) -> str:
    s = s.replace(" ,", ",").replace(" .", ".")
    return " ".join(s.split())

def _maybe(p: float) -> bool:
    return random.random() < p

def _choose(lst):
    return random.choice(lst)

# -------------------------------
# Generadores (mismos + nuevos “fue”)
# -------------------------------
def gen_time_place_stmt() -> str:
    n = _choose(NOUNS)
    dtyp = _choose(["def","ind","zero"])
    obj = (det(dtyp, n) + " " + n.surface).strip()
    time = _choose(TIMES)
    place = _choose(PLACES)

    if _maybe(0.6):
        S = _choose(SUBJECTS_1S)
        v = _choose(["comprar","ver","ir","vivir"])
        form = VERBS[v]["1s_pret"] if time.lower()=="ayer" and "1s_pret" in VERBS[v] else VERBS[v].get("1s_pres", VERBS[v].get("1s_pret"))
    else:
        S = _choose(SUBJECTS_3S)
        v = _choose(["comprar","ver","ir","vivir"])
        form = VERBS[v]["3s_pret"] if time.lower()=="ayer" and "3s_pret" in VERBS[v] else VERBS[v].get("3s_pres", VERBS[v].get("3s_pret"))

    pieces = []
    if time: pieces.append(time)
    pieces.append(S); pieces.append(form)
    if obj and v != "ir": pieces.append(obj)
    if place: pieces.append(place)
    return _clean_spaces(" ".join(pieces) + ".")

def gen_negation_stmt() -> str:
    S = "Yo" if _maybe(0.7) else _choose(SUBJECTS_3S)
    n = _choose(NOUNS)
    obj = (det("ind", n) + " " + n.surface).strip()
    vform = VERBS["querer"]["1s_pres"] if S=="Yo" else VERBS["querer"]["3s_pres"]
    return _clean_spaces(f"{S} no {vform} {obj}.")

def gen_wh_question() -> str:
    return _choose(["¿Cómo te llamas?","¿Dónde vives?","¿Quién viene?","¿Qué quieres?"])

def gen_imperative() -> str:
    return _choose(["Ven aquí ahora mismo.","Dámelo, por favor.","Haz la tarea.","Pon eso ahí."])

def gen_ojala() -> str:
    return _clean_spaces(_choose([
        "Ojalá llueva mañana.","Ojalá venga Ana.","Ojalá haya café.","Ojalá pueda venir Miguel mañana."
    ]))

def gen_fue_a_place() -> str:
    S = _choose(SUBJECTS_3S)
    dest = _choose([p for p in PLACES if p.startswith("a ")])
    if not dest: dest = "a Madrid"
    return f"{S} fue {dest}."

def gen_fue_copular() -> str:
    S = _choose(SUBJECTS_3S)
    pred = _choose(["profesora","profesor","feliz","triste","amable"])
    return f"{S} fue {pred}."

def gen_estar_en_place() -> str:
    S = _choose(SUBJECTS_3S)
    place = _choose([p for p in PLACES if p.startswith("en ") and p])
    return f"{S} estuvo {place}."

def gen_estar_adj() -> str:
    S = _choose(SUBJECTS_3S)
    adj = _choose(["cansado","cansada","contento","contenta","enfermo","enferma"])
    return f"{S} estaba {adj}."

GENS: List[Tuple[str, Callable[[], str]]] = [
    ("time_place", gen_time_place_stmt),
    ("negation",   gen_negation_stmt),
    ("wh",         gen_wh_question),
    ("imperative", gen_imperative),
    ("ojala",      gen_ojala),
    ("fue_ir",     gen_fue_a_place),
    ("fue_ser",    gen_fue_copular),
    ("estar_loc",  gen_estar_en_place),
    ("estar_attr", gen_estar_adj),
]

def sanity_ok(src: str, tgt: str) -> bool:
    s = src.strip(); t = tgt.strip(); low = s.lower()
    if "ojalá" in low and "OJALA" not in t: return False
    if "¿" in s or "?" in s:
        if not any(t.endswith(x) for x in [" COMO"," DONDE"," QUIEN"," QUE"," CUANDO"," PORQUE"]): return False
    if re.search(r"\bven\b", low):
        if not ("TU" in t and "VENIR" in t): return False
    if re.search(r"\bfue\s+a\b", low):
        if " IR" not in t: return False
    if re.search(r"\bfue\s+(profesor|profesora|feliz|triste|amable)\b", low):
        if " SER" in t: return False
    return True

def generate_split(n: int, sampling: str = "mixed") -> List[Dict[str, str]]:
    high = ["time_place","negation","fue_ir","wh"]
    mid  = ["imperative","estar_loc","estar_attr"]
    low  = ["ojala","fue_ser"]
    name2fn = dict(GENS)
    chosen: List[str] = []
    if sampling == "uniform":
        pool = [name for name,_ in GENS]
        chosen = [random.choice(pool) for _ in range(n)]
    elif sampling == "frequent":
        weights = {**{k:3 for k in high}, **{k:2 for k in mid}, **{k:1 for k in low}}
        pool = [name for name,_ in GENS for _ in range(weights.get(name,1))]
        chosen = [random.choice(pool) for _ in range(n)]
    else:
        for _ in range(n):
            r = random.random()
            if r < 0.6: chosen.append(random.choice(high))
            elif r < 0.9: chosen.append(random.choice(mid))
            else: chosen.append(random.choice(low))

    out = []
    for name in chosen:
        tries = 0
        while True:
            tries += 1
            src = name2fn[name]()
            tgt = translate_rule_based(src)
            if sanity_ok(src, tgt) or tries > 5:
                out.append({"src": src, "tgt": tgt, "tpl": name})
                break
    return out

def write_jsonl(rows: List[Dict[str, str]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_all(n_train=3000, n_dev=400, n_test=400, sampling="mixed",
                 out_dir="data/synthetic_rae", base_dir=None):
    global NOUNS, PLACES, TIMES
    base = _pick_existing_dir(base_dir)
    NOUNS, PLACES, TIMES = seed_from_rla_es(base)
    train = generate_split(n_train, sampling=sampling)
    dev   = generate_split(n_dev,   sampling=sampling)
    test  = generate_split(n_test,  sampling=sampling)
    write_jsonl(train, os.path.join(out_dir, "train.jsonl"))
    write_jsonl(dev,   os.path.join(out_dir, "dev.jsonl"))
    write_jsonl(test,  os.path.join(out_dir, "test.jsonl"))

# PRODUCTO ACOTADO PARA GENERAR MAS CONTENIDO DEL DATASET SINTETICO PARA ENTRENAR EL MODELO.
def product_time_place(n_nouns: int, n_places: int) -> List[Dict[str, str]]:
    """
    Producto cartesiano ACOTADO y simple:
      - Sustantivos: primeros n_nouns de NOUNS
      - Lugares: primeros n_places de PLACES (sin el vacío)
      - Tiempos: "", "Ayer", "Hoy", "Mañana"
      - Sujetos: Yo + 3ª persona
      - Verbos: comprar, ver, vivir
    Genera oraciones SVO con tiempo/lugar, las traduce y filtra con sanity_ok.
    """
    out = []
    verbs = ["comprar", "ver", "vivir"]
    times = ["", "Ayer", "Hoy", "Mañana"]
    subjects = ["Yo"] + SUBJECTS_3S
    nouns = NOUNS[:n_nouns]
    places = [p for p in PLACES[:n_places] if p]  # evita vacío

    for n in nouns:
        obj = (det("ind", n) + " " + n.surface).strip()
        for pl in places:
            for t in times:
                for S in subjects:
                    for v in verbs:
                        if S == "Yo":
                            form = VERBS[v].get("1s_pret") if t.lower() == "ayer" else VERBS[v].get("1s_pres", VERBS[v].get("1s_pret"))
                        else:
                            form = VERBS[v].get("3s_pret") if t.lower() == "ayer" else VERBS[v].get("3s_pres", VERBS[v].get("3s_pret"))
                        pieces = []
                        if t: pieces.append(t)
                        pieces += [S, form, obj, pl]
                        src = " ".join([x for x in pieces if x]) + "."
                        tgt = translate_rule_based(src)
                        if sanity_ok(src, tgt):
                            out.append({"src": src, "tgt": tgt, "tpl": "prod_time_place"})
    return out


def _mix_and_trim(sampled_rows: List[Dict[str, str]],
                  product_rows: List[Dict[str, str]],
                  n_target: int,
                  product_ratio: float,
                  seed: int = 1234) -> List[Dict[str, str]]:
    """
    Mezcla productivo + muestreo para obtener exactamente n_target ejemplos.
    product_ratio = proporción (0..1) del split que viene del producto.
    """
    random.seed(seed)
    k_prod = min(int(product_ratio * n_target), len(product_rows))
    k_samp = max(0, n_target - k_prod)
    random.shuffle(product_rows)
    random.shuffle(sampled_rows)
    out = product_rows[:k_prod] + sampled_rows[:k_samp]
    random.shuffle(out)
    return out[:n_target]


def generate_all(n_train=3000, n_dev=400, n_test=400,
                 sampling="mixed", out_dir="data/synthetic_rae", base_dir=None,
                 product_top_nouns: int = 0,
                 product_top_places: int = 0,
                 product_ratio: float = 0.33,
                 seed: int = 1234):
    """
    Si product_top_nouns > 0 y product_top_places > 0:
      - Genera un pool del producto acotado
      - Mezcla 'product_ratio' de ese pool en el split de train
    Dev/Test se generan solo por muestreo (más realistas).
    """
    global NOUNS, PLACES, TIMES
    random.seed(seed)

    base = _pick_existing_dir(base_dir)
    NOUNS, PLACES, TIMES = seed_from_rla_es(base)

    # pool de muestreo aleatorio
    train_s = generate_split(n_train, sampling=sampling)
    dev_s   = generate_split(n_dev,   sampling=sampling)
    test_s  = generate_split(n_test,  sampling=sampling)

    # pool productivo (opcional)
    prod_pool = []
    if product_top_nouns > 0 and product_top_places > 0:
        prod_pool = product_time_place(product_top_nouns, product_top_places)

    # mezcla en train (dev/test sin producto para medir generalización)
    if prod_pool:
        train = _mix_and_trim(train_s, prod_pool, n_train, product_ratio, seed)
    else:
        train = train_s

    dev  = dev_s
    test = test_s

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(train, os.path.join(out_dir, "train.jsonl"))
    write_jsonl(dev,   os.path.join(out_dir, "dev.jsonl"))
    write_jsonl(test,  os.path.join(out_dir, "test.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_dev",   type=int, default=400)
    ap.add_argument("--n_test",  type=int, default=400)

    ap.add_argument("--sampling", type=str, default="mixed", choices=["mixed","uniform","frequent"])
    ap.add_argument("--out_dir",  type=str, default="data/synthetic_rae")
    ap.add_argument("--base_dir", type=str, default=None,
                    help="Carpeta RAE es_ES; si no se pasa, intenta RAE/rla-es/... o extern/rla-es/...")

    # --- NUEVO: producto acotado + control de mezcla ---
    ap.add_argument("--product_top_nouns",  type=int, default=0, help="Top-N sustantivos para producto (0 = desactivado)")
    ap.add_argument("--product_top_places", type=int, default=0, help="Top-M lugares para producto (0 = desactivado)")
    ap.add_argument("--product_ratio",      type=float, default=0.33, help="Proporción del train que viene del producto [0..1]")
    ap.add_argument("--seed",               type=int, default=1234)

    args = ap.parse_args()

    generate_all(
        n_train=args.n_train,
        n_dev=args.n_dev,
        n_test=args.n_test,
        sampling=args.sampling,
        out_dir=args.out_dir,
        base_dir=args.base_dir,
        product_top_nouns=args.product_top_nouns,
        product_top_places=args.product_top_places,
        product_ratio=args.product_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()

