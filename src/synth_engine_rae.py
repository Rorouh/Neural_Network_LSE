# src/synth_engine_rae.py
import json, random, os, re, glob, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional
from .rule_engine import translate_rule_based

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

# Verbos base
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

# Transitivos para el producto OBJ (puedes ampliar)
VERBS_TRANS = ["comprar", "ver", "querer", "jugar"]

# -------------------------------
# RAE: localización + limpieza de flags /S., /MF, etc.
# -------------------------------
RAE_BASE_CANDIDATES = [
    os.path.join("RAE","rla-es","ortografia","palabras","RAE","l10n","es_ES"),
    os.path.join("extern","rla-es","ortografia","palabras","RAE","l10n","es_ES"),
]
RAE_FLAG_RE = re.compile(r"/[^/\s]+$")  # elimina sufijos tipo "/S.", "/MF", ...

def _strip_rae_flags(w: str) -> str:
    return RAE_FLAG_RE.sub("", w).strip()

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
                w = _norm_line(line)
                if w: out[bucket].append(w)
    # dedup + limpieza de flags
    for k,lst in out.items():
        seen, res = set(), []
        for x in lst:
            x0 = _strip_rae_flags(x)
            xl = x0.lower()
            if xl and xl not in seen:
                seen.add(xl); res.append(x0)
        out[k] = res
    return out

def take_sample(lst: List[str], k: int) -> List[str]:
    return random.sample(lst, min(k, len(lst))) if lst else []

def det(typ: str, n: Noun) -> str:
    if typ == "zero": return ""
    table = {
        ("def","m","sg"):"el",  ("def","f","sg"):"la",
        ("def","m","pl"):"los", ("def","f","pl"):"las",
        ("ind","m","sg"):"un",  ("ind","f","sg"):"una",
        ("ind","m","pl"):"unos",("ind","f","pl"):"unas",
    }
    return table[(typ, n.gender, n.number)]

def seed_from_rla_es(base_dir: Optional[str], max_nouns=2000, max_places=1500) -> Tuple[List[Noun], List[str], List[str]]:
    wl = load_rla_es_wordlists(base_dir)
    if not any(wl.values()):
        return FALLBACK_NOUNS, FALLBACK_PLACES, FALLBACK_TIMES

    # Sustantivos (heurística género/número muy simple)
    nouns: List[Noun] = []
    for w in take_sample(wl["nouns"], max_nouns):
        g = "f" if w.endswith("a") else "m"
        num = "pl" if (w.endswith("s") and not w.endswith("és")) else "sg"
        nouns.append(Noun(w, g, num))

    # Lugares (preferimos "en X" para este producto con objeto)
    places = []
    pool_places = wl["toponyms"] or wl["names"]
    for city in take_sample(pool_places, max_places):
        places.append(f"en {city}")
    places += ["en el trabajo","en el parque","en casa","en la tienda",""]  # + vacío opcional

    times = FALLBACK_TIMES
    return nouns or FALLBACK_NOUNS, places or FALLBACK_PLACES, times

# estos se rellenan en main()
NOUNS: List[Noun] = []
PLACES: List[str] = []
TIMES: List[str] = []

# -------------------------------
# Utilidades
# -------------------------------
def _clean_spaces(s: str) -> str:
    s = s.replace(" ,", ",").replace(" .", ".")
    return " ".join(s.split())

def _maybe(p: float) -> bool:
    return random.random() < p

def _choose(lst):
    return random.choice(lst)

# -------------------------------
# Generadores clásicos (se mantienen)
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
    dests = [p for p in PLACES if p.startswith("a ")]
    dest = _choose(dests) if dests else "a Madrid"
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

# -------------------------------
# NUEVO: Producto CCT × CCL × SUJ × OBJ × VERBO(transitivo)
# -------------------------------
def _conj(v: str, sujeto: str, time_token: str) -> str:
    """Conjugación simple para la FUENTE (castellano) según 'Ayer'."""
    if time_token.strip().lower() == "ayer":
        return VERBS[v]["1s_pret"] if sujeto == "Yo" and "1s_pret" in VERBS[v] else VERBS[v].get("3s_pret", VERBS[v].get("inf", v))
    else:
        return VERBS[v]["1s_pres"] if sujeto == "Yo" and "1s_pres" in VERBS[v] else VERBS[v].get("3s_pres", VERBS[v].get("inf", v))

def _make_obj_phrase(n: Noun) -> str:
    # Objeto “natural” en castellano; tu regla removerá determinantes en la salida LSE
    d = det("ind", n)
    return (d + " " + n.surface).strip()

def generate_product_split(n: int, max_time=20, max_place=40, max_subjects=8, max_verbs=15, max_objs=30) -> List[Dict[str,str]]:
    # selecciona listas acotadas
    times = [t for t in TIMES if t]  # sin vacío
    times = take_sample(times, max_time) or ["Hoy"]

    places_en = [p for p in PLACES if p.startswith("en ") and p]  # “en …” para transitivos con objeto
    places_en = take_sample(places_en, max_place) or ["en Madrid"]

    subjects = ["Yo"] + SUBJECTS_3S
    subjects = subjects[:max_subjects] if max_subjects < len(subjects) else subjects

    verbs = [v for v in VERBS_TRANS if v in VERBS]
    verbs = verbs[:max_verbs] if max_verbs < len(verbs) else verbs
    if not verbs: verbs = ["comprar"]

    # objetos a partir de NOUNS
    objs_pool = NOUNS[:]
    random.shuffle(objs_pool)
    if max_objs < len(objs_pool):
        objs_pool = objs_pool[:max_objs]
    if not objs_pool:
        objs_pool = FALLBACK_NOUNS

    # Producto cartesiano (acotado a n)
    tuples = []
    for t in times:
        for p in places_en:
            for s in subjects:
                for n in objs_pool:
                    for v in verbs:
                        tuples.append((t, p, s, n, v))
    random.shuffle(tuples)
    tuples = tuples[:n]

    out = []
    for (t, p, s, n, v) in tuples:
        v_form = _conj(v, s, t)
        obj_phrase = _make_obj_phrase(n)
        src = f"{t} {s} {v_form} {obj_phrase} {p}."
        src = _clean_spaces(src)
        tgt = translate_rule_based(src)
        if sanity_ok(src, tgt):
            out.append({"src": src, "tgt": tgt, "tpl": "product_tpl"})
    return out

# -------------------------------
# Escritura y main
# -------------------------------
def write_jsonl(rows: List[Dict[str, str]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_split(n: int, sampling: str = "mixed") -> List[Dict[str, str]]:
    high = ["time_place","negation","fue_ir","wh"]
    mid  = ["imperative","estar_loc","estar_attr"]
    low  = ["ojala","fue_ser"]
    name2fn = dict(GENS)

    if sampling == "uniform":
        pool = [name for name,_ in GENS]
        chosen = [random.choice(pool) for _ in range(n)]
    elif sampling == "frequent":
        weights = {**{k:3 for k in high}, **{k:2 for k in mid}, **{k:1 for k in low}}
        pool = [name for name,_ in GENS for _ in range(weights.get(name,1))]
        chosen = [random.choice(pool) for _ in range(n)]
    else:
        chosen = []
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

def generate_all(n_train=3000, n_dev=400, n_test=400, sampling="mixed",
                 out_dir="data/synthetic_rae", base_dir=None,
                 product_max_time=20, product_max_place=40, product_max_subjects=8, product_max_verbs=15, product_max_objs=30):
    global NOUNS, PLACES, TIMES
    base = _pick_existing_dir(base_dir)
    NOUNS, PLACES, TIMES = seed_from_rla_es(base)

    if sampling == "product_time_place":
        train = generate_product_split(n_train, product_max_time, product_max_place, product_max_subjects, product_max_verbs, product_max_objs)
        dev   = generate_product_split(n_dev,   min(10, product_max_time), min(20, product_max_place), min(6, product_max_subjects), min(10, product_max_verbs), min(20, product_max_objs))
        test  = generate_product_split(n_test,  min(10, product_max_time), min(20, product_max_place), min(6, product_max_subjects), min(10, product_max_verbs), min(20, product_max_objs))
    else:
        train = generate_split(n_train, sampling=sampling)
        dev   = generate_split(n_dev,   sampling=sampling)
        test  = generate_split(n_test,  sampling=sampling)

    write_jsonl(train, os.path.join(out_dir, "train.jsonl"))
    write_jsonl(dev,   os.path.join(out_dir, "dev.jsonl"))
    write_jsonl(test,  os.path.join(out_dir, "test.jsonl"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_dev",   type=int, default=400)
    ap.add_argument("--n_test",  type=int, default=400)

    ap.add_argument("--sampling", type=str, default="mixed",
                    choices=["mixed","uniform","frequent","product_time_place"])
    ap.add_argument("--out_dir",  type=str, default="data/synthetic_rae")
    ap.add_argument("--base_dir", type=str, default=None,
                    help="Carpeta RAE es_ES; si no se pasa, intenta RAE/rla-es/... o extern/rla-es/...")

    # NUEVOS límites del producto
    ap.add_argument("--product_max_time", type=int, default=20)
    ap.add_argument("--product_max_place", type=int, default=40)
    ap.add_argument("--product_max_subjects", type=int, default=8)
    ap.add_argument("--product_max_verbs", type=int, default=15)
    ap.add_argument("--product_max_objs", type=int, default=30)

    args = ap.parse_args()
    generate_all(args.n_train, args.n_dev, args.n_test,
                 sampling=args.sampling, out_dir=args.out_dir, base_dir=args.base_dir,
                 product_max_time=args.product_max_time,
                 product_max_place=args.product_max_place,
                 product_max_subjects=args.product_max_subjects,
                 product_max_verbs=args.product_max_verbs,
                 product_max_objs=args.product_max_objs)

if __name__ == "__main__":
    main()
