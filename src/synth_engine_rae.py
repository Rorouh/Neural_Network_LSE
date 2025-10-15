# src/synth_engine_rae.py
import json, random, os, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional
from .rule_engine import translate_rule_based
from .rae_lexicon import build_pools
from .rae_toponyms import load_toponyms, build_places_from_toponyms


random.seed(1234)

# -------------------------------
# Datos y tipos
# -------------------------------
@dataclass
class Noun:
    surface: str
    gender: str  # 'm' | 'f' | 'u'
    number: str = "sg"

# Curado mínimo (CCT/CCL y sujetos) — mantenemos estos hasta que tengas toponimia
TIMES = ["Ayer","Hoy","Mañana","Ahora","Luego","A las 9",""]
PLACES_BASE = ["en la tienda","en casa","en Madrid","en el parque","a la escuela","en el trabajo",""]

SUBJECTS_1S: List[str] = ["Yo"]
SUBJECTS_3S: List[str] = ["Ana","Miguel","Lucía","Carlos"]

# Verbos auxiliares/irregulares que usamos en plantillas específicas
VERB_FORMS = {
    "ser":   {"3s_pret":"fue"},
    "estar": {"3s_pret":"estuvo"},
}

# Pools globales (se llenan desde RAE)
POOL = {
    "nouns": [],          # List[Tuple[lemma, gender]]
    "verbs_trans": [],
    "verbs_intrans": [],
    "verbs_mixed": [],
    "adjs": [],
    "advs": [],
}

def _guess_number(word: str) -> str:
    return "pl" if word.endswith("s") and not word.endswith("és") else "sg"

def _build_nouns(max_nouns: int) -> List[Noun]:
    out = []
    for lemma, gender in POOL["nouns"][:max_nouns]:
        out.append(Noun(lemma, gender or "u", _guess_number(lemma)))
    if not out:  # fallback mínimo si los pools vienen vacíos
        out = [Noun("libro","m"), Noun("café","m"), Noun("película","f"), Noun("casa","f")]
    return out

def _choose(lst):
    return random.choice(lst)

def det(typ: str, n: Noun) -> str:
    if typ == "zero": return ""
    table = {
        ("def","m","sg"):"el",  ("def","f","sg"):"la",
        ("def","m","pl"):"los", ("def","f","pl"):"las",
        ("ind","m","sg"):"un",  ("ind","f","sg"):"una",
        ("ind","m","pl"):"unos",("ind","f","pl"):"unas",
        ("def","u","sg"):"el",  ("def","u","pl"):"los",
        ("ind","u","sg"):"un",  ("ind","u","pl"):"unos",
    }
    return table.get((typ, n.gender, n.number), table.get((typ,"u",n.number),""))

def _clean_spaces(s: str) -> str:
    return " ".join(s.replace(" ,", ",").replace(" .", ".").split())

# -------------------------------
# Generadores “simples”
# -------------------------------
def gen_time_place_stmt(NOUNS: List[Noun], places: List[str]) -> str:
    n = _choose(NOUNS)
    obj = (det(_choose(["def","ind","zero"]), n) + " " + n.surface).strip()
    time = _choose(TIMES)
    place = _choose(places)
    # decide transitivo/intransitivo/mixed
    # Si hay objeto, usa verbo transitivo; si no, usa intransitivo.
    if obj and random.random() < 0.65 and POOL["verbs_trans"]:
        v = _choose(POOL["verbs_trans"])
        form = "1s" if random.random() < 0.6 else "3s"
        verb_form = {"1s":"", "3s":""}[form]  # Fuente es infinitivo en src; lo que importa es el lemma en la regla
        # construye la oración: S + V + OBJ (+ PLACE)
        S = "Yo" if form=="1s" else _choose(SUBJECTS_3S)
        sent = f"{time} {S} {v} {obj} {place}".strip() + "."
    else:
        # intransitivo o mixto sin objeto
        v_list = POOL["verbs_intrans"] or POOL["verbs_mixed"] or ["ir","vivir"]
        v = _choose(v_list)
        S = "Yo" if random.random() < 0.6 else _choose(SUBJECTS_3S)
        sent = f"{time} {S} {v} {place}".strip() + "."
    return _clean_spaces(sent)

def gen_negation_stmt(NOUNS: List[Noun]) -> str:
    S = "Yo" if random.random() < 0.7 else _choose(SUBJECTS_3S)
    n = _choose(NOUNS)
    obj = (det("ind", n) + " " + n.surface).strip()
    v = _choose(POOL["verbs_trans"] or ["querer"])
    return _clean_spaces(f"{S} no {v} {obj}.")

def gen_wh_question() -> str:
    return _choose(["¿Cómo te llamas?","¿Dónde vives?","¿Quién viene?","¿Qué quieres?"])

def gen_imperative() -> str:
    return _choose(["Ven aquí ahora mismo.","Dámelo, por favor.","Haz la tarea.","Pon eso ahí."])

def gen_ojala() -> str:
    return _clean_spaces(_choose([
        "Ojalá llueva mañana.","Ojalá venga Ana.","Ojalá haya café.","Ojalá pueda venir Miguel mañana."
    ]))

def gen_fue_a_place(places_go: List[str]) -> str:
    S = _choose(SUBJECTS_3S)
    dest = _choose([p for p in places_go if p.startswith("a ")]) if any(p.startswith("a ") for p in places_go) else "a Madrid"
    return f"{S} {VERB_FORMS['ser']['3s_pret']} {dest}."

def gen_fue_copular() -> str:
    S = _choose(SUBJECTS_3S)
    pred = _choose(["profesora","profesor","feliz","triste","amable"])
    return f"{S} {VERB_FORMS['ser']['3s_pret']} {pred}."

def gen_estar_en_place(places_en: List[str]) -> str:
    S = _choose(SUBJECTS_3S)
    place = _choose([p for p in places_en if p.startswith("en ")]) or "en casa"
    return f"{S} {VERB_FORMS['estar']['3s_pret']} {place}."

def gen_estar_adj() -> str:
    S = _choose(SUBJECTS_3S)
    adj = _choose(["cansado","cansada","contento","contenta","enfermo","enferma"])
    return f"{S} estaba {adj}."

# -------------------------------
# Sampling “templates” y producto
# -------------------------------
def sanity_ok(src: str, tgt: str) -> bool:
    s = src.strip().lower()
    t = tgt.strip()
    if "ojalá" in s and "OJALA" not in t:
        return False
    if "¿" in src or "?" in src:
        if not any(t.endswith(x) for x in [" COMO"," DONDE"," QUIEN"," QUE"," CUANDO"," PORQUE"]):
            return False
    if " ven " in f" {s} ":
        if not ("TU" in t and "VENIR" in t):
            return False
    if " fue a " in s:
        if " IR" not in t:
            return False
    if " fue " in s and any(x in s for x in ["profesor","profesora","feliz","triste","amable"]):
        if " SER" in t:
            return False
    return True

def _emit(n_src: str) -> Tuple[str, str, str]:
    tgt = translate_rule_based(n_src)
    tpl = "auto"
    return n_src, tgt, tpl

def product_time_place(NOUNS: List[Noun],
                       places: List[str],
                       time_max: int,
                       place_max: int,
                       subj_max: int,
                       verbs_t_max: int,
                       obj_max: int) -> List[Dict[str,str]]:
    """
    Crea oraciones por producto acotado:
      CCT × CCL × SUJETO × OBJETO × VERBO(transitivo)
    """
    times = [t for t in TIMES if t][:time_max]
    places_use = places[:place_max]
    subjects = (SUBJECTS_1S + SUBJECTS_3S)[:subj_max]
    objs = NOUNS[:obj_max]
    v_trans = POOL["verbs_trans"][:verbs_t_max] or ["comprar","ver"]

    out = []
    for t in times:
        for pl in places_use:
            for s in subjects:
                for n in objs:
                    obj = (det("ind", n) + " " + n.surface).strip()
                    for v in v_trans:
                        sent = f"{t} {s} {v} {obj} {pl}".strip() + "."
                        src, tgt, tpl = _emit(_clean_spaces(sent))
                        if sanity_ok(src, tgt):
                            out.append({"src": src, "tgt": tgt, "tpl": "product_time_place"})
    return out

# -------------------------------
# Escritura y CLI
# -------------------------------
def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def generate_all(n_train=3000, n_dev=400, n_test=400,
                 sampling="mixed",
                 out_dir="data/synthetic_rae",
                 rae_lexicon_path=None,
                 product_cfg=None,
                 rae_toponyms_root=None,
                 topo_max_en=4000,
                 topo_max_a=4000,
                 topo_include_world=False):

    global POOL

    # 1) Cargamos pools desde tu JSON/CSV de la RAE
    if not rae_lexicon_path or not os.path.exists(rae_lexicon_path):
        raise FileNotFoundError(f"--rae_lexicon_path no encontrado: {rae_lexicon_path}")
    POOL = build_pools(rae_lexicon_path)

    # 2) Construimos Nouns
    NOUNS = _build_nouns(max_nouns=5000)

    # 2b) TOPÓNIMOS -> lugares reales (en / a)
    if not rae_toponyms_root:
        raise FileNotFoundError("--rae_toponyms_root es obligatorio")
    topo_names = load_toponyms(rae_toponyms_root, include_world=topo_include_world)
    PLACES_EN, PLACES_A = build_places_from_toponyms(
        topo_names, max_en=topo_max_en, max_a=topo_max_a, shuffle=True
    )

    # Reserva también una opción vacía (sin CCL) cuando interese
    PLACES_EN_PLUS_EMPTY = PLACES_EN + [""]


    # 3) Definimos la lista de generadores
    gens: List[Callable[[], str]] = [
        lambda: gen_time_place_stmt(NOUNS, PLACES_EN_PLUS_EMPTY),
        lambda: gen_negation_stmt(NOUNS),
        gen_wh_question,
        gen_imperative,
        gen_ojala,
        lambda: gen_fue_a_place(PLACES_A),
        gen_fue_copular,
        lambda: gen_estar_en_place(PLACES_EN),
        gen_estar_adj,
    ]

    def sample_one() -> Dict[str,str]:
        src = random.choice(gens)()
        tgt = translate_rule_based(src)
        return {"src": src, "tgt": tgt, "tpl": "auto"}

    # 4) Sampling
    def draw(n):
        out=[]
        for _ in range(n):
            tries=0
            while True:
                tries+=1
                r = sample_one()
                if sanity_ok(r["src"], r["tgt"]) or tries>5:
                    out.append(r); break
        return out

    train, dev, test = [], [], []
    if sampling in {"mixed","uniform","frequent"}:
        # Back-compat: generadores aleatorios
        train = draw(n_train)
        dev   = draw(n_dev)
        test  = draw(n_test)
    elif sampling in {"product","product_time_place"}:
        # Producto acotado
        pcfg = product_cfg or {}
        pt = product_time_place(
            NOUNS=NOUNS,
            places=PLACES_EN,
            time_max=int(pcfg.get("product_max_time", 20)),
            place_max=int(pcfg.get("product_max_place", 40)),
            subj_max=int(pcfg.get("product_max_subjects", 10)),
            verbs_t_max=int(pcfg.get("product_max_verbs", 20)),
            obj_max=int(pcfg.get("product_max_objs", 50)),
        )
        # Particiona
        all_rows = pt
        train = all_rows[:n_train]
        dev   = all_rows[n_train:n_train+n_dev]
        test  = all_rows[n_train+n_dev:n_train+n_dev+n_test]
    else:
        raise ValueError(f"sampling no soportado: {sampling}")

    # 5) Guarda
    write_jsonl(train, os.path.join(out_dir,"train.jsonl"))
    write_jsonl(dev,   os.path.join(out_dir,"dev.jsonl"))
    write_jsonl(test,  os.path.join(out_dir,"test.jsonl"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_dev",   type=int, default=400)
    ap.add_argument("--n_test",  type=int, default=400)

    ap.add_argument("--sampling", type=str, default="mixed",
                    choices=["mixed","uniform","frequent","product","product_time_place"])
    ap.add_argument("--out_dir",  type=str, default="data/synthetic_rae")

    ap.add_argument("--rae_lexicon_path", type=str, required=True,
                    help="Ruta al JSON o CSV que has generado (rae_es_ES_lemma_to_pos_simple.json / .csv)")

    #Parametros toponimos
    ap.add_argument("--rae_toponyms_root", type=str, required=True,
                help="Carpeta .../rla-es/ortografia/toponimos")
    ap.add_argument("--topo_max_en", type=int, default=4000)
    ap.add_argument("--topo_max_a",  type=int, default=4000)
    ap.add_argument("--topo_include_world", action="store_true",
                    help="Añade toponimos-mundo.txt además de es_ES")

    # Parámetros del producto acotado
    ap.add_argument("--product_max_time", type=int, default=20)
    ap.add_argument("--product_max_place", type=int, default=40)
    ap.add_argument("--product_max_subjects", type=int, default=10)
    ap.add_argument("--product_max_verbs", type=int, default=20)
    ap.add_argument("--product_max_objs", type=int, default=50)

    args = ap.parse_args()

    product_cfg = dict(
        product_max_time=args.product_max_time,
        product_max_place=args.product_max_place,
        product_max_subjects=args.product_max_subjects,
        product_max_verbs=args.product_max_verbs,
        product_max_objs=args.product_max_objs,
    )

    generate_all(
        n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test,
        sampling=args.sampling,
        out_dir=args.out_dir,
        rae_lexicon_path=args.rae_lexicon_path,
        product_cfg=product_cfg,
        rae_toponyms_root=args.rae_toponyms_root,
        topo_max_en=args.topo_max_en,
        topo_max_a=args.topo_max_a,
        topo_include_world=args.topo_include_world,
    )


if __name__ == "__main__":
    main()
