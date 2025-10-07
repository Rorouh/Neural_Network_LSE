# src/rule_engine.py
from typing import List, Tuple, Dict, Set
import unicodedata

from .util import load_yaml, load_lexicon_csv, load_verb_overrides
from .features import parse, find_subject, find_object, find_main_verb
from .postprocess import postprocess_gloss

def _noacc_lower(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii").lower()


# --- Configs con defaults seguros ---
def _load_yaml_safe(path: str, default: dict) -> dict:
    try:
        data = load_yaml(path)
        if data is None:
            return default.copy()
        merged = default.copy()
        merged.update(data)
        return merged
    except Exception:
        return default.copy()

CFG_REORDER = _load_yaml_safe(
    "rules/reorder.yml",
    {"order": ["TIME", "PLACE", "SUBJECT", "VERB", "PRED"], "questions": {"wh_final": True}},
)
CFG_DROP = _load_yaml_safe(
    "rules/drop.yml",
    {"articles": True, "copula": True, "auxiliaries": True, "other_determiners": False, "keep_in_adverbials": True},
)
CFG_LEXICAL = _load_yaml_safe(
    "rules/lexical.yml",
    {
        "time_lex": ["ayer", "hoy", "mañana", "ahora", "luego", "despues", "anoche", "siempre", "a veces"],
        "place_preps": ["en", "a", "hacia", "desde", "cerca de", "delante de", "detrás de"],
        "neg_lex": ["no", "ni", "tampoco", "nunca"],
        "wh_lex": {
            "QUE": ["qué", "que"],
            "QUIEN": ["quién", "quien"],
            "COMO": ["cómo", "como"],
            "DONDE": ["dónde", "donde"],
            "CUANDO": ["cuándo", "cuando"],
            "PORQUE": ["por qué", "porque"],
        },
    },
)
CFG_CLITICS = _load_yaml_safe(
    "rules/clitics.yml",
    {
        "split_enclitics": True,
        "indirect_object": {
            "me": "A MI",
            "te": "A TI",
            "le": "A EL",
            "nos": "A NOSOTROS",
            "os": "A VOSOTROS",
            "les": "A ELLOS",
            "se": "A SI",
        },
        "direct_object": {"lo": "ESO", "la": "ESO", "los": "ESOS", "las": "ESAS"},
    },
)

# --- WH map ---
_WH_MAP = {}
for k, forms in CFG_LEXICAL.get("wh_lex", {}).items():
    for f in forms:
        _WH_MAP[f.lower()] = k

# Léxico principal
LEX: Dict[str, Dict] = load_lexicon_csv("data/lexicon/es_to_lse.csv")

# VERB_OV: mapa forma->infinitivo con tu CSV ancho
VERB_OV = load_verb_overrides("data/lexicon/verb_lemma_override.csv")

ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
NEG_SET: Set[str] = {w.lower() for w in CFG_LEXICAL.get("neg_lex", [])}
CLITIC_FORMS: Set[str] = {"me", "te", "se", "le", "les", "nos", "os", "lo", "la", "los", "las"}

# Helpers básicos
def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _is_copula(tok):  return tok.lemma_.lower() in {"ser", "estar"} and tok.pos_ == "VERB"
def _is_aux(tok):     return tok.pos_ == "AUX" or tok.lemma_.lower() == "haber"

def _in_time_phrase(tok) -> bool:
    try:
        if tok.head.dep_ in {"obl", "nmod"}:
            return any(t.like_num for t in tok.head.subtree)
    except Exception:
        pass
    return False

def _should_keep(tok) -> bool:
    t = tok.text.lower()
    if t in NEG_SET:
        return False
    if t == "ojalá":
        return False
    if tok.pos_ == "DET":
        if CFG_DROP.get("keep_in_adverbials", True) and _in_time_phrase(tok):
            return False
        return not (CFG_DROP.get("articles", True) or CFG_DROP.get("other_determiners", False))
    if tok.pos_ in {"ADP", "SCONJ"}:
        return False
    if CFG_DROP.get("copula", True) and _is_copula(tok):
        return False
    if CFG_DROP.get("auxiliaries", True) and _is_aux(tok):
        return False
    return True

def _lex_by_form(text_lower: str, pos: str):
    return LEX["by_form"].get((text_lower, pos))

def _lex_by_lemma(lemma_lower: str, pos: str):
    return LEX["by_lemma"].get((lemma_lower, pos))

# Imperativos irregulares frecuentes (fallback adicional)
IMP_FALLBACK = {
    "ven": "venir",
    "pon": "poner",
    "ten": "tener",
    "haz": "hacer",
    "di": "decir",
    "sal": "salir",
    "ve": "ir",     # 've' ~ ir (resolvemos a IR por defecto)
    "sé": "ser",
    "da": "dar",
}

def _best_verb_lemma(token, base_after_split: str = None) -> str:
    """
    Elige el mejor lema para VERB teniendo en cuenta:
      - overrides por forma
      - imperativos irregulares frecuentes
      - desambiguación 've' (ver vs ir) por morfología
      - desambiguación 'fui/fue/...' (ser vs ir) por sintaxis: 'a + OBL' => ir
    """
    def _strip_acc(s: str) -> str:
        return _strip_accents(s.lower())

    form = token.text.lower()
    form_noacc = _strip_acc(form)

    # --- caso especial: 've' (3sg de 'ver' vs imperativo de 'ir')
    if form_noacc == "ve":
        # si spacy detecta imperativo → IR
        if "Imp" in token.morph.get("Mood"):
            return "ir"
        # heurística adicional: si es inicio de frase/orden sencilla 'Ve a ...'
        # y existe un OBL con 'a', lo tratamos como IR
        has_obl_a = False
        for ch in token.children:
            if ch.dep_ == "obl":
                if any(c.dep_ == "case" and c.text.lower() == "a" for c in ch.children):
                    has_obl_a = True
                    break
        if has_obl_a:
            return "ir"
        # en el resto de declarativas: VER
        return "ver"

    # --- caso especial: formas 'fui/fue/fuiste/...'
    FU_SER_IR = {"fui","fue","fuiste","fuisteis","fuimos","fueron"}
    if form_noacc in FU_SER_IR:
        # si hay un OBL con 'a' => IR (p.ej. "Ana fue a Madrid")
        for ch in token.children:
            if ch.dep_ == "obl":
                if any(c.dep_ == "case" and c.text.lower() == "a" for c in ch.children):
                    return "ir"
        # si no, suele ser cópula SER (p.ej. "Ana fue doctora")
        return "ser"

    # --- overrides por forma (incluye tu CSV nuevo)
    forms_to_try = []
    if base_after_split:
        forms_to_try.append(base_after_split.lower())
        forms_to_try.append(_strip_acc(base_after_split))
    forms_to_try.append(form)
    for f in forms_to_try:
        if f in VERB_OV:
            return VERB_OV[f].lower()

    # --- imperativos irregulares de fallback (ven, haz, pon, ...)
    if form_noacc in IMP_FALLBACK:
        return IMP_FALLBACK[form_noacc]

    # --- lemma spaCy si es infinitivo
    lem = token.lemma_.lower()
    if lem.endswith(("ar", "er", "ir")):
        return lem

    # --- último recurso
    return _strip_acc(token.text)

def _to_gloss(token) -> str:
    def valid(g): return isinstance(g, str) and g.strip() != "" and str(g).lower() != "nan"

    if token.pos_ == "VERB":
        lemma = _best_verb_lemma(token)
        g = _lex_by_lemma(lemma, token.pos_)
        return g if valid(g) else lemma.upper()

    g = _lex_by_form(token.text.lower(), token.pos_)
    if valid(g):
        return g
    g = _lex_by_lemma(token.lemma_.lower(), token.pos_)
    if valid(g):
        return g

    if token.pos_ == "PROPN":
        return f"#{token.text.upper()}"
    return f"#{token.text.upper()}"

# --- Clíticos ---
def _split_enclitics(text: str) -> Tuple[str, List[str]]:
    if not CFG_CLITICS.get("split_enclitics", False):
        return text, []
    pronouns = ["les", "las", "los", "nos", "os", "lo", "la", "le", "me", "te", "se"]
    base = text
    found: List[str] = []
    changed = True
    while changed:
        changed = False
        for p in pronouns:
            if base.lower().endswith(p) and len(base) > len(p) + 1:
                base = base[:-len(p)]
                found.append(p)
                changed = True
                break
    return base, found

def _expand_clitic_token(tok_text_lower: str) -> List[str]:
    if tok_text_lower in CFG_CLITICS.get("indirect_object", {}):
        return CFG_CLITICS["indirect_object"][tok_text_lower].split()
    if tok_text_lower in CFG_CLITICS.get("direct_object", {}):
        return [CFG_CLITICS["direct_object"][tok_text_lower]]
    return []

# --- TIME/PLACE ---
def _collect_time_phrases(doc) -> Tuple[List[str], Set[int]]:
    out: List[str] = []
    used_idx: Set[int] = set()
    time_wordset = {w.lower() for w in CFG_LEXICAL.get("time_lex", [])}
    for t in doc:
        if t.ent_type_ in {"TIME", "DATE"} or t.text.lower() in time_wordset:
            out.append(t.text.upper())
            used_idx.add(t.i)

    # patrón 'a las/la <NUM>'
    for i, tok in enumerate(doc):
        if tok.pos_ == "NUM":
            j = i - 1
            has_det = False
            has_prep = False
            while j >= 0 and j >= i - 3:
                if doc[j].text.lower() in {"las", "la"}:
                    has_det = True
                if doc[j].text.lower() == "a":
                    has_prep = True
                j -= 1
            if has_prep and has_det:
                phrase = f"A LAS {tok.text.upper()}" if doc[i-1].text.lower() == "las" else f"A LA {tok.text.upper()}"
                out.append(phrase)
                used_idx.update({i, i-1, i-2})
    return out, used_idx

def _collect_place_phrases(doc) -> Tuple[List[str], Set[int]]:
    out: List[str] = []
    used_idx: Set[int] = set()
    place_preps = {p.lower() for p in CFG_LEXICAL.get("place_preps", [])}
    for t in doc:
        if t.dep_ == "obl" and t.head.pos_ == "VERB" and t.pos_ in {"NOUN", "PROPN"}:
            cases = [c for c in t.children if c.dep_ == "case" and c.pos_ == "ADP" and c.text.lower() in place_preps]
            prep = cases[0].text.upper() if cases else None
            noun_gloss = _to_gloss(t)
            if prep:
                out.append(f"{prep} {noun_gloss}")
                used_idx.add(t.i)
                used_idx.update([c.i for c in cases])
            else:
                out.append(noun_gloss)
                used_idx.add(t.i)
    return out, used_idx

# --- Predicado ---
def _collect_predicate(doc, used_tokens_idx: Set[int]) -> List[str]:
    seq: List[str] = []
    for t in doc:
        if t.i in used_tokens_idx:
            continue
        if not _should_keep(t):
            continue
        if t.is_punct:
            continue
        if t.pos_ in {"PRON", "DET"} and t.text.lower() in CLITIC_FORMS:
            continue
        seq.append(_to_gloss(t))
    return seq

# --- Sujeto inferido (imperativo) ---
def _infer_subject(doc, verb_tok):
    if verb_tok is not None:
        if "Imp" in verb_tok.morph.get("Mood"):
            return "TU"
        if _strip_accents(verb_tok.text.lower()) in IMP_FALLBACK:
            return "TU"
    return None

# --- Traductor ---
def translate_rule_based(text: str) -> str:
    doc = parse(text)

    subj_tok_or_str = find_subject(doc)
    verb = find_main_verb(doc)

    add_ojala = "ojala" in _noacc_lower(text)


    time_glosses, used_time = _collect_time_phrases(doc)
    place_glosses, used_place = _collect_place_phrases(doc)

    is_question = ("?" in text) or ("¿" in text)
    wh_tok_idx = None
    wh_gloss = None
    if is_question:
        for t in doc:
            k = _WH_MAP.get(t.text.lower())
            if k:
                wh_tok_idx = t.i
                wh_gloss = k
                break

    used_idx: Set[int] = set()
    used_idx.update(used_time)
    used_idx.update(used_place)
    if verb is not None:
        used_idx.add(verb.i)
    if hasattr(subj_tok_or_str, "i"):
        used_idx.add(subj_tok_or_str.i)
    if wh_tok_idx is not None:
        used_idx.add(wh_tok_idx)

    slots = {"TIME": [], "PLACE": [], "SUBJECT": [], "PRED": [], "VERB": []}
    slots["TIME"].extend([g for g in time_glosses if g])
    slots["PLACE"].extend([g for g in place_glosses if g])

    if isinstance(subj_tok_or_str, str):
        slots["SUBJECT"].append(subj_tok_or_str)
    elif subj_tok_or_str is not None:
        slots["SUBJECT"].append(_to_gloss(subj_tok_or_str))
    else:
        inferred = _infer_subject(doc, verb)
        if inferred:
            slots["SUBJECT"].append(inferred)

    if verb:
        base_txt, encl = _split_enclitics(verb.text)
        for p in reversed(encl):
            slots["PRED"].extend(_expand_clitic_token(p.lower()))
        lemma = _best_verb_lemma(verb, base_txt)
        g = _lex_by_lemma(lemma, "VERB")
        verb_gloss = g if isinstance(g, str) and g else lemma.upper()
        slots["VERB"].append(verb_gloss)

    slots["PRED"].extend(_collect_predicate(doc, used_idx))

    # Negación: NO junto al verbo si existe
    if any(t.text.lower() in NEG_SET for t in doc):
        if slots["VERB"]:
            if slots["VERB"][0] != "NO":
                slots["VERB"] = ["NO"] + slots["VERB"]
        else:
            slots["PRED"] = ["NO"] + slots["PRED"]

    # LINEARIZA
    order = CFG_REORDER.get("order", ["TIME","PLACE","SUBJECT","VERB","PRED"])
    seq = []
    for slot in order:
        seq.extend(slots.get(slot, []))

    # ---- AÑADE OJALA AL INICIO SI APARECE EN LA ENTRADA
    if add_ojala:
        seq = ["OJALA"] + seq

    if wh_gloss:
        seq.append(wh_gloss)

    seq = [s for s in seq if s]
    out = " ".join(seq)
    out = postprocess_gloss(out)
    return out
