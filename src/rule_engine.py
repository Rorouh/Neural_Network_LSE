# src/rule_engine.py
from typing import List, Tuple, Dict, Set
from .util import load_yaml, load_lexicon_csv, load_verb_overrides
from .features import parse, find_subject, find_object, find_main_verb, collect_time, collect_place
from .postprocess import postprocess_gloss

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

# --- WH map para normalizar interrogativos y evitar duplicados ---
_WH_MAP: Dict[str, str] = {}
for k, forms in CFG_LEXICAL.get("wh_lex", {}).items():
    for f in forms:
        _WH_MAP[f.lower()] = k

LEX: Dict[str, Dict] = load_lexicon_csv("data/lexicon/es_to_lse.csv")
VERB_OV = load_verb_overrides("data/lexicon/verb_lemma_override.csv")

ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
NEG_SET: Set[str] = {w.lower() for w in CFG_LEXICAL.get("neg_lex", [])}
CLITIC_FORMS: Set[str] = {"me", "te", "se", "le", "les", "nos", "os", "lo", "la", "los", "las"}

# ---------- Helpers básicos ----------
def _is_article(tok): return tok.pos_ == "DET" and tok.text.lower() in ARTICLES
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
    # lo filtramos para evitar fallback #OJALÁ; lo añadiremos estructuralmente como OJALA
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

def _strip_accents(s: str) -> str:
    tr = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
    return s.translate(tr)

# Imperativos irregulares muy frecuentes (fallback)
IMP_FALLBACK = {
    "ven": "venir",
    "pon": "poner",
    "ten": "tener",
    "haz": "hacer",
    "di": "decir",
    "sal": "salir",
    "ve": "ir",   # podría ser 'ver', pero para imperativo preferimos IR
    "se": "ser",
    "sé": "ser",
    "da": "dar",
}

def _best_verb_lemma(token, base_after_split: str = None) -> str:
    # 1) overrides por forma
    forms_to_try = []
    if base_after_split:
        forms_to_try.append(base_after_split.lower())
        forms_to_try.append(_strip_accents(base_after_split.lower()))
    forms_to_try.append(token.text.lower())

    # 2) imperativos irregulares (considera también la forma del propio token)
    form_noacc = _strip_accents(token.text.lower())
    if form_noacc in IMP_FALLBACK:
        return IMP_FALLBACK[form_noacc]

    for f in forms_to_try:
        if f in VERB_OV:
            return VERB_OV[f].lower()
    # 3) lemma de spaCy si parece infinitivo
    lem = token.lemma_.lower()
    if lem.endswith(("ar", "er", "ir")):
        return lem
    # 4) fallback
    return _strip_accents(token.text).lower()

def _to_gloss(token) -> str:
    """Devuelve glosa (string). Fallback: dactilología '#FORM'."""
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

# ---------- Enclíticos ----------
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
    return base, found  # found = derecha->izquierda

def _expand_clitic_token(tok_text_lower: str) -> List[str]:
    if tok_text_lower in CFG_CLITICS.get("indirect_object", {}):
        return CFG_CLITICS["indirect_object"][tok_text_lower].split()
    if tok_text_lower in CFG_CLITICS.get("direct_object", {}):
        return [CFG_CLITICS["direct_object"][tok_text_lower]]
    return []

# ---------- TIME / PLACE frases ----------
def _collect_time_phrases(doc) -> Tuple[List[str], Set[int]]:
    out: List[str] = []
    used_idx: Set[int] = set()
    # 1) expresiones simples por léxico o ent_type
    time_wordset = {w.lower() for w in CFG_LEXICAL.get("time_lex", [])}
    for t in doc:
        if t.ent_type_ in {"TIME", "DATE"} or t.text.lower() in time_wordset:
            out.append(t.text.upper())
            used_idx.add(t.i)
    # 2) patrón 'a las <NUM>' / 'a la una'
    for i, tok in enumerate(doc):
        if tok.pos_ == "NUM":
            j = i - 1
            has_det, has_prep = False, False
            while j >= 0 and j >= i - 3:
                if doc[j].text.lower() in {"las", "la"}:
                    has_det = True
                if doc[j].text.lower() == "a":
                    has_prep = True
                j -= 1
            if has_prep and has_det:
                det = "LAS" if doc[i - 1].text.lower() == "las" else "LA"
                out.append(f"A {det} {tok.text.upper()}")
                used_idx.update({i, i - 1, i - 2})
    return out, used_idx

def _collect_place_phrases(doc) -> Tuple[List[str], Set[int]]:
    out: List[str] = []
    used_idx: Set[int] = set()
    place_preps = {p.lower() for p in CFG_LEXICAL.get("place_preps", [])}
    # 'obl' nominal con preposición hija 'case'
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

# ---------- Predicado ----------
def _collect_predicate(doc, used_tokens_idx: Set[int]) -> List[str]:
    seq: List[str] = []
    for t in doc:
        if t.i in used_tokens_idx:
            continue
        if not _should_keep(t):
            continue
        if t.is_punct:
            continue
        # no añadir clíticos sueltos (los expandimos aparte)
        if t.pos_ in {"PRON", "DET"} and t.text.lower() in CLITIC_FORMS:
            continue
        seq.append(_to_gloss(t))
    return seq

# ---------- Sujeto ----------
def _infer_subject(doc, verb_tok):
    if verb_tok is not None:
        if "Imp" in verb_tok.morph.get("Mood"):
            return "TU"
        if _strip_accents(verb_tok.text.lower()) in IMP_FALLBACK:
            return "TU"
    return None

# ---------- Traductor ----------
def translate_rule_based(text: str) -> str:
    doc = parse(text)

    # Roles base
    subj_tok_or_str = find_subject(doc)
    verb = find_main_verb(doc)

    # TIME / PLACE
    time_glosses, used_time = _collect_time_phrases(doc)
    place_glosses, used_place = _collect_place_phrases(doc)

    # ¿Pregunta? + WH (para evitar #CÓMO en predicado)
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

    # OJALÁ presente (lo añadiremos estructuralmente)
    has_ojala = any(t.text.lower() == "ojalá" or t.lemma_.lower() == "ojalá" for t in doc)

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

    # TIME / PLACE
    slots["TIME"].extend([g for g in time_glosses if g])
    slots["PLACE"].extend([g for g in place_glosses if g])

    # SUBJECT (con forzado para imperativo)
    is_imp = False
    if verb is not None:
        is_imp = ("Imp" in verb.morph.get("Mood")) or (_strip_accents(verb.text.lower()) in IMP_FALLBACK)

    subj_gloss = None
    if isinstance(subj_tok_or_str, str):
        subj_gloss = subj_tok_or_str
    elif subj_tok_or_str is not None:
        subj_gloss = _to_gloss(subj_tok_or_str)
    else:
        subj_gloss = _infer_subject(doc, verb)

    if is_imp and subj_gloss != "TU":
        subj_gloss = "TU"

    if subj_gloss:
        slots["SUBJECT"].append(subj_gloss)

    # VERBO + enclíticos pegados
    if verb:
        base_txt, encl = _split_enclitics(verb.text)
        base_txt_noacc = _strip_accents(base_txt)
        # expansiones de enclíticos del verbo (derecha->izquierda)
        for p in reversed(encl):
            slots["PRED"].extend(_expand_clitic_token(p.lower()))
        # glosa del verbo con mejor lemma
        lemma = _best_verb_lemma(verb, base_txt_noacc)
        g = _lex_by_lemma(lemma, "VERB")
        verb_gloss = g if isinstance(g, str) and g else lemma.upper()
        slots["VERB"].append(verb_gloss)

    # PRED (resto, sin clíticos)
    slots["PRED"].extend(_collect_predicate(doc, used_idx))

    # Clíticos sueltos (ME/TE/LO/...), solo si POS = PRON (evitar DET: "la escuela")
    for t in doc:
        if t.i in used_idx:
            continue
        if t.pos_ != "PRON":
            continue
        low = t.text.lower()
        if low in CLITIC_FORMS:
            slots["PRED"].extend(_expand_clitic_token(low))


    # Negación: NO delante del verbo si aparece
    if any(t.text.lower() in NEG_SET for t in doc):
        if slots["VERB"]:
            if slots["VERB"][0] != "NO":
                slots["VERB"] = ["NO"] + slots["VERB"]
        else:
            slots["PRED"] = ["NO"] + slots["PRED"]

    # Orden final
    order = CFG_REORDER.get("order", ["TIME", "PLACE", "SUBJECT", "VERB", "PRED"])
    seq: List[str] = []
    for slot in order:
        seq.extend(slots.get(slot, []))

    # OJALA al inicio si aparece
    if has_ojala:
        seq = ["OJALA"] + seq

    # WH al final si hay pregunta
    if wh_gloss:
        seq.append(wh_gloss)

    # Limpieza y postproceso
    seq = [s for s in seq if s]
    out = " ".join(seq)
    out = postprocess_gloss(out)
    return out
