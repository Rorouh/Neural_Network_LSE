# src/rae_lexicon.py
import os, json, csv, re
from typing import Dict, List, Tuple, Optional

# --- Normalización de POS (etiquetas de tu JSON/CSV) ---
def _norm_pos(tag: str) -> Optional[str]:
    if not tag:
        return None
    t = tag.strip().lower()

    # ruido que no queremos
    if "anticuado" in t or "desusado" in t:
        return None

    if t.startswith("sustantivo"):
        return "noun"
    if t.startswith("adjetivo"):
        return "adj"
    if t.startswith("adverbio"):
        return "adv"
    if "preposicion" in t:
        return "prep"
    if "pronombre" in t:
        return "pron"
    if t.startswith("verbo"):
        # clasificamos por subtipo si viene marcado
        if "tr+intr" in t:
            return "verb_mixed"
        if "transitivo" in t:
            return "verb_trans"
        if "intransitivo" in t:
            return "verb_intrans"
        if "pronominal" in t:
            return "verb_pron"
        return "verb_other"
    return None

_GENDER_RE = re.compile(r"sustantivo_(masculino|femenino|m\+f)")

def _infer_gender(tags: List[str]) -> str:
    # Devuelve 'm'/'f'/'u' (u = unknown/amb.)
    for t in tags:
        m = _GENDER_RE.search(t.lower())
        if m:
            g = m.group(1)
            if g == "masculino":
                return "m"
            if g == "femenino":
                return "f"
            if g == "m+f":
                return "u"
    return "u"

def _is_ok_lemma(w: str) -> bool:
    # Filtra acrónimos puros, tokens raros, espacios, etc.
    if not w or len(w) < 2:
        return False
    if " " in w:
        return False
    if re.search(r"[0-9/]", w):
        return False
    # deja mayúsculas tipo 'COVID' si quieres; si no, descomenta:
    # if w.isupper() and len(w) > 3: return False
    return True

def _read_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_csv(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            lemma = row["lema"]
            tag = row["pos_simple"]
            if not lemma:
                continue
            out.setdefault(lemma, []).append(tag)
    return out

def load_rae_lemma_to_pos(path: str) -> Dict[str, List[str]]:
    if path.endswith(".json"):
        return _read_json(path)
    if path.endswith(".csv"):
        return _read_csv(path)
    raise ValueError(f"Formato no soportado para {path}")

def build_pools(path: str) -> Dict[str, List]:
    """
    Devuelve:
      - nouns: List[Tuple[lemma, gender]]  (gender: 'm'|'f'|'u')
      - verbs_trans / verbs_intrans / verbs_mixed / verbs_pron / verbs_other: List[str] (infinitivos)
      - adjs: List[str]
      - advs: List[str]
    """
    raw = load_rae_lemma_to_pos(path)
    nouns: List[Tuple[str, str]] = []
    verbs_trans: List[str] = []
    verbs_intrans: List[str] = []
    verbs_mixed: List[str] = []
    verbs_pron: List[str] = []
    verbs_other: List[str] = []
    adjs: List[str] = []
    advs: List[str] = []

    for lemma, tags in raw.items():
        if not _is_ok_lemma(lemma):
            continue
        # normaliza a lista de etiquetas internas
        norm_tags = list(filter(None, (_norm_pos(t) for t in tags)))

        if not norm_tags:
            continue

        # Sustantivos (guarda género si lo tenemos)
        if any(t == "noun" for t in norm_tags):
            g = _infer_gender(tags)
            nouns.append((lemma, g))
            continue  # un lemma puede estar en más de 1 clase, pero priorizamos noun aquí

        # Verbos
        if "verb_trans" in norm_tags:
            verbs_trans.append(lemma)
        elif "verb_intrans" in norm_tags:
            verbs_intrans.append(lemma)
        elif "verb_mixed" in norm_tags:
            verbs_mixed.append(lemma)
        elif "verb_pron" in norm_tags:
            verbs_pron.append(lemma)
        elif "verb_other" in norm_tags:
            verbs_other.append(lemma)

        # Adjetivos / Adverbios
        if "adj" in norm_tags:
            adjs.append(lemma)
        if "adv" in norm_tags:
            advs.append(lemma)

    # dedup básicos
    def _uniq(xs): 
        seen=set(); out=[]
        for x in xs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    pools = {
        "nouns": nouns,  # [(lemma, gender)]
        "verbs_trans": _uniq(verbs_trans),
        "verbs_intrans": _uniq(verbs_intrans),
        "verbs_mixed": _uniq(verbs_mixed),
        "verbs_pron": _uniq(verbs_pron),
        "verbs_other": _uniq(verbs_other),
        "adjs": _uniq(adjs),
        "advs": _uniq(advs),
    }
    return pools
