import yaml
import pandas as pd
from pathlib import Path
import csv
import os, csv, unicodedata

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_lexicon_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    by_lemma, by_form = {}, {}
    def norm(s): 
        s = "" if s is None else str(s)
        return "" if s.lower() == "nan" else s

    for _, r in df.iterrows():
        es_lemma = norm(r.get("es_lemma",""))
        es_form  = norm(r.get("es_form",""))
        upos     = norm(r.get("upos",""))
        gloss    = norm(r.get("lse_gloss",""))
        if es_lemma: by_lemma[(es_lemma.lower(), upos)] = gloss
        if es_form:  by_form[(es_form.lower(),  upos)]   = gloss
    return {"by_lemma": by_lemma, "by_form": by_form}


def _normalize_ascii_lower(s: str) -> str:
    # 'Dámelo' -> 'damelo'
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower().strip()

def load_verb_overrides(path: str = "data/lexicon/verb_lemma_override.csv"):
    """
    Lee un CSV 'ancho' sin cabeceras:
      infinitivo,forma1,forma2,forma3,...
    Devuelve un dict que mapea cualquier 'forma' al 'infinitivo'.
    Inserta tanto la forma original (lower) como su versión sin tildes.
    También mapea el propio infinitivo a sí mismo (útil como respaldo).
    """
    mapping = {}
    if not os.path.exists(path):
        return mapping

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            lemma = (row[0] or "").strip().lower()
            if not lemma:
                continue
            # mapea el infinitivo a sí mismo
            if lemma not in mapping:
                mapping[lemma] = lemma
            na_lemma = _normalize_ascii_lower(lemma)
            if na_lemma not in mapping:
                mapping[na_lemma] = lemma

            # mapea todas las formas a ese infinitivo
            for form in row[1:]:
                form = (form or "").strip()
                if not form:
                    continue
                lf = form.lower()
                naf = _normalize_ascii_lower(lf)
                if lf not in mapping:
                    mapping[lf] = lemma
                if naf not in mapping:
                    mapping[naf] = lemma
    return mapping
