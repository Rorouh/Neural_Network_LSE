import yaml
import pandas as pd
from pathlib import Path
import csv

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

def load_verb_overrides(path="data/lexicon/verb_lemma_override.csv"):
    """
    Formato esperado (sin cabecera):
      lemma,form1,form2,form3,...
    Devuelve dict: { form_lower: lemma_lower }
    """
    p = Path(path)
    if not p.exists():
        return {}

    overrides = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            lemma = row[0].strip().lower()
            if not lemma:
                continue
            # opcional: también mapear el propio lema → lema
            overrides.setdefault(lemma, lemma)
            for cell in row[1:]:
                form = (cell or "").strip().lower()
                if form:
                    overrides[form] = lemma
    return overrides