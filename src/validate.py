import re

ARTS = {"EL","LA","LOS","LAS","UN","UNA","UNOS","UNAS"}

def has_articles(gloss: str) -> bool:
    return any(w in ARTS for w in gloss.split())

def ends_with_verb(gloss: str, verb_lex=set(("COMPRAR","VER","IR","VIVIR","QUERER","DORMIR","ESTUDIAR","LLAMAR"))):
    # muy aproximado: verbos conocidos
    if not gloss.strip(): return False
    return gloss.split()[-1] in verb_lex or gloss.endswith(" [WH]")

def score_gloss(gloss: str):
    s = gloss.strip()
    return {
        "no_articles": not has_articles(s),
        "verb_final": ends_with_verb(s),
    }
