import spacy
from spacy.tokens import Doc
from .util import load_yaml

NLP = spacy.load("es_core_news_md")

CFG_REORDER = load_yaml("rules/reorder.yml")
CFG_LEXICAL = load_yaml("rules/lexical.yml")
CFG_DROP    = load_yaml("rules/drop.yml")

TIME_WORDS  = set(CFG_LEXICAL.get("time_lex", []))
PLACE_PREPS = set(CFG_LEXICAL.get("place_preps", []))
NEG_WORDS   = set(CFG_LEXICAL.get("neg_lex", []))

def parse(text: str) -> Doc:
    return NLP(text)

def is_time_token(tok):
    t = tok.text.lower()
    return t in {w.lower() for w in TIME_WORDS} or tok.ent_type_ in {"TIME","DATE"}

def is_place_phrase(tok):
    return tok.dep_ == "obl" and tok.head.pos_ == "VERB"

def find_subject(doc):
    for t in doc:
        if t.dep_.startswith("nsubj"):
            return t
    # inferencias solo si la morfología lo dice claramente
    for t in doc:
        if t.pos_ == "VERB":
            if "Person=1" in t.morph:
                return "YO"
            if "Person=2" in t.morph:
                return "TU"
    return None

def find_object(doc):
    for t in doc:
        if t.dep_ in {"obj","iobj"}:
            return t
    return None

def find_main_verb(doc):
    for t in doc:
        if t.pos_ == "VERB" and t.head == t:
            return t
    for t in doc:
        if t.pos_ == "VERB":
            return t
    return None

def collect_time(doc):
    return [t for t in doc if is_time_token(t)]

def collect_place(doc):
    return [t for t in doc if is_place_phrase(t)]

def detect_tense_mood_polarity(doc: Doc):
    tense = None
    mood = None
    polarity = "Aff"
    txt = doc.text

    is_question = "?" in txt
    is_exclamation = "!" in txt

    if any(t.text.lower() == "ojalá" for t in doc):
        mood = "Subj"

    for t in doc:
        if t.pos_ == "VERB":
            morph = t.morph
            if "Tense=Past" in morph and not tense:   tense = "Past"
            elif "Tense=Fut" in morph and not tense:  tense = "Future"
            if "Mood=Sub" in morph:  mood = "Subj"
            elif "Mood=Imp" in morph: mood = "Imp"
        if t.text.lower() in {w.lower() for w in NEG_WORDS}:
            polarity = "Neg"

    if not tense: tense = "Present"

    # Heurística adicional de imperativo
    if mood is None:
        root = [t for t in doc if t.head == t]
        if root and root[0].pos_ == "VERB" and not any(tt.dep_.startswith("nsubj") for tt in doc):
            mood = "Imp"

    return tense, mood, polarity, (is_question, is_exclamation)

def subj_person_from_token_or_str(subj):
    if isinstance(subj, str):
        return {"YO":1, "TU":2, "NOSOTROS":1, "VOSOTROS":2}.get(subj)
    if subj is None: return None
    if "Person=1" in subj.morph: return 1
    if "Person=2" in subj.morph: return 2
    if "Person=3" in subj.morph: return 3
    return None
