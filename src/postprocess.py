import re

def postprocess_gloss(s: str, is_question: bool = False, is_exclamation: bool = False) -> str:
    # Normaliza espacios
    s = re.sub(r"\s+", " ", s.strip())
    # A mayúsculas (glosas)
    s = " ".join(w.upper() for w in s.split())

    # Puntuación final (según flags)
    # Si por cualquier motivo estuvieran ambos a True, priorizamos '?'
    if is_question:
        if not s.endswith("?"):
            s = f"{s} ?"
    elif is_exclamation:
        if not s.endswith("!"):
            s = f"{s} !"

    # Quita posibles dobles signos finales duplicados
    s = re.sub(r"\s+([?!])\s*\1+$", r" \1", s)
    return s
