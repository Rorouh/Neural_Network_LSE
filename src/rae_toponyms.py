# src/rae_toponyms.py
import os, glob

def _norm_line(s: str):
    if not s:
        return None
    s = s.strip()
    if not s: 
        return None
    if s.startswith("#") or s.startswith("//"):
        return None
    return " ".join(s.split())

def _read_list(fp: str):
    out = []
    if not os.path.exists(fp):
        return out
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = _norm_line(line)
            if w:
                out.append(w)
    return out

def load_toponyms(rae_toponyms_root: str, include_world: bool = True):
    """
    rae_toponyms_root debe apuntar a:
      .../rla-es/ortografia/toponimos
    Carga:
      - l10n/es_ES/*.txt  (ccaa, provincias, municipios, localidades si existe)
      - toponimos-mundo.txt (si include_world=True)
    Devuelve lista de nombres tal cual (con su artículo si forma parte del nombre).
    """
    if not rae_toponyms_root or not os.path.isdir(rae_toponyms_root):
        raise FileNotFoundError(f"No existe la carpeta de toponimos: {rae_toponyms_root}")

    es_es_dir = os.path.join(rae_toponyms_root, "l10n", "es_ES")
    if not os.path.isdir(es_es_dir):
        raise FileNotFoundError(f"No existe l10n/es_ES dentro de: {rae_toponyms_root}")

    files = []
    # coge todos los .txt de es_ES (ccaa, provincias, municipios, localidades, etc.)
    files.extend(sorted(glob.glob(os.path.join(es_es_dir, "*.txt"))))

    # añade toponimos-mundo.txt (está en la raíz de 'toponimos')
    world_fp = os.path.join(rae_toponyms_root, "toponimos-mundo.txt")
    if include_world and os.path.exists(world_fp):
        files.append(world_fp)

    names = []
    seen = set()
    for fp in files:
        for w in _read_list(fp):
            if w not in seen:
                seen.add(w)
                names.append(w)
    return names

def build_places_from_toponyms(names, max_en: int = None, max_a: int = None, shuffle: bool = True):
    """
    Construye pares:
      - places_en: ["en Madrid", "en La Habana", ...]
      - places_a : ["a Madrid", "a La Habana", ...]
    Nota: NO contraemos a 'al' / 'a la' porque en topónimos el artículo suele formar parte del nombre ("a El Salvador").
    """
    import random
    lst = list(names)
    if shuffle:
        random.shuffle(lst)

    if max_en is not None:
        en_src = lst[:max_en]
    else:
        en_src = lst

    if max_a is not None:
        a_src = lst[:max_a]
    else:
        a_src = lst

    places_en = [f"en {x}" for x in en_src]
    places_a  = [f"a {x}"  for x in a_src]
    return places_en, places_a
