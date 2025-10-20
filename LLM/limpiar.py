# limpiar.py
import re
from pathlib import Path

# Carpeta raíz del proyecto (por defecto, el directorio donde ejecutes el script)
ROOT = Path(__file__).parent  # o Path("/ruta/a/LLM") si quieres fijarla

# Opcional: carpetas/archivos a excluir (por si usas .venv, .git, etc.)
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", "node_modules"}
EXCLUDE_FILES = set()  # {"entrada/ignorar.txt", ...}

# Regex: línea que ya termina en signo de puntuación fuerte
ENDS_WITH_PUNCT = re.compile(r'[.!?…]\s*$')

def is_interrogative_or_exclamative(s: str) -> bool:
    # Si contiene ¿ ? ¡ ! en cualquier lugar, la consideramos interrogativa/exclamativa
    return any(ch in s for ch in ("?", "¿", "!", "¡"))

def normalize_line(s: str) -> str:
    # Quita espacios laterales y colapsa espacios internos múltiples a simples.
    # Si no quieres colapsar espacios internos, comenta la siguiente línea:
    s = " ".join(s.strip().split())
    return s

def ensure_trailing_dot_if_needed(s: str) -> str:
    """
    Si la línea NO es interrogativa/exclamativa y NO termina en . ! ? …,
    añade un punto final.
    """
    if not s:
        return s
    if is_interrogative_or_exclamative(s):
        return s
    if ENDS_WITH_PUNCT.search(s):
        return s
    return s + "."

def clean_txt_file(path: Path) -> tuple[int, int]:
    """
    Limpia un .txt in-place:
      - elimina líneas en blanco
      - normaliza espacios laterales
      - añade punto final si la línea no es ?/! y no acaba en .!?…
    Devuelve (n_entradas, n_salidas)
    """
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            continue  # salta líneas en blanco
        s = normalize_line(s)
        s = ensure_trailing_dot_if_needed(s)
        cleaned.append(s)

    # Escribe de vuelta al mismo fichero
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(cleaned))
        if cleaned and not cleaned[-1].endswith("\n"):
            f.write("\n")

    return len(lines), len(cleaned)

def iter_txt_files(root: Path):
    for p in root.rglob("*.txt"):
        # Excluir carpetas no deseadas
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        if rel in EXCLUDE_FILES:
            continue
        yield p

def main():
    total_in, total_out, total_files = 0, 0, 0
    for txt in iter_txt_files(ROOT):
        n_in, n_out = clean_txt_file(txt)
        total_in += n_in
        total_out += n_out
        total_files += 1
        print(f"✓ {txt}: {n_in} → {n_out} líneas")

    print("\nResumen:")
    print(f"  Ficheros procesados: {total_files}")
    print(f"  Total líneas (antes): {total_in}")
    print(f"  Total líneas (después): {total_out}")

if __name__ == "__main__":
    main()
