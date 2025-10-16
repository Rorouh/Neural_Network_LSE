# limpiar_c_deepseek.py
import re
from pathlib import Path

# Rutas (ajústalas si lo necesitas)
in_path = Path("C_deepseek.txt")
out_path = Path("C_deepseek_clean.txt")

# Coincide numeración al inicio de línea: "54. ", "54 ) ", "54.- ", etc.
NUM_PREFIX = re.compile(r'^\s*\d+\s*[\.\)\-–—]?\s*')

def limpiar_archivo(entrada: Path, salida: Path) -> None:
    oraciones_limpias = []
    # utf-8-sig elimina un posible BOM si lo hubiera
    with entrada.open("r", encoding="utf-8-sig") as f:
        for linea in f:
            s = linea.strip()
            if not s:                 # salta líneas vacías
                continue
            s = NUM_PREFIX.sub("", s)  # quita numeración inicial
            if s:                      # evita añadir vacíos accidentales
                oraciones_limpias.append(s)

    # Escribe una oración por línea, sin líneas en blanco
    with salida.open("w", encoding="utf-8") as f:
        f.write("\n".join(oraciones_limpias))

if __name__ == "__main__":
    if not in_path.exists():
        raise FileNotFoundError(f"No se encuentra {in_path.resolve()}")
    limpiar_archivo(in_path, out_path)
    print(f"Listo. Archivo limpio en: {out_path.resolve()}")
