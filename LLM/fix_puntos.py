# fix_puntos.py
from pathlib import Path

RUTA = Path("entrada1.txt")

CLOSERS = set('"\'”’»)]}›')     # caracteres de cierre comunes
END_PUNCT = set(".?!…")         # signos válidos al final

def add_period_if_missing(line: str) -> str:
    """
    Añade un punto final a la 'oración' contenida en la línea si no termina
    en . ? ! … (considerando cierres como comillas o paréntesis).
    Mantiene los saltos de línea originales.
    """
    if not line:
        return line

    # Separa saltos de línea finales para preservarlos
    stripped_line = line.rstrip("\r\n")
    linebreaks = line[len(stripped_line):]  # '\n', '\r\n', '', etc.

    # Si la línea está en blanco, devolver tal cual
    core = stripped_line.rstrip()
    if core == "":
        return line

    # Mover el índice hacia atrás saltando cierres tipo comillas/paréntesis
    i = len(core) - 1
    while i >= 0 and core[i] in CLOSERS:
        i -= 1

    # Si ahora no hay caracteres visibles, devolver tal cual
    if i < 0:
        return stripped_line + linebreaks

    # Si ya termina con puntuación final válida, devolver tal cual
    if core[i] in END_PUNCT:
        return stripped_line + linebreaks

    # Insertar el punto antes del bloque de cierres finales
    # core = [hasta i] + '.' + [desde i+1]
    core_with_dot = core[:i+1] + "." + core[i+1:]

    # Reconstruir con los espacios finales (si había) y saltos de línea
    trailing_spaces = stripped_line[len(core):]
    return core_with_dot + trailing_spaces + linebreaks


def main():
    # Leer el contenido original
    texto = RUTA.read_text(encoding="utf-8")

    # Procesar línea a línea (preservando los saltos de línea tal cual)
    lineas = texto.splitlines(keepends=True)
    lineas_fijas = [add_period_if_missing(l) for l in lineas]

    # Si el archivo no terminaba en salto de línea, splitlines(keepends=True)
    # no añade nada; también lo respetamos.
    texto_final = "".join(lineas_fijas)

    # Guardar en el mismo fichero
    RUTA.write_text(texto_final, encoding="utf-8")


if __name__ == "__main__":
    main()
