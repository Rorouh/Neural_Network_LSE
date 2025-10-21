#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Une todos los .txt de la carpeta actual en 'entrada2.txt'.
- Mantiene el orden alfabético de los ficheros.
- Excluye 'entrada2.txt' si ya existe.
- Garantiza un salto de línea al final de cada fichero concatenado.
"""

import os
import glob

OUTPUT = "entrada2.txt"

def main():
    # Todos los .txt de la carpeta, menos el de salida
    txt_files = sorted(
        f for f in glob.glob("*.txt")
        if os.path.basename(f) != OUTPUT
    )

    if not txt_files:
        print("No se encontraron ficheros .txt para unir.")
        return

    # Escribe el resultado
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as out:
        for path in txt_files:
            with open(path, "r", encoding="utf-8", errors="ignore") as inp:
                content = inp.read()
            # Asegura que cada fichero termina en salto de línea
            if content and not content.endswith("\n"):
                content += "\n"
            out.write(content)

    print(f"Unidos {len(txt_files)} ficheros en '{OUTPUT}'.")

if __name__ == "__main__":
    main()
