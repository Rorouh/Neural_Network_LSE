# System prompt: reglas de adaptación a LSE escrita (glosa)
SYSTEM_PROMPT = """
ROL
Eres lingüista experto en gramática del español y en adaptación a LSE escrita (glosa).

OBJETIVO
Transformar cualquier texto en español a LSE escrita cumpliendo estrictamente estas normas. No añadas información nueva ni alteres el significado.

ENTRADA (MÚLTIPLES ORACIONES)
- La entrada puede ser una o varias oraciones, separadas por “.”, “!”, “?”, “;” o saltos de línea.
- Segmenta cada oración de forma independiente. Ignora fragmentos vacíos (p. ej., dobles espacios o saltos de línea consecutivos).
- Conserva el orden original de las oraciones del bloque.

PRINCIPIOS GENERALES (OBLIGATORIOS)
1) VERBOS:
   - Nunca conjugues: siempre en INFINITIVO.
   - “ser/estar” copulativos se ELIMINAN (deja SUJETO + ATRIBUTO).
   - “estar (localización)”, “existir” y “tener” -> se unifican como la expresion "HAY".

2) OMISIONES:
   - Omite PREPOSICIONES.
   - Omite DETERMINANTES que tienen funcion de articulo. Conserva ADVERBIOS si aportan CCT/CCL u otras marcas relevantes.
   - Conserva nombres propios y numerales como datos concretos.
   - Los SUJETOS OMITIDOS no se omiten por lo que tienen que expresarse con su respectiva persona.


ORDEN BASE (PRIORIDAD)
A) Orden canónico LSE con foco en contexto:
   MARCA TEMPORAL (cuándo) + ADVERBIO DE LUGAR (dónde) + SUJETO + RESTO DEL PREDICADO [CD (qué), CI (a quién)] + (si NEGATIVA) PARTICULA NEGATIVA + VERBO (INFINITIVO) + (si PREGUNTA) PARTÍCULA INTERROGATIVA o SI/NO

B) Si NO hay MARCA TEMPORAL/ADVERBIO DE LUGAR visibles y/o el contexto obligue a más generalidad (limitación de datos o ambigüedad), usar orden general:
   ADVERBIO + SUJETO + PREDICADO + VERBO (INFINITIVO)

   Nota: “ADVERBIO” aquí actúa como marcador general inicial (p. ej., PASADO, FUTURO/PRÓXIMO, HOY, MAÑANA, ALLÍ, AQUÍ…).

MARCAS TEMPORALES (CUÁNDO)
- Si la oración NO incluye explícitamente el tiempo y NO es claramente presente, añadir al inicio:
  • "PASADO" -> para acciones pasadas o formas en pasado detectables.
  • "FUTURO" -> para acciones futuras (futuro morfológico, “ir a + inf.”, mañana, próximo…).
- Si es claramente presente y no hay CCT, NO añadir “PRESENTE”.

TIPOS DE ORACIÓN (REGLAS ESPECÍFICAS)
1) PASADO (sin CCT explícito):
   PASADO + (CCL si hay) + SUJETO + PREDICADO + VERBO + (Particula Negativa)

2) FUTURO (sin CCT explícito):
   FUTURO + (CCL si hay) + SUJETO + PREDICADO + VERBO + (Particula Negativa)

3) PRESENTE (sin CCT):
   Mantener sin añadir “PRESENTE”. Aplicar el ORDEN BASE.

4) NEGATIVA:
   — Regla preferente (canónica LSE por claridad): colocar “NO” inmediatamente despues del VERBO:
      CCT + CCL + SUJETO + PREDICADO + VERBO + NO
   — Regla alternativa (si la consigna del proyecto lo exige): colocar la PARTÍCULA NEGATIVA (“NO/NI/TAMPOCO/NUNCA”) al FINAL, tras el VERBO:
      ADVERBIO/CCT + SUJETO + PREDICADO + VERBO + PARTÍCULA NEGATIVA
   Si hay ambigüedad, aplicar la regla preferente.

5) POSITIVA / AFIRMATIVA:
   Aplicar ORDEN BASE sin marcas adicionales.

6) IMPERATIVA:
   Aplicar ORDEN BASE y añadir “DEBE” AL FINAL.
   Ej.: ADVERBIO/CCT + SUJETO + PREDICADO + VERBO + DEBE

7) SUBJUNTIVA:
   Añadir OJALÁ al INICIO y luego ORDEN BASE.
   Ej.: OJALÁ + CCT/ADVERBIO + SUJETO + PREDICADO + VERBO

8) EXCLAMATIVA:
   Mantener ORDEN BASE y añadir “!” al FINAL.
   Ej.: ADVERBIO/CCT + SUJETO + PREDICADO + VERBO + !

9) INTERROGATIVA:
   — Sí/No: añadir “SI/NO” al FINAL.
   — Interrogativa parcial: colocar la PARTÍCULA INTERROGATIVA al FINAL (QUÉ, QUIÉN, DÓNDE, CUÁNDO, CUÁL, POR QUÉ, CUÁNTO/OS/AS, CÓMO).
   Ademas, añadir “?” final si el canal lo permite, sin reemplazar la partícula.

NORMALIZACIÓN DE COMPLEMENTOS
- CCL sin preposiciones: Cuando existe un CCL, se cambie el orden siguiendo la regla de "Sujeto" + "Objeto"... Por ejemplo: “en la casa de Ana” -> “ANA CASA”; “al instituto” -> “INSTITUTO”.
- Otros complementos sin preposiciones ni determinantes: “con mi amigo” -> “AMIGO MÍO”.
- “tener/existir/estar (lugar)” -> se transforma a la palabra "HAY" (p. ej., “no tengo dinero” -> “YO DINERO NO HAY si negativa, o “YO DINERO HAY" si afirmativa).

FORMATO DE SALIDA
- Devuelve **SOLO glosas**, **una línea por cada oración** de la entrada, en el **mismo orden**.
- Sin explicaciones, sin comillas, sin numeración, sin bloques de código.

PROCEDIMIENTO PASO A PASO
1) Segmentar el bloque de entrada en oraciones (., !, ?, ;, saltos de línea). Ignorar fragmentos vacíos.
2) Para cada oración: detectar tiempo (CCT o inferencia PASADO/FUTURO), lugar (CCL), sujeto, CD/CI, negación y modalidad (enunciativa/negativa/interrogativa/exclamativa/imperativa/subjuntiva).
3) Si falta CCT y no es presente, anteponer PASADO o FUTURO dependiendo de el tiempo verbal correspondiente de la oracion.
4) Reescribir CCL y demás complementos sin preposiciones/determinantes.
5) Aplicar mapeo de verbos: eliminar “ser/estar” copulativos; unificar tener/existir/estar-loc en "HAY".
6) Conservar INFINITIVO para TODOS los verbos.
7) Ensamblar usando el ORDEN BASE (CCT + CCL + SUJETO + PREDICADO + VERBO + (NO) + [marcador final de modalidad]).
   — Si el contexto obliga a generalidad por escasez de CCT/CCL, usar la estructura general: ADVERBIO + SUJETO + PREDICADO + VERBO.
8) Modalidades: añadir los marcadores según el bloque “TIPOS DE ORACIÓN”.
9) Emitir la glosa final de esa oración en una línea. Repetir para cada oración.
10) Auto-chequeo final para cada línea:
   - ¿Orden correcto?
   - ¿Sin preposiciones ni determinantes?
   - ¿Verbos en infinitivo?
   - ¿“ser/estar” copulativos eliminados y “tener/existir/estar-loc” -> HABER?
   - ¿Marca temporal añadida cuando faltaba y no era presente?
   - ¿Negación y/o interrogativa/exclamativa aplicadas en el lugar correcto?
   - ¿Salida en una sola línea por oración?

EJEMPLOS CANÓNICOS
1) Entrada: “Mi hermano está en casa de Ana.”
   -> CASA ANA MI HERMANO 

2) “No tengo dinero.”
   (preferente) -> YO DINERO NO HAY

3) “¿Vas a venir mañana a mi casa?”
   -> MAÑANA CASA MÍA TÚ VENIR SI/NO?

4) “¿Qué compraste en la tienda?”
   -> PASADO TIENDA TÚ COMPRAR QUÉ?

5) “Luis es médico.”
   -> LUIS MÉDICO

6) Imperativa: “Cierra la puerta.”
   -> PUERTA TÚ CERRAR DEBE

7) Subjuntiva: “Ojalá llueva mañana.”
   -> OJALÁ MAÑANA LLOVER

8) Exclamativa: “¡Qué bonito es este lugar!”
   -> LUGAR ESTE BONITO !

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lse_gemini_to_csv.py
--------------------
- Lee las primeras N líneas NO vacías de un TXT (una oración por línea).
- Envía TODAS esas líneas en UNA sola petición a Gemini.
- Pide que la salida venga en UNA ÚNICA LÍNEA con glosas separadas por ". " (y punto final).
- Parsea la salida, empareja (español, glosa) y apendea a un CSV.
- Si todo fue OK (conteo coincide), elimina del TXT las líneas procesadas (backup .bak).

Uso:
  pip install -U google-genai
  (PowerShell) $env:GEMINI_API_KEY="TU_CLAVE"
  python lse_gemini_to_csv.py --input entrada1.txt --out dataset.csv --limit 20 \
         --model gemini-2.5-flash --max-tokens 1024
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from google import genai
from google.genai import types


# ---- Utilidades de fichero ----
def read_first_nonempty_with_indices(path: str, limit: int) -> Tuple[List[str], List[int], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    texts, idxs = [], []
    for i, raw in enumerate(all_lines):
        s = raw.strip()
        if not s:
            continue
        texts.append(s)
        idxs.append(i)
        if len(texts) >= limit:
            break
    return texts, idxs, all_lines

def rewrite_file_without_indices(path: str, all_lines: List[str], remove_idxs: List[int], create_backup: bool = True):
    if create_backup:
        try:
            import shutil
            shutil.copyfile(path, path + ".bak")
        except Exception as e:
            print(f"ADVERTENCIA: no se pudo crear .bak: {e}", file=sys.stderr)
    keep = []
    remove = set(remove_idxs)
    for i, line in enumerate(all_lines):
        if i not in remove:
            keep.append(line)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.writelines(keep)
    os.replace(tmp, path)

def append_rows_csv(path: str, rows: List[List[str]]):
    new = not Path(path).exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if new:
            w.writerow(["oracion_español", "oracion_transformada_lse"])
        w.writerows(rows)

# ---- Gemini ----
def build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)

def make_config(model_id: str, max_tokens: int):
    cfg = dict(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
    )
    # Desactiva thinking (seguro para flash y para mantener tokens de salida bajo control)
    try:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass
    return types.GenerateContentConfig(**cfg)

def extract_text(resp) -> str:
    txt = (getattr(resp, "text", None) or "").strip()
    if txt:
        return txt
    for c in getattr(resp, "candidates", []) or []:
        content = getattr(c, "content", None)
        if not content:
            continue
        for p in getattr(content, "parts", []) or []:
            t = getattr(p, "text", None)
            if t and t.strip():
                return t.strip()
    return ""

def first_finish_reason(resp) -> Optional[str]:
    cands = getattr(resp, "candidates", None) or []
    if cands:
        fr = getattr(cands[0], "finish_reason", None)
        if fr:
            return str(fr)
    return None

# ---- Parsing de la línea: "G1. G2. ... Gn." → [G1, G2, ..., Gn]
def split_glosas_dot_line(one_line: str, expected: int) -> List[str]:
    s = (one_line or "").strip()
    if not s:
        return []
    # proteger "..." para no dividir mal
    s = s.replace("...", "…")
    # dividir por ". " o punto final
    import re
    parts = [p.strip() for p in re.split(r"\.\s+|\.$", s) if p.strip()]
    parts = [p.replace("…", "...") for p in parts]
    # normalizar espacios
    parts = [" ".join(p.split()) for p in parts]
    # si el modelo puso un punto al final sin espacio, ya lo cubre el regex
    # devolvemos tal cual (el conteo se validará fuera)
    return parts

# ---- Pipeline principal ----
def transform_batch_to_csv(input_txt: str, out_csv: str, limit: int, model: str, max_tokens: int, api_key: str) -> int:
    # 1) leer N líneas e índices
    sentences, idxs, all_lines = read_first_nonempty_with_indices(input_txt, limit)
    if not sentences:
        print("No hay líneas no vacías que procesar.", file=sys.stderr)
        return 0

    n = len(sentences)

    # 2) construir instrucción + payload
    instr = (
        "Transforma CADA línea de ENTRADA a LSE escrita (glosa), siguiendo estrictamente las reglas del sistema.\n"
        f"Devuelve TODO en **UNA ÚNICA LÍNEA**, con EXACTAMENTE {n} glosas separadas por '. ' y terminando con '.'.\n"
        "Sin comillas, sin numeración, sin texto extra.\n"
        "Ejemplo de formato de salida: GLOSA1. GLOSA2. GLOSA3.\n"
        "ENTRADA:\n"
    )
    payload = instr + "\n".join(sentences)

    # 3) llamada a Gemini
    client = build_client(api_key)
    cfg = make_config(model, max_tokens)

    resp = client.models.generate_content(model=model, contents=payload, config=cfg)
    out_line = extract_text(resp)
    if not out_line:
        fr = first_finish_reason(resp) or "N/A"
        raise RuntimeError(f"Respuesta vacía del modelo (finish_reason={fr}). "
                           f"Prueba reduciendo --limit o subiendo --max-tokens si tu plan lo permite.")

    # 4) parsear a lista de glosas
    glosas = split_glosas_dot_line(out_line, expected=n)

    if len(glosas) != n:
        fr = first_finish_reason(resp) or "N/A"
        # No tocar ficheros si no coincidimos
        raise RuntimeError(f"Conteo de glosas={len(glosas)} != líneas entrada={n} (finish_reason={fr}). "
                           f"Posibles causas: límite de tokens, el modelo añadió texto extra o faltan puntos. "
                           f"Salida recibida:\n{out_line}")

    # 5) emparejar y escribir CSV
    rows = [[src, tgt] for src, tgt in zip(sentences, glosas)]
    append_rows_csv(out_csv, rows)

    # 6) borrar del TXT las líneas procesadas (por índice)
    rewrite_file_without_indices(input_txt, all_lines, idxs, create_backup=True)

    return len(rows)

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Convierte las primeras N líneas de un TXT a LSE (glosa), guarda CSV y consume el TXT.")
    ap.add_argument("--input", required=True, help="Ruta al TXT (una oración por línea). Ej.: entrada1.txt")
    ap.add_argument("--out", required=True, help="CSV de salida. Ej.: dataset.csv")
    ap.add_argument("--limit", type=int, default=20, help="N líneas no vacías a tomar (por defecto 20)")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), help="Modelo Gemini (p. ej. gemini-2.5-flash)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="max_output_tokens para la salida")
    ap.add_argument("--api-key", help="API key si no usas GEMINI_API_KEY / GOOGLE_API_KEY")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Falta la API key. Define GEMINI_API_KEY/GOOGLE_API_KEY o pasa --api-key.", file=sys.stderr)
        sys.exit(2)

    try:
        count = transform_batch_to_csv(
            input_txt=args.input,
            out_csv=args.out,
            limit=args.limit,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=api_key,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"OK. Filas añadidas: {count}. TXT actualizado y {args.input}.bak creado.")

if __name__ == "__main__":
    main()
