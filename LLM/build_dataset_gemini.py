# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = r"""
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
   - ¿Salida en una sola línea por oración, en glosa?

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

""".strip()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_dataset_gemini.py — imprime primero la respuesta cruda (glosas separadas por '.')
---------------------------------------------------------------------------------------
- Lee N líneas no vacías de un .txt (una por línea).
- Envía en lotes (--batch-size) en un único prompt por lote.
- Pide al modelo UNA SOLA LÍNEA con EXACTAMENTE N glosas separadas por ". ".
- Imprime esa línea cruda por stdout (primero), la parsea por puntos y guarda CSV.
- Si --consume, borra del TXT solo las líneas procesadas OK (hace .bak salvo --no-backup).
"""

import os, sys, csv, time, argparse, shutil, re
from pathlib import Path
from typing import List, Tuple

try:
    from google import genai
    from google.genai import types
except Exception:
    print("ERROR: Instala la librería con: pip install -U google-genai", file=sys.stderr)
    raise

# ------------------ PROMPT DE SISTEMA (resumido; puedes sustituir por el tuyo) ------------------
SYSTEM_PROMPT = r"""
Eres lingüista experto en gramática del español y en adaptación a LSE escrita (glosa).
Transforma texto español a LSE escrita cumpliendo las reglas del proyecto. No añadas información
ni alteres el significado. Verbo SIEMPRE en infinitivo; eliminar ser/estar copulativos; unificar
tener/existir/estar-loc -> HABER; sin preposiciones ni determinantes; usar marcadores en MAYÚSCULAS.
NO OMITIR SUJETOS en las oraciones.
Una glosa por oración. Mantén el orden original de las oraciones del bloque.
""".strip()

# ------------------ utilidades ------------------
def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def select_first_nonempty_with_indices(path: str, limit: int) -> Tuple[List[str], List[int], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    texts, idxs = [], []
    for i, raw in enumerate(all_lines):
        s = raw.strip()
        if s:
            texts.append(s)
            idxs.append(i)
            if len(texts) >= limit:
                break
    return texts, idxs, all_lines

def rewrite_file_without_indices(path: str, all_lines: List[str], remove_idxs: List[int], create_backup: bool = True):
    if create_backup:
        try:
            shutil.copyfile(path, path + ".bak")
        except Exception as e:
            print(f"ADVERTENCIA: no se pudo crear .bak: {e}", file=sys.stderr)
    keep = []
    rem = set(remove_idxs)
    for i, line in enumerate(all_lines):
        if i not in rem:
            keep.append(line)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.writelines(keep)
    os.replace(tmp, path)

def build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)

def make_config(model_id: str, max_tokens: int):
    # Salida de texto plana (no JSON) para poder imprimirla tal cual.
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
    )

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

def append_rows_csv(path: str, rows: List[List[str]]):
    exists = Path(path).exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if not exists:
            w.writerow(["español", "transformacion_lse"])
        for r in rows:
            w.writerow(r)

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def split_by_periods(raw: str, expected: int) -> List[str]:
    """
    Parte por '.' evitando cortar '...'.
    Devuelve lista de longitud variable; el llamador validará contra 'expected'.
    """
    s = (raw or "")
    # protege puntos suspensivos
    s = s.replace("...", "…")
    # quitamos posibles saltos y normalizamos espacios extremos (pero no dentro)
    s = s.strip()
    # separa por punto+espacios o punto final
    parts = [p.strip() for p in re.split(r"\.\s+|\.$", s) if p.strip()]
    # revertir puntos suspensivos
    parts = [p.replace("…", "...") for p in parts]
    # Fallbacks suaves si no cuadra:
    if len(parts) != expected:
        # intenta por líneas
        alt = [p.strip() for p in s.splitlines() if p.strip()]
        if len(alt) == expected:
            return alt
        # intenta por ' || '
        if "||" in s:
            alt2 = [p.strip() for p in s.split("||") if p.strip()]
            if len(alt2) == expected:
                return alt2
    return parts

# ------------------ programa principal ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TXT con una oración por línea")
    ap.add_argument("--out", required=True, help="CSV de salida")
    ap.add_argument("--limit", type=int, default=20, help="N primeras oraciones a procesar")
    ap.add_argument("--batch-size", type=int, default=20, help="líneas por petición")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--qpm", type=int, default=1, help="peticiones por minuto (espaciado)")
    ap.add_argument("--consume", action="store_true", help="borrar del TXT las líneas procesadas OK")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--only-first-batch", action="store_true", help="procesa solo un lote por ejecución")
    args = ap.parse_args()

    # Claves API
    keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    keys = [k.strip() for k in keys_env.split(",") if k.strip()]
    if not keys:
        print("ERROR: Define GEMINI_API_KEYS o GEMINI_API_KEY.", file=sys.stderr)
        sys.exit(2)

    ensure_parent(args.out)
    texts, idxs, all_lines = select_first_nonempty_with_indices(args.input, args.limit)
    if not texts:
        print("No hay oraciones que procesar.", file=sys.stderr)
        sys.exit(0)

    delay = max(60.0 / max(1, args.qpm), 0.0)
    cfg = make_config(args.model, args.max_tokens)

    out_rows: List[List[str]] = []
    processed_ok_idxs: List[int] = []

    key_idx = 0
    client = build_client(keys[key_idx])
    batch = max(1, args.batch_size)

    groups_texts = list(chunk(texts, batch))
    groups_idxs  = list(chunk(idxs,  batch))

    if args.only_first_batch:
        groups_texts = groups_texts[:1]
        groups_idxs  = groups_idxs[:1]

    had_success = False

    for group_texts, group_idxs in zip(groups_texts, groups_idxs):
        n = len(group_texts)

        # Instrucción: UNA SOLA LÍNEA, glosas separadas por ". "
        instr = (
            "Convierte CADA línea de ENTRADA a LSE escrita (glosa) siguiendo las reglas del sistema. "
            f"Devuelve UNA SOLA LÍNEA con EXACTAMENTE {n} glosas en el MISMO orden, separadas por '. '. "
            "No añadas texto extra, ni encabezados, ni numeraciones. "
            "No utilices puntos dentro de cada glosa (solo separadores). "
            "Ejemplo de formato: G1. G2. G3.\n"
            "ENTRADA:\n"
        )
        payload = instr + "\n".join(group_texts)

        tries, ok, last_err = 0, False, None
        while tries < 5 and not ok:
            tries += 1
            t0 = time.time()
            try:
                resp = client.models.generate_content(
                    model=args.model,
                    contents=payload,
                    config=cfg,
                )
                raw = extract_text(resp)

                # === 1) IMPRIME PRIMERO LA RESPUESTA CRUDA EN PANTALLA ===
                # (sin adornos, tal cual llega)
                print(raw, flush=True)

                # === 2) Parseo por puntos ===
                outs = split_by_periods(raw, expected=n)

                if len(outs) != n:
                    cand = (getattr(resp, "candidates", None) or [None])[0]
                    fr = getattr(cand, "finish_reason", None)
                    raise RuntimeError(f"Conteo salida={len(outs)} != entrada={n} (finish_reason={fr})")

                # === 3) Guardar CSV y marcar procesadas ===
                for src_line, tgt_line in zip(group_texts, outs):
                    out_rows.append([src_line, " ".join(tgt_line.split())])
                processed_ok_idxs.extend(group_idxs)
                ok = True

            except Exception as e:
                last_err = e
                if len(keys) > 1:
                    key_idx = (key_idx + 1) % len(keys)
                    client = build_client(keys[key_idx])
                time.sleep(min(2 ** tries, 30))

            # Espaciado por QPM
            elapsed = time.time() - t0
            time.sleep(max(0.0, delay - elapsed))

        if not ok:
            sys.stderr.write(f"Fallo lote ({n}): {last_err}\n")
        else:
            had_success = True

    # CSV
    append_rows_csv(args.out, out_rows)

    if not had_success:
        print("Ningún lote se procesó con éxito.", file=sys.stderr)
        sys.exit(4)

    if args.consume and processed_ok_idxs:
        try:
            rewrite_file_without_indices(
                args.input,
                all_lines,
                processed_ok_idxs,
                create_backup=not args.no_backup
            )
            print(f"TXT actualizado: eliminadas {len(processed_ok_idxs)} líneas OK. Copia en {args.input}.bak", file=sys.stderr)
        except Exception as e:
            print(f"ERROR al reescribir el TXT: {e}", file=sys.stderr)
            sys.exit(3)

    print(f"Listo. Filas añadidas: {len(out_rows)}", file=sys.stderr)

if __name__ == "__main__":
    main()
