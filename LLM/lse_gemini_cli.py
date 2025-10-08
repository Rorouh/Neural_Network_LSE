#!/usr/bin/env python3
"""
lse_gemini_cli.py
Pide una oración por consola y la transforma a LSE escrita usando la Gemini API.

Uso:
  1) pip install -U google-genai
  2) export GEMINI_API_KEY="TU_CLAVE"
  3) python lse_gemini_cli.py --model gemini-2.5-pro
     (o sin --model para usar el valor por defecto)

Modelos recomendados:
  - gemini-2.5-pro   (máxima calidad)
  - gemini-2.5-flash (rápido y económico)
  - gemini-1.5-pro-latest (fallback si 2.5 no está disponible)
"""

import os
import sys
import json
import argparse
from typing import Optional
from google import genai
from google.genai import types

API_KEY = "AIzaSyD6itqASIKL83fJx7T6FQ389nEOfQfo_Yk"

# ----- Configuración -----
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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
   - “estar (localización)”, “existir” y “tener” → se unifican como VERBO “HABER”.

2) OMISIONES:
   - Omite PREPOSICIONES.
   - Omite DETERMINANTES (artículos, demostrativos, posesivos, cuantificadores, numerales, etc.). Conserva ADVERBIOS si aportan CCT/CCL u otras marcas relevantes.
   - Conserva nombres propios y numerales como datos concretos.

3) MARCAS Y MAYÚSCULAS:
   - Usa MAYÚSCULAS para marcadores: PASADO, FUTURO, PRÓXIMO, NO, HABER, OJALÁ, DEBE, SI/NO y partículas interrogativas (QUÉ, QUIÉN, DÓNDE, CUÁNDO, CUÁL, POR QUÉ, CUÁNTO/OS/AS, CÓMO).
   - La glosa completa puede ir en mayúsculas para homogeneizar.

ORDEN BASE (PRIORIDAD)
A) Orden canónico LSE con foco en contexto:
   CCT (cuándo) + CCL (dónde) + SUJETO + RESTO DEL PREDICADO [CD (qué), CI (a quién)] + (si NEGATIVA) NO + VERBO (INFINITIVO) + (si PREGUNTA) PARTÍCULA INTERROGATIVA o SI/NO

B) Si NO hay CCT/CCL visibles y/o el contexto obligue a más generalidad (limitación de datos o ambigüedad), usar orden general:
   ADVERBIO + SUJETO + PREDICADO + VERBO (INFINITIVO)

   Nota: “ADVERBIO” aquí actúa como marcador general inicial (p. ej., PASADO, FUTURO/PRÓXIMO, HOY, MAÑANA, ALLÍ, AQUÍ…).

MARCAS TEMPORALES (CUÁNDO)
- Si la oración NO incluye explícitamente el tiempo y NO es claramente presente, añadir al inicio:
  • PASADO → para acciones pasadas o formas en pasado detectables.
  • FUTURO o PRÓXIMO → para acciones futuras (futuro morfológico, “ir a + inf.”, mañana, próximo…).
- Si es claramente presente y no hay CCT, NO añadir “PRESENTE”.

TIPOS DE ORACIÓN (REGLAS ESPECÍFICAS)
1) PASADO (sin CCT explícito):
   PASADO + (CCL si hay) + SUJETO + PREDICADO + (NO) + VERBO

2) FUTURO (sin CCT explícito):
   FUTURO o PRÓXIMO + (CCL si hay) + SUJETO + PREDICADO + (NO) + VERBO

3) PRESENTE (sin CCT):
   Mantener sin añadir “PRESENTE”. Aplicar el ORDEN BASE.

4) NEGATIVA:
   — Regla preferente (canónica LSE por claridad): colocar “NO” inmediatamente antes del VERBO:
      CCT + CCL + SUJETO + PREDICADO + NO + VERBO
   — Regla alternativa (si la consigna del proyecto lo exige): colocar la PARTÍCULA NEGATIVA (“NO/NI/TAMPOCO/NUNCA”) al FINAL, tras el VERBO:
      ADVERBIO/CCT + SUJETO + PREDICADO + VERBO + PARTÍCULA NEGATIVA
   Si hay ambigüedad, aplicar la regla preferente (NO antes del VERBO).

5) POSITIVA / AFIRMATIVA:
   Aplicar ORDEN BASE sin marcas adicionales.

6) IMPERATIVA:
   Aplicar ORDEN BASE y añadir “DEBE” AL FINAL.
   Ej.: ADVERBIO/CCT + SUJETO + PREDICADO + VERBO + DEBE

7) SUBJUNTIVA:
   Añadir OJALÁ al INICIO y luego ORDEN BASE.
   Ej.: OJALÁ + CCT/ADVERBIO + SUJETO + PREDICADO + VERBO

8) EXCLAMATIVA:
   Mantener ORDEN BASE y añadir “!” al FINAL (opcional si el entorno de glosa admite signos). No introducir léxico extra.
   Ej.: ADVERBIO/CCT + SUJETO + PREDICADO + VERBO + !

9) INTERROGATIVA:
   — Sí/No: añadir “SI/NO” al FINAL.
   — Interrogativa parcial: colocar la PARTÍCULA INTERROGATIVA al FINAL (QUÉ, QUIÉN, DÓNDE, CUÁNDO, CUÁL, POR QUÉ, CUÁNTO/OS/AS, CÓMO).
   Opcionalmente, añadir “?” final si el canal lo permite, sin reemplazar la partícula.

NORMALIZACIÓN DE COMPLEMENTOS
- CCL sin preposiciones: “en la casa de Ana” → “CASA ANA”; “al instituto” → “INSTITUTO”.
- Otros complementos sin preposiciones ni determinantes: “con mi amigo” → “AMIGO MÍO” (o solo “AMIGO” si la posesión no es crítica).
- “tener/existir/estar (lugar)” → HABER (p. ej., “tengo dinero” → “YO DINERO NO HABER” si negativa, o “YO DINERO HABER” si afirmativa).

FORMATO DE SALIDA
- Devuelve **SOLO glosas**, **una línea por cada oración** de la entrada, en el **mismo orden**.
- Sin explicaciones, sin comillas, sin numeración, sin bloques de código.
- (Opcional, si el llamador lo pide): en lugar de saltos de línea, separar glosas con “ || ” en una sola línea.

PROCEDIMIENTO PASO A PASO
1) Segmentar el bloque de entrada en oraciones (., !, ?, ;, saltos de línea). Ignorar fragmentos vacíos.
2) Para cada oración: detectar tiempo (CCT o inferencia PASADO/FUTURO/PRÓXIMO), lugar (CCL), sujeto, CD/CI, negación y modalidad (declarativa/negativa/interrogativa/exclamativa/imperativa/subjuntiva).
3) Si falta CCT y no es presente, anteponer PASADO o FUTURO/PRÓXIMO.
4) Reescribir CCL y demás complementos sin preposiciones/determinantes.
5) Aplicar mapeo de verbos: eliminar “ser/estar” copulativos; unificar tener/existir/estar-loc en HABER.
6) Conservar INFINITIVO para TODOS los verbos.
7) Ensamblar usando el ORDEN BASE (CCT + CCL + SUJETO + PREDICADO + (NO) + VERBO + [marcador final de modalidad]).
   — Si el contexto obliga a generalidad por escasez de CCT/CCL, usar la estructura general: ADVERBIO + SUJETO + PREDICADO + VERBO.
8) Modalidades: añadir los marcadores según el bloque “TIPOS DE ORACIÓN”.
9) Emitir la glosa final de esa oración en una línea. Repetir para cada oración.
10) Auto-chequeo final para cada línea:
   - ¿Orden correcto?
   - ¿Sin preposiciones ni determinantes?
   - ¿Verbos en infinitivo?
   - ¿“ser/estar” copulativos eliminados y “tener/existir/estar-loc” → HABER?
   - ¿Marca temporal añadida cuando faltaba y no era presente?
   - ¿Negación y/o interrogativa/exclamativa aplicadas en el lugar correcto?
   - ¿Salida en una sola línea por oración, en glosa?

EJEMPLOS CANÓNICOS
1) Entrada (2 oraciones):
   “Ayer estuvimos en el parque. Yo compré helados.”
   Salida:
   AYER PARQUE HABER
   AYER YO HELADOS COMPRAR

2) “Mi hermano está en casa de Ana.”
   → CASA ANA MI HERMANO HABER

3) “No tengo dinero.”
   (preferente) → YO DINERO NO HABER
   (alternativa proyecto) → YO DINERO HABER NO

4) “¿Vas a venir mañana a mi casa?”
   → MAÑANA CASA MÍA TÚ VENIR SI/NO

5) “¿Qué compraste en la tienda?”
   → PASADO TIENDA TÚ COMPRAR QUÉ

6) “Luis es médico.”
   → LUIS MÉDICO

7) Imperativa: “Cierra la puerta.”
   → PUERTA TÚ CERRAR DEBE

8) Subjuntiva: “Ojalá llueva mañana.”
   → OJALÁ MAÑANA LLOVER

9) Exclamativa: “¡Qué bonito es este lugar!”
   → LUGAR ESTE BONITO  !

10) Entrada (bloque de 3 oraciones mezcladas):
    “El año pasado fui a Madrid. Mi hermano no vino. ¿Vendrás tú el próximo mes?”
    Salida:
    AÑO PASADO MADRID YO IR
    PASADO MI HERMANO NO VENIR
    PRÓXIMO MES TÚ VENIR SI/NO

"""

def build_client(api_key: str) -> genai.Client:
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print("No se pudo inicializar el cliente Gemini.\nDetalle:", e, file=sys.stderr)
        sys.exit(1)

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

def make_config(model_id: str, max_tokens: int, thinking_budget: Optional[int]):
    """
    - Para 'gemini-2.5-pro' activamos thinking con presupuesto controlado (bajo).
    - Para el resto (p.ej., 'gemini-2.5-flash') forzamos thinking a 0.
    """
    kwargs = dict(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
    )
    if "gemini-2.5-pro" in model_id:
        tb = 8 if thinking_budget is None else max(1, thinking_budget)
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=tb)
        except Exception:
            pass
    else:
        # Importante: desactivar thinking en Flash/otros
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
    return types.GenerateContentConfig(**kwargs)

def call_gemini(sentence: str, model_id: str, api_key: str, max_tokens: int,
                thinking_budget: Optional[int], debug: bool=False) -> str:
    client = build_client(api_key)

    def _once(mod, mx, tb):
        resp = generate_once(client, mod, sentence, mx, tb, debug)
        out = extract_text(resp)
        fr = first_finish_reason(resp) or ""
        return out, fr

    # 1) Intento tal cual
    out, fr = _once(model_id, max_tokens, thinking_budget)
    if out:
        return out

    # 2) Si PRO + MAX_TOKENS: recorta thinking/salida
    if "gemini-2.5-pro" in model_id and "MAX_TOKENS" in (fr or ""):
        for tb in (6, 4):
            for mx in (96, 64, 32):
                out, fr = _once(model_id, mx, tb)
                if out:
                    return out

    # 3) Si sigue sin texto, caer a FLASH sin thinking
    if "MAX_TOKENS" in (fr or "") and model_id != "gemini-2.5-flash":
        out, fr = _once("gemini-2.5-flash", 96, None)
        if out:
            return out

    raise RuntimeError(f"Salida vacía. finish_reason={fr or 'N/A'}")

def generate_once(client, model_id: str, sentence: str, max_tokens: int, thinking_budget: Optional[int], debug: bool):
    config = make_config(model_id=model_id, max_tokens=max_tokens, thinking_budget=thinking_budget)
    resp = client.models.generate_content(
        model=model_id,
        contents=sentence,
        config=config,
    )
    if debug:
        try:
            print("\n[DEBUG] Respuesta cruda:\n", json.dumps(resp.to_dict(), ensure_ascii=False, indent=2))
        except Exception:
            print("\n[DEBUG] No se pudo serializar; repr():\n", repr(resp))
    return resp

def main():
    parser = argparse.ArgumentParser(description="Transformar oración a LSE escrita con Gemini.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ID del modelo (p.ej., gemini-2.5-flash o gemini-2.5-pro)")
    parser.add_argument("--api-key", help="API key de Gemini (si no usas variables de entorno)")
    parser.add_argument("--max-tokens", type=int, default=256, help="max_output_tokens (por defecto 256)")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Presupuesto de thinking (solo para gemini-2.5-pro). Ej.: 32")
    parser.add_argument("--debug", action="store_true", help="Muestra la respuesta cruda para depuración")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Falta la API key. Usa --api-key o define GEMINI_API_KEY / GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(2)

    sentence = input("Introduce una oración en español: ").strip()
    if not sentence:
        print("No se introdujo texto. Saliendo.")
        sys.exit(0)

    try:
        out = call_gemini(
            sentence=sentence,
            model_id=args.model,
            api_key=api_key,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            debug=args.debug,
        )
    except Exception as e:
        print(f"Error al llamar a la API: {e}", file=sys.stderr)
        sys.exit(3)

    print("\nGlosa (LSE escrita):")
    print(out)

if __name__ == "__main__":
    main()