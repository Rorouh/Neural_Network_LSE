# auto_runner_to_csv.py
import os, sys, time, subprocess, re, math
from pathlib import Path

# === CONFIGURACIÓN ===
PROJECT_DIR   = r"C:\Users\super\Desktop\Informatica\LSE-translator\LLM"
INPUT_FILE    = "entrada1.txt"
OUTPUT_FILE   = "dataset.csv"
SCRIPT_NAME   = "lse_gemini_cli.py"   # <-- tu script nuevo

MODEL         = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TOKENS    = 1024

# Tamaño del primer lote (líneas por request). Se ajusta solo si hay MAX_TOKENS.
INITIAL_LIMIT = 20
MIN_LIMIT     = 5

# Peticiones por minuto por clave (free tier suele ~1/min)
QPM           = 6

# Espera adicional (colchón) además del pacing por QPM
EXTRA_SLEEP_S = 5

# Limite de iteraciones por ejecución (seguridad). 0 = ilimitado hasta vaciar TXT o error duro.
MAX_CYCLES    = 0

LOG_FILE      = "runner.log"

# Lee API keys: usa GEMINI_API_KEYS="k1,k2" o GEMINI_API_KEY="k1"
ENV           = os.environ.copy()
KEYS_ENV      = ENV.get("GEMINI_API_KEYS") or ENV.get("GEMINI_API_KEY") or ""
API_KEYS      = [k.strip() for k in KEYS_ENV.split(",") if k.strip()]

if not API_KEYS:
    print("ERROR: Define GEMINI_API_KEY o GEMINI_API_KEYS en el entorno.", file=sys.stderr)
    sys.exit(2)

# === Utilidades ===
def remaining_lines(path_txt: str) -> int:
    p = Path(PROJECT_DIR) / path_txt
    if not p.exists():
        return 0
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def pace_sleep(qpm: int, started_at: float):
    # Espacia a 1/QPM por minuto; restamos lo ya invertido y añadimos un pequeño colchón
    per_req = 60.0 / max(1, qpm)
    elapsed = time.time() - started_at
    to_sleep = max(0.0, per_req - elapsed) + EXTRA_SLEEP_S
    time.sleep(to_sleep)

def parse_retry_seconds(stderr_text: str) -> float | None:
    # Busca "Please retry in 38.12s." o 'retryDelay': '38s'
    m = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", stderr_text, flags=re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9]+)s", stderr_text, flags=re.I)
    if m:
        return float(m.group(1))
    return None

def run_once(limit: int, api_key: str) -> tuple[int, str, str]:
    """
    Ejecuta una pasada: devuelve (returncode, stdout, stderr).
    """
    cmd = [
        sys.executable, "-u", SCRIPT_NAME,
        "--input", INPUT_FILE,
        "--out",   OUTPUT_FILE,
        "--limit", str(limit),
        "--model", MODEL,
        "--max-tokens", str(MAX_TOKENS),
        "--api-key", api_key,
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        env=ENV,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr

def log_append(text: str):
    p = Path(PROJECT_DIR) / LOG_FILE
    with p.open("a", encoding="utf-8") as f:
        f.write(text + ("\n" if not text.endswith("\n") else ""))

def main():
    Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    cycles = 0
    key_idx = 0
    limit   = INITIAL_LIMIT

    while True:
        if MAX_CYCLES and cycles >= MAX_CYCLES:
            print(f"Fin por MAX_CYCLES={MAX_CYCLES}.")
            break

        rem = remaining_lines(INPUT_FILE)
        if rem == 0:
            print("No quedan líneas en el TXT. Salgo.")
            break

        # Ajusta limit si hay menos líneas que el lote deseado
        cur_limit = min(limit, rem)
        api_key = API_KEYS[key_idx]

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n[{ts}] --- RUN (limit={cur_limit}, key#{key_idx+1}/{len(API_KEYS)}) ---"
        print(header)
        log_append(header)

        started = time.time()
        rc, out, err = run_once(cur_limit, api_key)

        # Loguea todo
        if out.strip():
            log_append("[STDOUT]\n" + out.strip())
        if err.strip():
            log_append("[STDERR]\n" + err.strip())

        if rc == 0:
            print(f"OK (limit={cur_limit}).")
            # Si todo salió bien y usamos limit reducido por errores previos,
            # puedes subir de nuevo progresivamente (opcional):
            if limit < INITIAL_LIMIT:
                limit = min(INITIAL_LIMIT, max(limit + max(1, limit//2), MIN_LIMIT))
            cycles += 1
            # Ritmo por QPM
            pace_sleep(QPM, started)
            continue

        # --- Manejo de errores comunes ---
        all_text = (out or "") + "\n" + (err or "")
        # 429 / cuota: rota de clave o espera
        if "RESOURCE_EXHAUSTED" in all_text or "429" in all_text:
            retry_s = parse_retry_seconds(all_text)
            if len(API_KEYS) > 1:
                # rota de clave
                key_idx = (key_idx + 1) % len(API_KEYS)
                msg = f"Cuota agotada para la clave actual. Cambio a key #{key_idx+1}."
                print(msg); log_append(msg)
                wait = retry_s if retry_s is not None else (60.0 / max(1, QPM)) + EXTRA_SLEEP_S
                time.sleep(wait)
                continue
            else:
                # sin más claves: esperar y reintentar
                wait = retry_s if retry_s is not None else (60.0 / max(1, QPM)) + 30
                msg = f"Cuota agotada y no hay más claves. Espero {math.ceil(wait)}s y reintento."
                print(msg); log_append(msg)
                time.sleep(wait)
                continue

        # MAX_TOKENS o mismatch de conteo: reduce lote y reintenta
        if "MAX_TOKENS" in all_text or "Conteo de glosas" in all_text or "!= líneas entrada" in all_text:
            new_limit = max(MIN_LIMIT, limit // 2) if limit > MIN_LIMIT else max(MIN_LIMIT, cur_limit // 2)
            if new_limit < limit:
                limit = new_limit
            elif cur_limit > MIN_LIMIT:
                limit = max(MIN_LIMIT, cur_limit // 2)
            else:
                # no podemos reducir más; abortamos la ejecución (evitamos ciclo infinito)
                msg = "No se puede reducir más el lote. Revisa MAX_TOKENS o acorta las oraciones."
                print(msg); log_append(msg)
                break
            msg = f"Reduciendo lote a limit={limit} por tokens/conteo."
            print(msg); log_append(msg)
            pace_sleep(QPM, started)
            continue

        # Otros errores: imprime y detén (para no borrar nada por accidente)
        print("Error no recuperable. Revisa runner.log y la salida del script.")
        break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")
