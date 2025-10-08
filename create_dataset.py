# create_dataset.py
import os, re, json, argparse
from datetime import datetime

# Importa tu generador RAE
from src.synth_engine_rae import generate_all

DEFAULT_ROOT = os.path.join("data", "rae_holdout", "blocks_data")
SHARD_PREFIX = "rae_shard"

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _next_shard_index(root: str, prefix: str = SHARD_PREFIX) -> int:
    _ensure_dir(root)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_n = 0
    for name in os.listdir(root):
        m = pat.match(name)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return max_n + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT,
                    help="Carpeta raíz donde se guardan los shards (por defecto data/rae_holdout/blocks_data).")
    ap.add_argument("--n_train", type=int, default=10000, help="Tamaño del shard (número de ejemplos de train).")
    ap.add_argument("--sampling", type=str, default="mixed",
                    choices=["mixed","uniform","frequent"],
                    help="Estrategia de muestreo del generador.")
    ap.add_argument("--base_dir", type=str, default=None,
                    help="Ruta a RAE/rla-es/.../es_ES si quieres forzar una ubicación; si no, el generador intenta autodetectar.")
    args = ap.parse_args()

    # Determina siguiente índice
    shard_id = _next_shard_index(args.root, SHARD_PREFIX)
    shard_name = f"{SHARD_PREFIX}{shard_id}"
    out_dir = os.path.join(args.root, shard_name)
    _ensure_dir(out_dir)

    print(f"[INFO] Creando {shard_name} en: {out_dir}")
    print(f"[INFO] n_train={args.n_train} sampling={args.sampling} base_dir={args.base_dir}")

    # Genera solo train (dev/test a 0)
    generate_all(
        n_train=args.n_train,
        n_dev=0,
        n_test=0,
        sampling=args.sampling,
        out_dir=out_dir,
        base_dir=args.base_dir
    )

    # Guarda metadatos
    meta = {
        "shard": shard_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_train": args.n_train,
        "sampling": args.sampling,
        "base_dir": args.base_dir
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Shard creado: {out_dir}")
    print(f"[TIP] Lanza el entrenamiento; el train cogerá automáticamente todos los shards en {args.root}.")

if __name__ == "__main__":
    main()
