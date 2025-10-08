# src/train.py
import os, re, glob, json, argparse, inspect
import yaml
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, TrainingArguments, Trainer
)
from inspect import signature
from src.util import strip_accents_keep_hash_upper
from src.train_utils import WeightedTrainer


# -----------------------------
# Utilidades de carga / fusión
# -----------------------------
def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def build_hf_dataset_from_paths(train_path, dev_path, test_path):
    train = load_jsonl(train_path)
    dev   = load_jsonl(dev_path)
    test  = load_jsonl(test_path)
    ds_train = Dataset.from_dict({"src": [x["src"] for x in train], "tgt": [x["tgt"] for x in train]})
    ds_dev   = Dataset.from_dict({"src": [x["src"] for x in dev],   "tgt": [x["tgt"] for x in dev]})
    ds_test  = Dataset.from_dict({"src": [x["src"] for x in test],  "tgt": [x["tgt"] for x in test]})
    return DatasetDict(train=ds_train, validation=ds_dev, test=ds_test)

def _collect_shards(blocks_root: str, prefix: str = "rae_shard"):
    """Devuelve lista de carpetas shard ordenadas por índice final (rae_shardN)."""
    if not os.path.isdir(blocks_root):
        return []
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    items = []
    for name in os.listdir(blocks_root):
        m = pat.match(name)
        full = os.path.join(blocks_root, name)
        if m and os.path.isdir(full):
            items.append((int(m.group(1)), full))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]

def load_train_rows_from_blocks(blocks_root: str, dedup: bool = True):
    """Lee train.jsonl de cada shard y los fusiona. Dedup por 'src' si dedup=True."""
    shards = _collect_shards(blocks_root)
    if not shards:
        raise RuntimeError(f"No se encontraron shards en {blocks_root}")
    seen_src = set()
    fused = []
    total = 0
    for shard in shards:
        fpath = os.path.join(shard, "train.jsonl")
        if not os.path.exists(fpath):
            continue
        rows = load_jsonl(fpath)
        total += len(rows)
        if dedup:
            for r in rows:
                s = r.get("src")
                if s not in seen_src:
                    fused.append(r)
                    seen_src.add(s)
        else:
            fused.extend(rows)
    print(f"[INFO] Shards usados: {len(shards)} | ejemplos totales: {total} | tras dedup: {len(fused)}")
    return fused

def build_hf_dataset_from_blocks(train_blocks_dir, dev_path, test_path, dedup=True):
    train_rows = load_train_rows_from_blocks(train_blocks_dir, dedup=dedup)
    dev_rows   = load_jsonl(dev_path)
    test_rows  = load_jsonl(test_path)
    ds_train = Dataset.from_dict({"src": [x["src"] for x in train_rows], "tgt": [x["tgt"] for x in train_rows]})
    ds_dev   = Dataset.from_dict({"src": [x["src"] for x in dev_rows],   "tgt": [x["tgt"] for x in dev_rows]})
    ds_test  = Dataset.from_dict({"src": [x["src"] for x in test_rows],  "tgt": [x["tgt"] for x in test_rows]})
    return DatasetDict(train=ds_train, validation=ds_dev, test=ds_test)


# -----------------------------
# Helpers varios
# -----------------------------
def _flt(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)

def _int(x, default):
    try:
        return int(x)
    except Exception:
        return int(default)

def _bool(x, default):
    if isinstance(x, bool): return x
    if isinstance(x, str):  return x.strip().lower() in {"1","true","yes","y","on"}
    return bool(default)

def _filter_kwargs_for_cls(cls, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(cls)
        params = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return kwargs


# -----------------------------
# Main
# -----------------------------
def main(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("model_name", "t5-small")
    out_dir    = cfg.get("output_dir", "runs/exp_synth_t5")

    # Dos opciones: rutas directas o carpeta de shards
    train_path = cfg.get("train_path", "data/synthetic/train.jsonl")
    dev_path   = cfg.get("dev_path",   "data/synthetic/dev.jsonl")
    test_path  = cfg.get("test_path",  "data/synthetic/test.jsonl")
    train_blocks_dir = cfg.get("train_blocks_dir", None)
    dedup_train = bool(cfg.get("dedup_train", True))

    max_src_len = _int(cfg.get("max_source_length", 96), 96)
    max_tgt_len = _int(cfg.get("max_target_length", 64), 64)

    per_device_train_bs = _int(cfg.get("per_device_train_batch_size", cfg.get("batch_size", 16)), 16)
    per_device_eval_bs  = _int(cfg.get("per_device_eval_batch_size",  cfg.get("eval_batch_size", 16)), 16)
    grad_accum_steps    = _int(cfg.get("grad_accum_steps", 1), 1)

    num_epochs   = _flt(cfg.get("num_train_epochs", 8), 8)
    lr           = _flt(cfg.get("learning_rate", 5e-5), 5e-5)
    weight_decay = _flt(cfg.get("weight_decay", 0.01), 0.01)
    warmup_ratio = _flt(cfg.get("warmup_ratio", 0.05), 0.05)
    seed         = _int(cfg.get("seed", 42), 42)

    logging_steps = _int(cfg.get("logging_steps", 100), 100)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Añade tokens especiales (p.ej. {"additional_special_tokens": ["#"]})
    spec_path = "configs/special_tokens.json"
    if os.path.exists(spec_path):
        spec = json.load(open(spec_path, "r", encoding="utf-8"))
        tokenizer.add_special_tokens(spec)

    # Ahora sí, obtenemos id de '#'
    hash_id = tokenizer.convert_tokens_to_ids("#") if "#" in tokenizer.get_vocab() else None

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Redimensiona por si añadimos tokens especiales
    model.resize_token_embeddings(len(tokenizer))

    # Construye dataset (desde shards si se especifica)
    if train_blocks_dir:
        print(f"[INFO] Cargando shards desde: {train_blocks_dir} (dedup={dedup_train})")
        ds = build_hf_dataset_from_blocks(train_blocks_dir, dev_path, test_path, dedup=dedup_train)
    else:
        ds = build_hf_dataset_from_paths(train_path, dev_path, test_path)

    # Normalización opcional de targets (quita tildes, preserva '#', pone mayúsculas)
    def maybe_norm_targets(batch):
        if cfg.get("strip_accents_targets", False):
            batch["tgt"] = [strip_accents_keep_hash_upper(t) for t in batch["tgt"]]
        return batch
    ds = ds.map(maybe_norm_targets, batched=True)

    # Tokenización
    def preprocess(batch):
        model_inputs = tokenizer(
            batch["src"],
            max_length=max_src_len,
            truncation=True
        )
        labels = tokenizer(
            text_target=batch["tgt"],
            max_length=max_tgt_len,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds_tok = ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])

    # TrainingArguments (CPU-friendly)
    args_kwargs = {
        "output_dir": out_dir,
        "per_device_train_batch_size": per_device_train_bs,
        "per_device_eval_batch_size":  per_device_eval_bs,
        "learning_rate": float(lr),
        "num_train_epochs": num_epochs,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "seed": seed,

        "fp16": False,
        "bf16": False,
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "gradient_accumulation_steps": grad_accum_steps,
    }

    print(f"[DEBUG] learning_rate type={type(args_kwargs['learning_rate'])} value={args_kwargs['learning_rate']}")
    args = TrainingArguments(**_filter_kwargs_for_cls(TrainingArguments.__init__, args_kwargs))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )

    # Sanity step (después de ds_tok)
    try:
        import torch
        from torch.utils.data import DataLoader
        small = ds_tok["train"].select(range(min(8, len(ds_tok["train"]))))
        tmp_dl = DataLoader(small, batch_size=4, shuffle=False, collate_fn=data_collator)
        batch = next(iter(tmp_dl))
        model.train()
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss_val = float(out.loss.detach().cpu())
        out.loss.backward()
        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        model.zero_grad(set_to_none=True)
        print(f"[SANITY] one-step loss={loss_val:.4f} grad_norm={total_norm:.4f}")
    except Exception as e:
        print("[SANITY] skipped:", repr(e))

    # Trainer (ponderación de '#' opcional)
    trainer_cls = Trainer
    trainer_kwargs = {}
    if bool(cfg.get("upweight_hash_tokens", False)) and hash_id is not None:
        trainer_cls = WeightedTrainer
        trainer_kwargs.update({
            "hash_token_id": hash_id,
            "hash_weight": float(cfg.get("hash_weight", 2.0))
        })

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        **trainer_kwargs
    )

    print(f"[DEBUG] learning_rate type={type(args.learning_rate)} value={args.learning_rate}")
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Modelo y tokenizer guardados en: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()
    main(args.config)
