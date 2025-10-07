import json
from datasets import Dataset
from transformers import AutoTokenizer

MODEL = "t5-small"  # o el que uses
max_src_len = 96
max_tgt_len = 64

def load_jsonl(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows

def main():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    data = load_jsonl("data/synthetic/train.jsonl")[:8]  # miro 8 ejemplos
    ds = Dataset.from_dict({"src":[x["src"] for x in data], "tgt":[x["tgt"] for x in data]})

    def preprocess(batch):
        enc = tok(batch["src"], max_length=max_src_len, truncation=True)
        tgt = tok(text_target=batch["tgt"], max_length=max_tgt_len, truncation=True)
        enc["labels"] = tgt["input_ids"]
        return enc

    ds_tok = ds.map(preprocess, batched=True)
    # imprime un ejemplo de labels (ids y tokens) para confirmar
    ex = ds_tok[0]
    print("SRC:", data[0]["src"])
    print("TGT:", data[0]["tgt"])
    print("labels_ids:", ex["labels"][:30])
    print("labels_tokens:", tok.convert_ids_to_tokens(ex["labels"][:30]))

    # porcentaje de tokens válidos en labels (no -100) si luego colapsas con collator
    # aquí aún no hay -100 (eso lo pone el collator al padear), pero al menos ves que NO son listas vacías.
    print("len(labels):", len(ex["labels"]))

if __name__ == "__main__":
    main()
