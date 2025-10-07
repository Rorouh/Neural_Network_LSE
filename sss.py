import json, random
random.seed(0)
def head_jsonl(inp, outp, n):
    rows = [json.loads(x) for x in open(inp, encoding="utf-8")]
    random.shuffle(rows)
    rows = rows[:n]
    with open(outp, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
head_jsonl("data/synthetic/train.jsonl","data/synthetic/mini.jsonl",128)