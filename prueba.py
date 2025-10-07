# prueba.py
from src.rule_engine import translate_rule_based as tr

SAMPLES = [
    "Ayer compré un libro en la tienda.",
    "No quiero café.",
    "¿Cómo te llamas?",
    "Ven aquí ahora mismo.",
    "Ojalá llueva mañana.",
    "A las 9 voy a la escuela.",
    "Dámelo, por favor.",
]

for s in SAMPLES:
    print(f"{s} → {tr(s)}")
