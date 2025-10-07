# tests/test_rules.py
import re
from src.rule_engine import translate_rule_based as tr

def test_basics_exact():
    assert tr("Ayer compré un libro en la tienda.") == "AYER EN TIENDA YO LIBRO COMPRAR"
    assert tr("No quiero café.") == "YO CAFE NO QUERER"
    assert tr("¿Cómo te llamas?") == "TU A TI LLAMAR COMO"
    assert tr("Ojalá llueva mañana.") == "OJALA MAÑANA LLOVER"
    assert tr("A las 9 voy a la escuela.") == "A LAS 9 A ESCUELA YO IR"
    assert tr("Dámelo, por favor.") == "YO A MI ESO #FAVOR DAR"

def test_imperative_structure():
    out = tr("Ven aquí ahora mismo.")
    assert "TU" in out and "VENIR" in out and "AHORA" in out
