
# -*- coding: utf-8 -*-
"""
lse_validator.py
----------------
Validador/correktor de salidas LSE (glosa) tras el modelo. Garantiza:
- Orden aproximado: CCT + CCL + ... + (NO) + VERBO + (partícula interrogativa/ SI/NO)
- Eliminación de artículos/preposiciones
- Colocación de "NO" inmediatamente antes del último verbo
- Unificación "estar (lugar)/existir/tener/hay" -> HABER (heurística segura)
- Eliminación de cópulas SER/ESTAR (copulativas)
- Inserción opcional de marca temporal (PASADO/FUTURO) si se infiere del español de entrada
- Normalización de mayúsculas y espacios

Limitaciones:
- No hace análisis sintáctico profundo. Es heurístico y configurable por lexicón.
- La detección de CCT/CCL se basa en listas ampliables.
- No lematiza verbos arbitrarios: solo corrige formas frecuentes de SER/ESTAR/TENER/HABER.

Uso básico:
    from lse_validator import LSEValidator
    v = LSEValidator()
    corrected = v.validate(gloss_lse, source_es="¿Vas a venir mañana a mi casa?")
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

SPACES_RE = re.compile(r"\s+")

def norm(s: str) -> str:
    s = s.strip().replace("¿","?").replace("¡","!")
    s = re.sub(r"\s*[;|]\s*", " ", s)
    s = SPACES_RE.sub(" ", s)
    s = s.upper()
    return s

def split_tokens(s: str) -> List[str]:
    s = norm(s)
    if not s:
        return []
    return s.split(" ")

def join_tokens(toks: List[str]) -> str:
    return " ".join([t for t in toks if t])

DEFAULT_ARTICLES: Set[str] = {
    "EL","LA","LOS","LAS","UN","UNA","UNOS","UNAS","DEL","AL"
}
DEFAULT_PREPS: Set[str] = {
    "DE","A","EN","CON","POR","PARA","SIN","SOBRE","ENTRE","HACIA","DESDE",
    "SEGÚN","SEGUN","CONTRA","DURANTE","MEDIANTE","TRAS","HASTA"
}
DEFAULT_NEG: Set[str] = {"NO","NUNCA","NI","TAMPOCO"}

DEFAULT_QPARTS: Set[str] = {"QUÉ","QUE","QUIÉN","QUIEN","DÓNDE","DONDE","CUÁNDO","CUANDO",
                            "CUÁL","CUAL","QUÉ?","QUE?","CUÁNTO","CUANTO","CUÁNTOS","CUANTOS","CÓMO","COMO"}
POR_QUE_PAT = re.compile(r"\bPOR\s+QU[EÉ]\b")
YN_TOKEN = "SI/NO"

DEFAULT_CCT: Set[str] = {
    "PASADO","FUTURO","PRÓXIMO","PROXIMO","HOY","AYER","MAÑANA","MANANA",
    "AHORA","ANOCHE","ANTES","LUEGO","DESPUÉS","DESPUES",
    "AÑO","AÑO PASADO","AÑO PRÓXIMO","AÑO PROXIMO",
    "MES","MES PASADO","MES PRÓXIMO","MES PROXIMO",
    "SEMANA","SEMANA PASADA","SEMANA PRÓXIMA","SEMANA PROXIMA",
    "HACE","DENTRO"
}

DEFAULT_CCL: Set[str] = {
    "CASA","CASA ANA","INSTITUTO","UNIVERSIDAD","ESCUELA","PARQUE","MADRID","GRANADA",
    "TIENDA","HOSPITAL","BIBLIOTECA","OFICINA","TRABAJO","MERCADO","CALLE","PLAZA"
}

SER_FORMS = {"SOY","ERES","ES","SOMOS","SOIS","SON","FUI","FUE","ERAN","ERA","SERÁ","SERA","SER","SIENDO","SIDO"}
ESTAR_FORMS = {"ESTOY","ESTÁS","ESTAS","ESTÁ","ESTA","ESTAMOS","ESTÁIS","ESTAIS","ESTÁN","ESTAN",
               "ESTUVO","ESTABA","ESTARÁ","ESTARA","ESTAR","ESTADO","ESTANDO"}
TENER_FORMS = {"TENGO","TIENES","TIENE","TENEMOS","TENÉIS","TENEIS","TIENEN",
               "TUVE","TUVO","TENÍA","TENIA","TENDRÁ","TENDRA","TENER","TENIDO"}
HABER_FORMS = {"HAY","HABER","HUBO","HABÍA","HABIA","HABRÁ","HABRA"}

INF_SUFFIX = ("AR","ER","IR")

@dataclass
class LSEValidator:
    articles: Set[str] = field(default_factory=lambda: set(DEFAULT_ARTICLES))
    preps: Set[str] = field(default_factory=lambda: set(DEFAULT_PREPS))
    neg_tokens: Set[str] = field(default_factory=lambda: set(DEFAULT_NEG))
    cct_lex: Set[str] = field(default_factory=lambda: set(DEFAULT_CCT))
    ccl_lex: Set[str] = field(default_factory=lambda: set(DEFAULT_CCL))

    allow_insert_time: bool = True
    guess_time_from_source: bool = True

    def _rm_articles_preps(self, toks: List[str]) -> List[str]:
        return [t for t in toks if t not in self.articles and t not in self.preps]

    def _normalize_haber_cluster(self, toks: List[str]) -> List[str]:
        out = []
        for i,t in enumerate(toks):
            if t in TENER_FORMS or t in HABER_FORMS:
                out.append("HABER")
            elif t in ESTAR_FORMS:
                out.append("HABER")
            else:
                out.append(t)
        return out

    def _drop_copulas(self, toks: List[str]) -> List[str]:
        return [t for t in toks if t not in SER_FORMS and t not in ESTAR_FORMS]

    def _extract_cct(self, toks: List[str]) -> tuple[list[str], list[str]]:
        t = toks[:]
        cct = []
        bigrams = [("AÑO","PASADO"), ("AÑO","PRÓXIMO"), ("AÑO","PROXIMO"),
                   ("MES","PASADO"), ("MES","PRÓXIMO"), ("MES","PROXIMO"),
                   ("SEMANA","PASADA"), ("SEMANA","PRÓXIMA"), ("SEMANA","PROXIMA"),
                   ("DENTRO","DE"), ("HACE","DOS"), ("HACE","TRES")]
        used = [False]*len(t)
        for i in range(len(t)-1):
            if used[i] or used[i+1]: continue
            a,b = t[i], t[i+1]
            if (a,b) in bigrams:
                cct.extend([a,b])
                used[i]=used[i+1]=True
        for i,x in enumerate(t):
            if used[i]: continue
            if x in self.cct_lex:
                cct.append(x); used[i]=True
        rest = [t[i] for i in range(len(t)) if not used[i]]
        return cct, rest

    def _extract_ccl(self, toks: List[str]) -> tuple[list[str], list[str]]:
        t = toks[:]
        ccl = []
        used = [False]*len(t)
        for i in range(len(t)-1):
            a,b = t[i], t[i+1]
            if not used[i] and not used[i+1]:
                if (a == "CASA" and re.match(r"^[A-ZÑÁÉÍÓÚÜ]+$", b)) or \
                   (f"{a} {b}" in self.ccl_lex):
                    ccl.extend([a,b]); used[i]=used[i+1]=True
        for i,x in enumerate(t):
            if used[i]: continue
            if x in self.ccl_lex:
                ccl.append(x); used[i]=True
        rest = [t[i] for i in range(len(t)) if not used[i]]
        return ccl, rest

    def _move_negation_before_last_verb(self, toks: List[str]) -> List[str]:
        if not any(t in self.neg_tokens for t in toks):
            return toks
        last_v = None
        for i in range(len(toks)-1, -1, -1):
            t = toks[i]
            if t.endswith(INF_SUFFIX) or t in {"HABER","IR","VENIR","SER","ESTAR"}:
                last_v = i; break
        if last_v is None:
            toks = [t for t in toks if t not in self.neg_tokens] + ["NO"]
            return toks
        toks_wo_no = [t for t in toks if t not in self.neg_tokens]
        lv = None
        for i in range(len(toks_wo_no)-1, -1, -1):
            if toks_wo_no[i].endswith(INF_SUFFIX) or toks_wo_no[i] in {"HABER","IR","VENIR","SER","ESTAR"}:
                lv = i; break
        if lv is None:
            return toks_wo_no + ["NO"]
        toks_wo_no.insert(lv, "NO")
        return toks_wo_no

    def _extract_question_particle(self, toks: List[str]) -> tuple[Optional[str], List[str]]:
        s = " ".join(toks)
        if POR_QUE_PAT.search(s):
            new = []
            skip_next = False
            found = False
            for i,t in enumerate(toks):
                if skip_next:
                    skip_next = False
                    continue
                if t == "POR" and i+1 < len(toks) and toks[i+1] in {"QUÉ","QUE"}:
                    found = True
                    skip_next = True
                    continue
                new.append(t)
            return ("POR QUÉ" if found else None, new)
        if YN_TOKEN in toks:
            new = [t for t in toks if t != YN_TOKEN]
            return YN_TOKEN, new
        wh = None
        new = []
        for t in toks:
            if t in DEFAULT_QPARTS and wh is None:
                wh = "QUÉ" if t in {"QUE","QUÉ","QUE?"} else t
            else:
                new.append(t)
        return wh, new

    def _maybe_insert_time(self, toks: List[str], source_es: Optional[str]) -> List[str]:
        if not self.allow_insert_time or not self.guess_time_from_source or not source_es:
            return toks
        if toks and (toks[0] in self.cct_lex or toks[0] in {"PASADO","FUTURO","PRÓXIMO","PROXIMO"}):
            return toks
        s = source_es.lower()
        fut = any(w in s for w in ["mañana","próximo","proximo","el año que viene","la semana que viene"]) \
              or re.search(r"\b(voy|vas|va|vamos|vais|van)\s+a\s+\w+(?:ar|er|ir)\b", s) is not None \
              or re.search(r"\b\w+rá\b", s) is not None
        pas = any(w in s for w in ["ayer","anoche","hace ","pasado"]) \
              or re.search(r"\b(?:fui|fue|estuvo|estaba|tuvo|compró|compraron|llegó|llegaron)\b", s) is not None
        if fut: return ["FUTURO"] + toks
        if pas: return ["PASADO"] + toks
        return toks

    def validate(self, gloss_lse: str, source_es: Optional[str]=None) -> str:
        toks = split_tokens(gloss_lse)
        toks = self._normalize_haber_cluster(toks)
        toks = self._rm_articles_preps(toks)
        cct, rest = self._extract_cct(toks)
        ccl, rest = self._extract_ccl(rest)
        rest = self._drop_copulas(rest)
        qpart, rest = self._extract_question_particle(rest)
        rest = self._move_negation_before_last_verb(rest)
        toks = cct + ccl + rest
        toks = self._maybe_insert_time(toks, source_es)
        if qpart:
            toks.append(qpart)
        s = join_tokens(toks)
        s = re.sub(r"\s+([!?])", r"\1", s)
        return s

# Pequeño test si se ejecuta como script
if __name__ == "__main__":
    v = LSEValidator()
    tests = [
        ("MAÑANA CASA MÍA TÚ VAS A VENIR ?", "¿Vas a venir mañana a mi casa?"),
        ("YO DINERO HABER NO", "No tengo dinero."),
        ("LUIS ES MÉDICO", "Luis es médico."),
        ("CASA ANA MI HERMANO ESTA", "Mi hermano está en casa de Ana."),
        ("AYER YO HELADOS COMPRÓ", "Ayer yo compré helados."),
        ("TIENDA TU COMPRAR QUE", "¿Qué compraste en la tienda?"),
        ("YO EN MADRID ESTOY", "Estoy en Madrid."),
    ]
    for out, src in tests:
        print("SRC:", src)
        print("IN :", out)
        print("OUT:", v.validate(out, source_es=src))
        print("---")
