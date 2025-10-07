# LSE Translator (Español → LSE Escrita por Glosas)

> Traductor de **texto en español** a **LSE escrita (glosas en alfabeto latino)** mediante un **enfoque híbrido**: reglas lingüísticas + modelo neuronal (Transformer).

---

## ✨ Características

* **Salida en glosas** (alfabeto latino) siguiendo **gramática LSE** (tiempo/lugar al inicio, SOV, adjetivo pospuesto, sin artículos ni “ser/estar”, etc.).
* **Motor por reglas** configurable vía **YAML**.
* **Generador de datos sintéticos** para pre-entrenar el modelo.
* **Entrenamiento NMT** con **T5/mT5** (HuggingFace).
* **Validador gramatical** (checks de reglas duras: verbo final, sin artículos, WH al final…).
* **Postprocesado** para normalización y corrección ligera.

---

## 📁 Estructura del proyecto

```
lse-translator/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  ├─ training.yaml            # Hiperparámetros de entrenamiento
│  └─ special_tokens.json      # Tokens especiales ([WH], [Q], etc.)
├─ data/
│  ├─ lexicon/
│  │  └─ es_to_lse.csv         # Diccionario ES→LSE (glosas)
│  ├─ synthetic/
│  │  ├─ train.jsonl           # Datos sintéticos (src,tgt)
│  │  ├─ dev.jsonl
│  │  └─ test.jsonl
│  └─ gold/
│     ├─ dev.jsonl             # (Opcional) Pares revisados a mano
│     └─ test.jsonl
├─ rules/
│  ├─ reorder.yml              # Orden oracional, posiciones, negación, WH…
│  ├─ drop.yml                 # Qué elementos se eliminan (artículos, copula…)
│  ├─ lexical.yml              # Listas léxicas (tiempo, WH…) y fallback
│  └─ questions.yml            # Comportamiento de interrogativas
├─ src/
│  ├─ __init__.py
│  ├─ rule_engine.py           # Motor de reglas (MVP)
│  ├─ features.py              # Detección de roles (SUJ/OBJ/TIEMPO/LUGAR…)
│  ├─ synth_generator.py       # Generador de pares sintéticos
│  ├─ train.py                 # Entrenamiento (HF Transformers)
│  ├─ decode.py                # Inferencia (traducción en lote)
│  ├─ postprocess.py           # Limpiezas/forzados finales
│  ├─ validate.py              # Checks de gramática LSE
│  └─ util.py                  # IO YAML/CSV utilidades
└─ tests/
   └─ test_cases.jsonl         # Casos unitarios lingüísticos
```

---

## 🔧 Requisitos

* **Python** 3.10+
* **pip** / **venv**
* (Opcional) **GPU** con PyTorch para entrenar más rápido

Instalación:

```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy download es_core_news_md
```

---

## 🚀 Uso rápido (MVP)

1. **Generar datos sintéticos**

```bash
python -m src.synth_generator
```

2. **Entrenar el modelo**

```bash
python -m src.train
# o con config específica:
# python -m src.train --cfg configs/training.yaml
```

3. **Traducir (inferencia)**

```python
from src.decode import translate_list
print(translate_list(["Ayer compré un libro en la tienda."]))
# → ["AYER TIENDA YO LIBRO COMPRAR"]
```

---

## 🧠 Conceptos clave

* **Glosa**: representación textual de un signo (en mayúsculas), p.ej., `YO`, `COMPRAR`, `LIBRO`.
* **Reglas**: definen cómo transformar el español a la estructura LSE (reordenar, eliminar artículos, posponer adjetivos…).
* **Sintético**: datos generados automáticamente aplicando el motor de reglas a oraciones en español.
* **Gold**: datos revisados manualmente; se usan para *fine-tuning* y evaluación de calidad.

---

## ⚙️ Configuración de reglas

### `rules/reorder.yml` (orden y posiciones)

```yaml
order:
  - TIME      # AYER/HOY/MAÑANA/"A LAS 9"
  - PLACE     # LUGAR
  - SUBJECT   # SUJETO
  - OBJECT    # OBJETO
  - MODS      # modificadores que quieras usar
  - VERB      # verbo al final

adj:
  postpose: true

negation:
  position: "before_verb"

time:
  fronted: true

place:
  fronted: true

questions:
  wh_final: true
  yesno_mark: "[Q]"
```

### `rules/drop.yml` (qué eliminar)

```yaml
drop:
  articles: true       # el/la/un/una/los/las
  copula: true         # ser/estar en copulativas
  auxiliaries: true    # haber/ser auxiliares
  determiners_exceptions: ["este","ese","aquel"]  # si quieres conservar
```

### `rules/lexical.yml` (listas léxicas y fallback)

```yaml
time_lex: [AYER, HOY, MAÑANA, "A LAS", "A LAS *"]
place_preps: [en, a, hacia, desde]
neg_lex: [no, nunca]

wh_lex:
  QUE:   ["qué","que"]
  QUIEN: ["quién","quien"]
  COMO:  ["cómo","como"]
  DONDE: ["dónde","donde"]
  CUANDO:["cuándo","cuando"]
  PORQUE:["por qué","porque"]

fallback:
  fingerspell_prefix: "#"
```

### `rules/questions.yml` (interrogativas)

```yaml
wh:
  move_to_end: true
  keep_punctuation: false

yesno:
  add_marker: true
  marker: "[Q]"
```

> **Dónde añadir TUS reglas**: edita estos YAML o crea nuevos (p.ej., `transform.yml`) y conéctalos en `src/rule_engine.py`.

---

## 📚 Diccionario ES→LSE

`data/lexicon/es_to_lse.csv` define mapeos de palabras españolas a glosas LSE:

```csv
es_lemma,es_form,upos,features,lse_gloss,notes
yo,yo,PRON,Person=1|Number=Sing,YO,
tú,tú,PRON,Person=2|Number=Sing,TU,
ser,ser,VERB,,(DROP_COPULA),copulativa
estar,estar,VERB,,(DROP_COPULA),copulativa
haber,haber,VERB,Aux=Yes,(DROP_AUX),auxiliar
ir,ir,VERB,,IR,
comprar,compré,VERB,Tense=Past,COMPRAR,
libro,libro,NOUN,,LIBRO,
tienda,tienda,NOUN,,TIENDA,
ayer,ayer,ADV,Temp=Past,AYER,tiempo
no,no,PART,Polarity=Neg,NO,negación
qué,qué,PRON,Int=Yes,QUE,[WH]
cómo,cómo,ADV,Int=Yes,COMO,[WH]
```

* **`es_lemma/es_form`** para cubrir lemas y formas irregulares.
* **`lse_gloss`** SIEMPRE en **MAYÚSCULAS**.
* Si falta mapeo, el motor usará **dactilología** `#PALABRA`.

---

## 🧪 Datos (formato)

Archivos **JSONL** (una línea por ejemplo):

```json
{"src": "Ayer compré un libro en la tienda.", "tgt": "AYER TIENDA YO LIBRO COMPRAR"}
{"src": "Mi casa es grande.", "tgt": "CASA MI GRANDE"}
{"src": "¿Cómo te llamas?", "tgt": "TU NOMBRE QUE [WH]"}
```

* `data/synthetic/*.jsonl`: pares generados por reglas.
* `data/gold/*.jsonl`: pares revisados (opcional para *fine-tuning*).

---

## 🏭 Generación de sintético

Configura plantillas y bancos de palabras dentro de `src/synth_generator.py`. Ejecuta:

```bash
python -m src.synth_generator
```

Genera:

* `data/synthetic/train.jsonl`
* `data/synthetic/dev.jsonl`
* `data/synthetic/test.jsonl`

---

## 🏋️ Entrenamiento

Parámetros en `configs/training.yaml`:

```yaml
model_name: "google/mt5-small"   # o "t5-small"
output_dir: "outputs/mt5-lse"
max_source_length: 128
max_target_length: 128
per_device_train_batch_size: 32
per_device_eval_batch_size: 32
learning_rate: 3e-4
num_train_epochs: 5
warmup_ratio: 0.06
weight_decay: 0.01
eval_steps: 200
save_steps: 200
logging_steps: 50
fp16: true
seed: 42
```

Ejecuta:

```bash
python -m src.train
```

> **Consejo**: Pre-entrena con **sintético** y, si tienes pares revisados, haz *fine-tune* con `data/gold/`.

---

## 🔮 Inferencia

```python
from src.decode import translate_list
outs = translate_list([
  "Ayer compré un libro en la tienda.",
  "¿Dónde vives?"
])
print(outs)
# ['AYER TIENDA YO LIBRO COMPRAR', 'TU VIVIR DONDE [WH]']
```

---

## 🧹 Post-proceso

`src/postprocess.py` limpia mayúsculas/espacios y puede **forzar reglas** (p.ej., mover `NO` antes del verbo, quitar artículos rezagados).

---

## ✅ Validación gramatical

`src/validate.py` aplica checks (sin artículos, verbo final, WH al final…):

```python
from src.validate import score_gloss
print(score_gloss("AYER TIENDA YO LIBRO COMPRAR"))
# {'no_articles': True, 'verb_final': True}
```

Úsalo para:

* Filtrar sintético defectuoso.
* Medir cumplimiento en dev/test.
* Guiar mejoras del motor/reglas.

---

## 🧪 Tests lingüísticos

`tests/test_cases.jsonl` (expected exact):

```json
{"src":"Ayer compré un libro en la tienda.","expect":"AYER TIENDA YO LIBRO COMPRAR"}
{"src":"Mi casa es grande.","expect":"CASA MI GRANDE"}
{"src":"¿Cómo te llamas?","expect":"TU NOMBRE QUE [WH]"}
{"src":"No quiero café.","expect":"YO CAFE NO QUERER"}
{"src":"El perro rojo duerme en casa.","expect":"CASA PERRO ROJO DORMIR"}
```

(Ejecuta con tu framework de tests favorito o crea un script simple que compare `expect` con la salida del motor neuronal y/o de reglas.)

---

## 🧩 Personalización y ampliaciones

* **Orden** (PLACE al final, etc.): cambia `rules/reorder.yml`.
* **Eliminar/Conservar** (artículos, copula): `rules/drop.yml`.
* **WH/tiempo/lugar**: `rules/lexical.yml` y `rules/questions.yml`.
* **Léxico**: añade glosas en `data/lexicon/es_to_lse.csv`.
* **Transformaciones complejas** (pasivas→activas, relativas→dos oraciones, topicalización): añade funciones en `src/rule_engine.py` y habilítalas con un nuevo YAML (`rules/transform.yml`).

---

## 🛣️ Roadmap sugerido

1. Completar diccionario base (100–200 entradas frecuentes).
2. Ajustar orden y reglas mínimas (SOV, adjetivo pospuesto, tiempo/lugar frontal, negación).
3. Generar 2–5k sintéticos variados.
4. Entrenar mT5-small.
5. Evaluar y corregir (post-proceso + nuevas reglas).
6. Añadir *gold* (pares revisados) para *fine-tune*.
7. Escalar léxico, dominios y transformaciones (pasivas, relativas, clasificadores si procede).

---

## 🆘 Problemas comunes

* **spaCy no encuentra el modelo**:
  `python -m spacy download es_core_news_md`
* **Memoria GPU insuficiente**:
  reduce `batch_size` o usa `t5-small`.
* **Salidas con artículos/VO**:
  ajusta reglas + añade **post-proceso** de corrección dura.
* **OOV frecuentes**:
  enriquece `es_to_lse.csv` o permite `#PALABRA` (dactilología).

---

## 📄 Licencia

Añade aquí la licencia que prefieras (por ejemplo, MIT).

---

## ✍️ Autoría y contacto

Proyecto de traducción **ES → LSE (glosas)** con enfoque híbrido (reglas + NMT).
Para dudas/mejoras, abre un issue o comenta en el repo.
