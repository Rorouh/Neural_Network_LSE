#!/usr/bin/env bash
set -euo pipefail

python -m src.synth_engine_rae \
  --rae_lexicon_path "RAE/rla-es/data/rae_es_ES_lemma_to_pos_simple.json" \
  --rae_toponyms_root "RAE/rla-es/data/toponimos" \
  --topo_include_world \
  --out_dir "data/rae_holdout/blocks_data/rae_shard1" \
  --n_train 10000 --n_dev 800 --n_test 800 \
  --sampling product_time_place \
  --product_max_time 20 --product_max_place 40 \
  --product_max_subjects 8 --product_max_verbs 15 --product_max_objs 30
