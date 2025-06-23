#!/bin/bash
python finetune_vivit.py --config D0
python finetune_vivit.py --config D1
python finetune_vivit.py --config D2

python run_k_folds.py --config P0
python run_k_folds.py --config P1
python run_k_folds.py --config P2
python run_k_folds.py --config D0P0
python run_k_folds.py --config D1P1
python run_k_folds.py --config D2P2
python run_k_folds.py --config D1P0
python run_k_folds.py --config D2P0
python run_k_folds.py --config D2P1