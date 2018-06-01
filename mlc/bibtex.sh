#!/bin/sh
python Feature_pretrain.py icml_mlc_data/data/bibtex/bibtex-train.torch icml_mlc_data/data/bibtex/bibtex-test.torch 150 150 10

python mlc.py --margin_type 0

python mlc.py --margin_type 1 

python mlc.py --margin_type 2

python mlc.py --margin_type 3

