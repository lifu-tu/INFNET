#!/bin/sh
python Feature_pretrain.py icml_mlc_data/data/bookmarks/bookmarks-train.torch   icml_mlc_data/data/bookmarks/bookmarks-dev.torch    150 150 10

python mlc.py --L2  0.001  --trainfile  icml_mlc_data/data/bookmarks/bookmarks-train.torch  --devfile  icml_mlc_data/data/bookmarks/bookmarks-dev.torch --testfile  icml_mlc_data/data/bookmarks/bookmarks-test.torch  --margin_type  0

python mlc.py --L2  0.001  --trainfile  icml_mlc_data/data/bookmarks/bookmarks-train.torch  --devfile  icml_mlc_data/data/bookmarks/bookmarks-dev.torch --testfile  icml_mlc_data/data/bookmarks/bookmarks-test.torch  --margin_type  1

python mlc.py --L2  0.001  --trainfile  icml_mlc_data/data/bookmarks/bookmarks-train.torch  --devfile  icml_mlc_data/data/bookmarks/bookmarks-dev.torch --testfile  icml_mlc_data/data/bookmarks/bookmarks-test.torch  --margin_type  2

python mlc.py --L2  0.001  --trainfile  icml_mlc_data/data/bookmarks/bookmarks-train.torch  --devfile  icml_mlc_data/data/bookmarks/bookmarks-dev.torch --testfile  icml_mlc_data/data/bookmarks/bookmarks-test.torch  --margin_type  3
