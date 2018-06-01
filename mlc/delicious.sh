#!/bin/sh
python Feature_pretrain.py icml_mlc_data/data/delicious/delicious-train.torch   icml_mlc_data/data/delicious/delicious-test.torch    250 250 10

python mlc.py --L2  0.001 --hidden1   250  --hidden2 250 --hidden1_a  250  --hidden2_a 250  --C1  10  --trainfile  icml_mlc_data/data/delicious/delicious-train.torch --devfile  icml_mlc_data/data/delicious/delicious-test.torch  --testfile  icml_mlc_data/data/delicious/delicious-test.torch  --FeatureNet  mlc_250_10.pickle --infNet  mlc_250_10.pickle  --margin_type  0

python mlc.py --L2  0.001 --hidden1   250  --hidden2 250 --hidden1_a  250  --hidden2_a 250  --C1  10  --trainfile  icml_mlc_data/data/delicious/delicious-train.torch --devfile  icml_mlc_data/data/delicious/delicious-test.torch  --testfile  icml_mlc_data/data/delicious/delicious-test.torch  --FeatureNet  mlc_250_10.pickle --infNet  mlc_250_10.pickle  --margin_type  1

python mlc.py --L2  0.001 --hidden1   250  --hidden2 250 --hidden1_a  250  --hidden2_a 250  --C1  10  --trainfile  icml_mlc_data/data/delicious/delicious-train.torch --devfile  icml_mlc_data/data/delicious/delicious-test.torch  --testfile  icml_mlc_data/data/delicious/delicious-test.torch  --FeatureNet  mlc_250_10.pickle --infNet  mlc_250_10.pickle  --margin_type  2

python mlc.py --L2  0.001 --hidden1   250  --hidden2 250 --hidden1_a  250  --hidden2_a 250  --C1  10  --trainfile  icml_mlc_data/data/delicious/delicious-train.torch --devfile  icml_mlc_data/data/delicious/delicious-test.torch  --testfile  icml_mlc_data/data/delicious/delicious-test.torch  --FeatureNet  mlc_250_10.pickle --infNet  mlc_250_10.pickle  --margin_type  3
