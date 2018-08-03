# Sequence Labeling with INFNET

You can get the pretained twitter embedding by excuting the command:
```
wget http://ttic.uchicago.edu/~lifu/TE_TweeboParser/wordvects.tw100w5-m40-it2
```
And you can also get the processed data used in my experiments for sequence labeling here
```
wget http://ttic.uchicago.edu/~lifu/Tweet_Pos.tar.gz
```

# Reference for the pretrained embedding

```
@inproceedings{tu-17-long,
  title={Learning to Embed Words in Context for Syntactic Tasks},
  author={Lifu Tu and Kevin Gimpel and Karen Livescu},
  booktitle={Proceedings of the 2nd Workshop on Representation Learning for NLP},
  year={2017},
  publisher = {Association for Computational Linguistics}
}
```

# More Detail about the three folder:

- CRF: code for BLSTM_CRF for sequence labeling task.
- adv_infnet : code for large-margin training criteria for joint training of the structured energy function and inference network
- CRF_infnet : code for the training of inference network when given the parameters of energy function





