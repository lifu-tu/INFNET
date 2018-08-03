# Multi-Label Classification with INFNET

You can get the data by executing the following command:

```
wget http://www.cics.umass.edu/~belanger/icml_mlc_data.tar.gz
```

Kalpesh Krishna has implemented a Tensorflow version of the this project!
Check it out at [here] (http://github.com/theshadow29/infnet-spen).

Here are some mistakes that you maybe make (Thanks for Kalpesh's feedback)   

- Not having a classification threshold and assuming it to be 0.5.

- Not pre-training b_i jointly with F(x) in the first stage, and misunderstanding the scheme used to load the initial A(x) parameters.

- Calculating sum(hinge loss objective) rather than mean(hinge loss objective).

- Calculating label averaged F-1 scores rather than example averaged F-1 scores


# Dependencies
Theano = 0.8, Python = 2.7, Numpy, lasagne
