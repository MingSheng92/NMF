# Projected Gradient for Non-Negative Matrix Factorization

## Introduction 

Non-Negative Matrix Factorization (NMF) is a group of unsupervised machine learning techniques which factors a multivariate dataset as the approximate product of two lower dimension matrices. NMF is widely applied in various fields such as astronomy, computer and recommender systems. In this project we are interested in implementing NMF algorithm and analyze the robustness of the algorithm when the dataset is contaminated by different types of noise.

Our objective is to ﬁnd two non-negative matrix factors _W_ and _H_ which multiply to give as close as possible a representation of our original image _V_.

![f1]  

Where _W_ contains a basis optimized for the linear approximation of data in _V_, weighted by the components in _H_, i.e. Each column of _W_ is a basis element. If _V_ is our proverbial cake, then _W_ and _H_ are our list of core ingredients and their respective amounts. 

More in depth information on NMF can be found at [Wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization).

## Projected Gradient methods for NMF


## Method
### Datasets



WIP.... 

Additional information: \
https://angms.science/doc/NMF/nmf_pgd.pdf \
https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf



[f1]: http://chart.apis.google.com/chart?cht=tx&chl=V\approx{WH}
