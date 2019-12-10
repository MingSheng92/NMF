# Projected Gradient for Non-Negative Matrix Factorization

## Introduction 

Non-Negative Matrix Factorization (NMF) is a group of unsupervised machine learning techniques which factors a multivariate dataset as the approximate product of two lower dimension matrices. NMF is widely applied in various fields such as astronomy, computer and recommender systems. In this project we are interested in implementing NMF algorithm and analyze the robustness of the algorithm when the dataset is contaminated by different types of noise.

Our objective is to ﬁnd two non-negative matrix factors _W_ and _H_ which multiply to give as close as possible a representation of our original image _V_.

<a href="https://www.codecogs.com/eqnedit.php?latex=V\approx{WH}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V\approx{WH}" title="V\approx{WH}" /></a> <br />

Where _W_ contains a basis optimized for the linear approximation of data in _V_, weighted by the components in _H_, i.e. Each column of _W_ is a basis element. If _V_ is our proverbial cake, then _W_ and _H_ are our list of core ingredients and their respective amounts. 

More in depth information on NMF can be found at [Wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization).

## Projected Gradient methods for NMF

By altering the existing alternating non-negative least squares algorithm where it will alternatively fixes one matrix and improves the other.

<a href="https://www.codecogs.com/eqnedit.php?latex=W^{k&plus;1}&space;=&space;arg\min_{&space;W&space;\geqslant&space;0&space;}&space;f(W,&space;H^k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{k&plus;1}&space;=&space;arg\min_{&space;W&space;\geqslant&space;0&space;}&space;f(W,&space;H^k)" title="W^{k+1} = arg\min_{ W \geqslant 0 } f(W, H^k)" /></a> <br/>
<a href="https://www.codecogs.com/eqnedit.php?latex=H^{k&plus;1}&space;=&space;arg\min_{&space;H&space;\geqslant&space;0&space;}&space;f(W^{k&plus;1},&space;H)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H^{k&plus;1}&space;=&space;arg\min_{&space;H&space;\geqslant&space;0&space;}&space;f(W^{k&plus;1},&space;H)" title="H^{k+1} = arg\min_{ H \geqslant 0 } f(W^{k+1}, H)" /></a>

Applying projected gradient method into alternating non-negative least squares algorithm we could get a faster convergence rate. Proposed projected gradient method can be seen in the pseudo code below: 

![Figure1](https://github.com/MingSheng92/NMF/blob/master/pg_method.PNG)

For more information on the proofs and intuition of the algorithm, please kindly refer to the [research paper](https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf).

## How to use

Example on how to use program can be found at [PG-NMF_example.ipynb](https://github.com/MingSheng92/NMF/blob/master/PG-NMF_example.ipynb).

Additional information: \
https://angms.science/doc/NMF/nmf_pgd.pdf \

latex generated with https://www.codecogs.com/latex/eqneditor.php
