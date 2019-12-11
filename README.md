# Projected Gradient for Non-Negative Matrix Factorization

This is only an attempt to implement NMF algorithm proposed in C. Lin, <b>"Projected Gradient Methods for Nonnegative Matrix Factorization,"</b> in Neural Computation, vol. 19, no. 10, pp. 2756-2779, Oct. 2007.
doi: 10.1162/neco.2007.19.10.2756

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

## Results 

As shown in the figure below, we manage to get a decent reconstuction(denoise) from the noisy images. 

![Figure2](https://github.com/MingSheng92/NMF/blob/master/ProjectedNMF.jpg)

From the figure above, YaleB dataset has better reconstruction result, as I believe this is due to the dataset have perfectly centered face compared to ORL dataset. This suggests that we need some pre-processing on ORL to obatain a better result.

## How to use

Example on how to use program can be found at [PG-NMF_example.ipynb](https://github.com/MingSheng92/NMF/blob/master/PG-NMF_example.ipynb).

## Future work 

Will try to implement a few more different algorithms so that we can compare the robustness of different proposed approach.

## Additional information

https://angms.science/doc/NMF/nmf_pgd.pdf 

latex generated with https://www.codecogs.com/latex/eqneditor.php
