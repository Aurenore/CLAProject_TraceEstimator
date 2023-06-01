# Computational Linear Algebra: Trace Estimator Project
MATH-453, EPFL 
# Introduction
Given a symmetric invertible matrix $A \in \mathbb{R}^{n\times n}$, an important task, for example in computational chemistry, is to compute the trace of its inverse $tr{(A^{-1})}$
where $tr{(\cdot)}$ denotes the trace of a matrix, that is, the sum of its diagonal elements.
The main goal of this project is to implement Algorithm 2 from Bai et al. paper ["Some large-scale matrix computation problems"](https://doi.org/10.1016/0377-0427(96)00018-0) for estimating $tr(A^{-1})$. We will study the method based on the Lanczos procedure and the Monte-Carlo method. To estimate a Riemann-Stieltjes integral, we will specifically focus on the Gauss-Radau rule. Finally, we will present numerical experiments for various positive definite matrices.

# Structure of the repository

The repository is structured as follows:
```
┣figures/ : folder containing the figures from algorithm 1, algorithm 2 and performances comparison
┣graphs_algo1.ipynb : notebook implementing algorithm 1 and creating the plots 
┣graphs_algo2.ipynb : notebook implementing algorithm 2 and creating the plots 
┣helpers.py : python file containing helper functions
┣matrices.py : python file containing functions that outputs SPD matrices
┗running_time.ipynb : runs different implementations to compute the trace of a matrix and plots the running times to compare them.
```