---
title: "How to Kernelize the Ridge Regression Algorithm"
date: 2020-06-08T08:06:25+06:00
description: Markdown rendering samples
menu:
  sidebar:
    name: Kernelize Ridge Regression
    identifier: kernel_ridge_regression
    weight: 30
author:
  name: Aditi Asati
  image: /images/author/aditi_edited_3.jpg
math: true
---

## How to kernelize the Ridge Regression Algorithm
The ridge regression algorithm learns a linear function to map input points to a real number by minimizing an objective function. The optimization problem in ridge regression is given by:

$$
\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \langle w, \Phi(X_i) \rangle \right)^2 + \lambda \lVert w \rVert_2^2$$

Here, $\lambda$ denotes the tradeoff constant. The first term in the objective function is the training error (also called empirical risk of a predictor function) in terms of linear least squares error. The second term is the regularization term penalizing functions which have large L2-norm squared.  

Linear methods like ridge regression work best when the underlying dataset is sort of linear. However, when we use linear methods to model a non-linear dataset, the machine learning model often tends to underfit the dataset as the model class is not strong enough to predict well on such a dataset.

Kernel methods were introduced to overcome this problem. Kernel methods allow us to implement linear methods on a non-linear dataset by implicitly embedding the input into a high-dimensional euclidean space in which the data-points become linear. 

In this blog, we will kernelize the Ridge Regression algorithm and see an implementation of it on a dataset using `scikit-learn` library.

**Representer theorem**: Let $\mathcal{X}$ and $\mathcal{Y}$ be the input space and output space respectively. Let $k : \mathcal{X} \times \mathcal{X} \rightarrow\mathbb{R}$ be a kernel, and let $\mathcal{H}$ be the corresponding RKHS. Given a training set $(X_i, Y_i)_{i=1, \ldots, n} \subset \mathcal{X} \times \mathcal{Y}$ and classifier $f_w(x) := \langle w, \Phi(x) \rangle_{\mathcal{H}}$, let $R_n$ denote the empirical risk of the classifier in relation to a loss function $l$, and $\Omega : [0, \infty[ \rightarrow \mathbb{R}$, which is a strictly monotonically increasing function. 
Consider the following regularized risk minimization problem:
    
$$\min_{w \in \mathcal{H}} \left( R_n(w) + \lambda \Omega(\|w\|_{\mathcal{H}}) \right)$$
    
Then, the theorem states that the optimal solution of the problem always exists and is given as
    
$${w^* = \sum_{i=1}^{n} \alpha_i k(X_i, \cdot)}$$
    
From the representer theorem, the solution $w$ of problem can be expressed as a linear combination of input points in the feature space:

$$w = \sum_{j=1}^n \alpha_j\Phi(X_j)$$

Substituting $w$ in the above ridge regression problem yields the following optimization problem:

$$\min_{\alpha \in \mathbb{R}^n} \frac{1}{n} \lVert Y - K\alpha \rVert_2^2 + \lambda \alpha^T K \alpha$$

where $K$ is the kernel matrix defined.
The solution can be derived analytically and is given by

$${\alpha = (n\lambda I + K)^{-1}Y}$$

We can also calculate the prediction for an unseen point just using kernels as

$$f(x) =\sum_{j=1}^n \alpha_jk(X_j,x)$$

```
pip install scikit-learn
```