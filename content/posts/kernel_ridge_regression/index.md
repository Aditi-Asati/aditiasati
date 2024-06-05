---
title: "How to Kernelize the Ridge Regression Algorithm"
date: 2024-06-04T08:06:25+06:00
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

The ridge regression algorithm learns a linear function to map input points to a real number by minimizing an objective function. The optimization problem in ridge regression is given by:

$$
\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \langle w, \Phi(X_i) \rangle \right)^2 + \lambda \lVert w \rVert_2^2$$

Here, $\lambda$ denotes the tradeoff constant. The first term in the objective function is the training error (also called empirical risk of a predictor function) in terms of linear least squares error. The second term is the regularization term penalizing functions which have large L2-norm squared.  

Linear methods like ridge regression work best when the underlying dataset is sort of linear. However, when we use linear methods to model a non-linear dataset, the machine learning model often tends to underfit the dataset as the model class is not strong enough to predict well on such a dataset.

Kernel methods were introduced to overcome this problem. Kernel methods allow us to implement linear methods on a non-linear dataset by implicitly embedding the input into a high-dimensional euclidean space in which the data-points become linear. 

In this blog, we will learn how to kernelize the Ridge Regression algorithm and also see an implementation of it using `scikit-learn` library.

For we begin, let's familiarize ourselves with two terms which we will use later on. 

- A **kernel function** takes two points in the input space $\mathcal{X}$ and maps them to a real number. Mathematically speaking, a kernel function is a function $k : \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ where $\mathbb{R}$ is the set of all real numbers. Intuitively, kernel function measures how ”similar” two points are in the feature space.

Now that we have understood what kernel functions are, let us understand kernel matrix.

* Given a kernel function $k$ and a set of points $x_1, ..., x_n \in \mathcal{X}$, the corresponding **kernel matrix** is defined as $K = (k(x_i, x_j))_{i,j \in n \times n}$. So each $ij$-th entry in the kernel matrix $K$ is the value of the kernel function at points $x_i$ and $x_j$.

We are now ready to derive the kernel version of the Ridge Regression algorithm.

The idea is to rewrite the optimization problem of the Ridge Regression algorithm in terms of the kernel function or the kernel matrix. To this end, we will invoke the ***Representer theorem*** which basically states that for an optimization problem of the form:
$$ \min_{w \in \mathcal{H}} \left( R_n(w) + \lambda \Omega(\|w\|_{\mathcal{H}}) \right) $$ 

$\quad$ where:
- $R_n$ is the empirical risk (training error) 
- $\Omega$ is the regularizer function
- $\lambda$ is the tradeff constant
- $\mathcal{H}$ is the euclidean feature space 
- $w$ is a vector in $\mathcal{H}$ 

{{<alert type="tip">}}
$\mathcal{H}$ is actually the space of all real-valued functions from the input space $X$ to $\mathbb{R}$. It is infact a vector space.

{{</alert>}}

The solution exists and is given by:
$${w^* = \sum_{i=1}^{n} \alpha_i k(X_i, \cdot)}$$

$\quad$ where:
- $\alpha_i$ is a real number for all $i$
- $k(X_i, .)$ is a function from the input space to $\mathbb{R}$.

Notice that the optimization problem in Ridge Regression appears in the same form as mentioned in the Representer theorem. 
Hence, we can apply the theorem and obtain a solution $w$ of the Ridge Regression problem as a linear combination of input points in the feature space:

$$w = \sum_{j=1}^n \alpha_j\Phi(X_j)$$

Now this $w$ can be substituted back into the above ridge regression problem yielding the following optimization problem in terms of the kernel matrix $K$:

$$\min_{\alpha \in \mathbb{R}^n} \frac{1}{n} \lVert Y - K\alpha \rVert_2^2 + \lambda \alpha^T K \alpha$$

The solution to this optimization problem can be derived analytically. Let's take a look at the derivation:

**Step 1**:
The objective function in the above optimization problem is 

$$Obj(\alpha) := \frac{1}{n} \lVert Y - K\alpha \rVert_2^2 + \lambda \alpha^T K \alpha$$

It is a convex function, therefore, any local minima is already a global minima of the function. Hence, it is enough to find one local minima of the objective function.

**Step 2**:
In order to find a local minima, we will take the derivative of the objective function with respect to $\alpha$ and set it to 0:

$$grad(Obj)(\alpha) = -\frac{1}{n}K^T(y-K\alpha) + \lambda K\alpha + 0$$

(Note that $K$ is symmetric)

$$K(-y + K\alpha + n\lambda \alpha) = 0$$

This implies
$$\alpha = (K + n\lambda I)^{-1}Y$$.

And we have obtained the solution of the Kernel Ridge Regression problem.

Now lets also formulate the evaluation function $f(x)$ in terms of the kernel function, after all the essence of kernelizing an algorithm is to write the optimization problem and the evaluation/ target function in terms of the kernel function/ kernel matrix.

$$f(x) = \langle w, \Phi(x) \rangle = \langle w, k(x, .) \rangle$$

Note that $\Phi(x) = k_x = k(x, .)$

Substituting the value of w (given by the representer theorem) into the above equation yields the following:

$$f(x) = \langle \sum_{i=1}^{n} \alpha_i k(X_i, \cdot), \Phi(x) \rangle$$


$$f(x) =\sum_{j=1}^n \alpha_jk(X_j,x)$$

```
pip install scikit-learn
```