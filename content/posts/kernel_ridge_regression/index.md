---
title: "How to Kernelize the Ridge Regression Algorithm"
date: 2024-06-06T08:06:25+06:00
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
{{< vs 3 >}}

The solution to this optimization problem can be derived analytically. Let's take a look at the derivation:

{{< vs 3 >}}

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
$$\alpha = (K + n\lambda I)^{-1}Y.$$

And we have obtained the solution of the Kernel Ridge Regression problem!!

Now lets also formulate the evaluation function $f(x)$ in terms of the kernel function, after all the essence of kernelizing an algorithm is to write the optimization problem and the evaluation/ target function in terms of the kernel function/ kernel matrix.

$$f(x) = \langle w, \Phi(x) \rangle = \langle w, k(x, .) \rangle$$

{{<alert type="info">}}
Note that $\Phi(x) = k_x = k(x, .)$.
{{</alert>}}

Substituting the value of w (given by the representer theorem) into the above equation yields the following:

$$f(x) = \langle \sum_{i=1}^{n} \alpha_i k(X_i, \cdot), k(x,.) \rangle$$


$$f(x) =\sum_{i=1}^n \alpha_i \langle k(X_i, .),k(x,.) \rangle = \sum_{i=1}^n \alpha_i \langle \Phi(X_i),\Phi(x) \rangle$$

{{<alert type="info">}}
Note that $k(x,y) = \langle \Phi(x), \Phi(y) \rangle$. For more details regarding it, read [here](https://drive.google.com/file/d/1QbEQNjbfIPkVEe-qEvwFPEyLx-tg255r/view?usp=sharing).
{{</alert>}}

$$f(x) = \sum_{i=1}^n\alpha_ik(X_i, x)$$

We have finally obtained the target function for the kernel ridge regression algorithm. Observe that the function depends only on the kernel and not on the input points explicitly as discussed earlier. 

Now, lets look at the implementation of the Kernel Ridge Regression algorithm using `scikit-learn` library.

We will begin by installing necessary dependencies.
Run the following command on the terminal.

```py
pip install scikit-learn
```
{{< vs 3 >}}

For demonstration, we will use the diabetes toy dataset present in the `scikit-learn` library.

Lets now import the required utilities.

```py
from sklearn.datasets import load_diabetes
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
```
{{< vs 3 >}}

We will now load the diabetes dataset and extract the `data` matrix and the `target` array.

```py
diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target
```
{{< vs 3 >}}

Its time to split the dataset into train and test sets in order to evaluate the generalization performance of the Kernel Ridge Regression (KRR) model and also to avoid overfitting.

```py
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
```

{{<alert type="info">}}
It's common practice to standardize the training and testing features (`X_train` and `X_test`) before training a model. However, we omit this step here because the feature matrix `data` provided by the `load_diabetes()` function is already standardized.
{{</alert>}}

{{< vs 3 >}}

We will be tuning the hyperparameters of the KRR model using Gridsearch cross validation method. 

{{<alert type="info">}}
It's crucial to always tune the hyperparameters of any machine learning model while training, since the performance of the model is very sensitive to the choice of the hyperparameters. As an example, think of how changing the regularization constant in KRR algorithm can affect its training.
{{</alert>}}

```py
krr_model = KernelRidge()
param_grid = {"alpha": [1e-5, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2], "kernel": ["linear", "rbf", "poly", "sigmoid", ]}
grid_search = GridSearchCV(krr_model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, cv=5)
```
{{<alert type="info">}}
To understand Grid search cross validation, read [here](https://towardsdatascience.com/cross-validation-and-grid-search-efa64b127c1b).
{{</alert>}}

{{< vs 3 >}}

Lets train our KRR model on the diabetes dataset and get the best hyperparameter values along with the trained model.

```py
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(best_params)
best_model = grid_search.best_estimator_
```
{{< vs 3 >}}

Now that we have trained our KRR model, its time to make predictions on the test set and compute the performance metrics on both the training and test sets.

```py
predictions = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, predictions)
test_mse = mean_squared_error(y_test, predictions)

train_predictions = best_model.predict(X_train)
train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)

print(f"Test MAE : {test_mae} and Test MSE : {test_mse}")
print(f"Train MAE : {train_mae} and train MSE : {train_mse}")
```
{{< vs 3 >}}

Here's the full code:
{{< gist Aditi-Asati 996d0cd86b7fd15911cefe44a608c225 >}}

Woohoo! We have come a long way. I hope you found this blog helpful.
Thanks for reading!