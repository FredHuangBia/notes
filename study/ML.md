# Machine Learning
## Logistic Regression
The function used to map from variable $x$ to a probability is the sigmoid: 
$$\sigma(x)=\frac{1}{1+e^{-x}}$$

Now we use a parameterized function
$$h(x)=\frac{1}{1+e^{-w^Tx+b}}$$

If we add one extra value to vector $x$, $x=x_1,x_2,...,x_n,1$, then we could write the above equation in a easier format
$$h(x)=\frac{1}{1+e^{-w^Tx}}$$

Given some data points, given $y^i\in\{0,1\}$ to solve for the best parameter $w$, we could use likelihood function
$$\prod_{i=1}^{i=n} h(x^i)^{y^i}(1-h(x^i))^{1-y^i}$$

Maximize the above equation is equivalent to maximize the log if it, so we take log and divide by $n$, it becomes
$$ \frac{1}{n} \sum_{i=1}^{i=n}y^i\log(h(x^i))+(1-y^i)\log(1-h(x^i)) $$

We could use gradient decent or derivative equals 0 to slove for $w$.

Notice that if we have a fixes threshold for deciding the classification, then the threshold is actually $e^{-w^Tx} = threshold$, it corresponds to a linear hyperplane.

Can logistic regression solve multi-class classification problem? Yeah! Replace the sigmoid function with softmax.
$$softmax_{k}(x)=\frac{e^{x_k}}{\sum_{i=1}^{i=d}e^{x_d}}$$

Here in logistic regression, it becomes
$$p(y=k)=h_k(x)=\frac{e^{{w_{k}^T}x_k}}{\sum_{i=1}^{i=d}e^{{w_{i}^T}x_i}}$$

Again, given the value of $y^i$, we could solve it with maximum likelihood.

Why doesn’t MSE work with logistic regression? \
One of the main reasons why MSE doesn’t work with logistic regression is when the MSE loss function is plotted with respect to weights of the logistic regression model, the curve obtained is not a convex curve which makes it very difficult to find the global minimum. This non-convex nature of MSE with logistic regression is because non-linearity has been introduced into the model in the form of a sigmoid function which makes the relation between weight parameters and Error very complex. Another reason is, for extreme wrong classification, loss value using MSE was much much less compared to the loss value computed using the log loss function. Hence it is very clear to us that MSE doesn’t strongly penalize misclassifications even for the perfect mismatch!

<br>

## Linear Regression
Let $\epsilon$ bethe noise term, we have
$$Y = X\beta + \epsilon$$
To solve it, we use least square
$$L = ||X\beta-Y||^2$$
$$L = (X\beta-Y)^T(X\beta-Y)=(\beta^TX^T-Y^T)(X\beta-Y)$$
$$L = \beta^TX^TX\beta - Y^TX\beta-\beta^TX^TY+Y^TY$$
$$\frac{\partial L}{\partial \beta}=2X^TX\beta-2X^TY=0$$
$$\beta=(X^TX)^{-1}X^TY$$

When $X^TX$ is not invertible, the solution is not unique, and we need to solve it by SVD.
<br>

## Perceptron

<br>

## SVM
Given a lot of data points $(x_i, y_i)$, where $y_i\in \{-1, 1\}$, we want to find a hyperplane $w^Tx+b=0$ that best seperates the data. By best, we want the gap between the hyperplane to the two categories to be as large as possible. For positive examples, we want $w^Tx+b \geq 1$, and for negative examples, we want $w^Tx+b\leq-1$. This is equivalent to $y(w^Tx+b)\geq 1$. The gap between the two hyperplanes $w^Tx+b=\{1,-1\}$ is $\frac{2}{||w||}$. So to maximize the gap, we want
$$\min ||x|| \;\; subject\; to\; y(w^Tx+b)\geq 1$$
But what if the data points are not linearly separable by a single hyperplane? We have to find some soft margin and a loss function. Now assume for each point, it contributes to the loss is $L=max(0,1-y(w^Tx+b))$. Now, for all the points that are correctly classified, $y(w^Tx+b)\geq 1$, so it does not contribute to the loss. Now the total loss function is
$$||w|| + \lambda \frac{1}{n}\sum_i max(0, 1-y_i(w^Tx_i+b))$$
Now, if we increase $\lambda$, we are getting punished more if we misclassify a point. This is enforcing to shrink the margin between the two hyperplanes. Because we have less tolerance for potentially wrong label, we tend to overfit. And vice versa.

We usually solve the dual problem. And for non-linearly separable data, we use kernel trick.
<br>

## Boosting
Sequentially combine a group of weak classifiers to form a stronger one.

<br>

## Naive Bayes
Bayes rule
$$P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}=\frac{P(X,Y)}{P(X)}$$
$P(Y|X)$ is called posterior, $P(Y)$ is called prior, $P(X|Y)$ is called likelihood, $P(X)$ is called evidence.

If $X$ contains multiple variables, and if we assume they are mutually independent given $Y$, then $P(X|Y)=\prod_i P(X_i|Y)$. Now even if we havent seen any data pint that the feature is $X$, but we've probabily seen all $P(X_i|Y)$.

What if the variables are continuous? We could divide them into several discrete buckets.

<br>

## Decicsion Tree
Given a lot of decision critrerion, we choose the one that maximize information Gain. We do this until all the data are correctly classified, or some other stopping criterion is met.

$$Entropy = H = \sum_i -p_i \log(p_i)$$
When we split data by a node, the original data distribution changes from one set to two smaller sets, each have their own entropy. We could weight the two child sets according to the number of data points in them.
$$IG = H(parent) - \sum_i w_i H(child)$$

<br>

## Random Forest
Decision tree is very sensitive to the training data. So it may not generalize well. Random forest generalize better. 

- Step 1: Build new sub datasets from the original one. Such as randomly select with replacement, this is called boostraping. 
  - This is trying to make our random forest less sensitive to the original data.
- Step 2: Randomly select a subset of features, such as $x_0, x_1$ from each subset. 
  - This is trying to reduce correlation between the trees.
- Step 3: Using the sub datasets and sub set of features, build trees for each of them.

When testing, we use all the trees to classify each test point. Then we take the majority voting, this is called aggregation. This boostraping + aggregation is called bagging.

How to pick number of features? Usually $\sqrt{NumFeatures}$.

<br>

## KNN & K-Means
KNN is a supervised learning method to classify or regress a new data according to its nearest neighbors, while K-Means is a unsupervised learning algorith for clustering.

<br>

## K-Means vs Mean-Shift vs DBScan
K-Means:
- given k centers
- update assignment of every point and update the center
- until max steps or no assignment change in last iteration

Mean-shift:
- Create a sliding window/cluster for each data-point
- Each of the sliding windows is shifted towards their centroid. This step will be repeated until no shift yields a higher density (number of points in the sliding window)
- Selection of sliding windows by deleting overlapping windows. When multiple sliding windows overlap, the window containing the most points is preserved, and the others are deleted.
- Assigning the data points to the sliding window in which they reside.

DBScan is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together

<br>

## GMM

## HMM
Hidden markov chain + observed variables. Transition matrix is the probability of going to each state from each state.  Emission matrix of the probability of observations at each state. Now, lets see we see a serieous of observations $Y$, and what is the most likely hidden states sequence $X$?
$$P(X|Y) = \frac{P(Y|X)P(X)}{P(Y)}=\frac{\prod_i P(Y_i|X_i)\prod_iP(X_i|X_{{i-1}})}{P(Y)}$$

<br>

## ROC & AUC
ROC curve is the polot of true positive rate (also called recall) versus false positive rate. The true positive rate is how likely a positive will be predicted correctly. And the false positive rate is how likely we mispredict a negative as positive.
$$TPR = \frac{TP}{TP+FN}$$
$$FPR = \frac{FP}{TN+FP}$$
AUC is the area under this curve. The larger, the better. Best is TPR=1 and FPR=0.

## Collaborative filtering
It is a commenly used method in recommendation system.

Collaborative filtering is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue than that of a randomly chosen person.

## $F_1$ score & $F_\beta$ scure
$$F_1=\frac{2 * precision * recall}{precision + recall}$$
$F_\beta$ score is a way to weight the importance of precision and recall differently. Why we don't use arithmatic average? Because that assumes precision and recall could replace each other, when one drop, the increase of the other could mitigate that. This is obviously wrong.

## How to choose ML algorithm?
**Size of training data**: If we have very few training data comparing to available features, its better to choose a high bias low variance model, such as linear regression, linear SVM, Naive Bayes etc. Otherwise, its better to choose a high variance low bias model, such as decision tree, kernel SVM, KNN, etc.

**Interpretability**: 

**Speed or training time:** 

## Mahalanobis distance
The Mahalanobis distance is a measure of distance between a point $P$ and a distribution $D$. It is a multi-dimensional generalization of the idea of measuring how many standard deviations away $P$ is from the mean of $D$. If the distribution is rescaled to have unit variance, then Mahalanobis distance is the same as Euclidean distance.

Given a probability distribution $Q$ with mean $y$, and covariance matrix $S$, the Mahalanobis distance of a point $x$ from $Q$ is:

$$d_M(x,y,Q)=\sqrt{(x-y)^T S^{-1} (x-y)^T}$$
