Download link :https://programming.engineering/product/ece421-assignment-3-unsupervised-learning-and-probabilistic-models/


# ECE421-Assignment-3-Unsupervised-Learning-and-Probabilistic-Models
ECE421 – Assignment 3: Unsupervised Learning and Probabilistic Models
Objectives:

In this assignment, you will implement learning and inference procedures for some of the proba-bilistic models described in class, apply your solutions to some simulated datasets, and analyze the results.

General Note:

Full points are given for complete solutions, including justifying the choices or assumptions you made to solve each question.

A written report should be included in the nal submission. Do not dump your codes and outputs in the report. Keep it short, readable, and well-organized.

Programming assignments are to be solved and submitted individually. You are encouraged to discuss the assignment with other students, but you must solve it on your own.

Please ask all questions related to this assignment on Piazza, using the tag pa3.

There are 3 starter les attached, helper.py, starter_kmeans.py and starter_gmm.py which will help you with your implementation.

1 K-MEANS [9 PT.]

K-means [9 pt.]

K-means clustering is one of the most widely used data analysis algorithms. It is used to summarize data by discovering a set of data prototypes that represent clusters of data. The data prototypes are usually referred to as cluster centers. Usually, K-means clustering proceeds by alternating between assigning data points to clusters and then updating the cluster centers. In this assignment, we will investigate a di erent learning algorithm that directly minimizes the K-means clustering loss function.

1.1 Learning K-means

The K cluster centers can be thought of as K, D-dimensional parameter vectors and we can place them in a K D parameter matrix , where the kth row of the matrix denotes the kth cluster center k. The goal of K-means clustering is to learn such that it minimizes the loss function,

N

kk22, where N is the number of training observations.

L( ) = Pn=1 minkK=1 kxn

Even though the loss function is not smooth due to the \min” operation, one may still be able to nd its solutions through iterative gradient-based optimization. The \min” operation leads to discontinuous derivatives, in a way that is similar to the e ect of the ReLU activation function, but nonetheless, a good gradient-based optimizer can work e ectively.

Implement the distance_func() function in starter_kmeans.py le to calculate the squared pairwise distance for all pair of N data points and K clusters.

def distance_func(X, mu):

Inputs

X: is an NxD matrix (N observations and D dimensions)

mu: is an KxD matrix (K means and D dimensions)

Outputs

pair_dist: is the squared pairwise distance matrix (NxK)

Hint: To properly use broadcasting, you can rst add a dummy dimension to X, by using tf.expand_dims or torch.unsqueeze, so that its new shape becomes (N, 1, D).

For the dataset data2D.npy, set K = 3 and nd the K-means clusters by minimizing the L( ) using the gradient descent optimizer. The parameters should be initialized by sampling from the standard normal distribution. Include a plot of the loss vs the number of updates.

Use the Adam optimizer for this assignment with the following hyper-parameters: learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5. The learning should converge within a few hundred updates.

Hold out 1/3 of the data for validation, and for each value of K = 1; 2; 3; 4; 5:

Train a K-means model.

2 MIXTURES OF GAUSSIANS [16 PT.]

Include a 2D scatter plot of training data points colored by their cluster assignments.

Compute and report the percentage of the training data points belonging to each of the K clusters. (include this information in the legend of the scatter plot.)

Compute and report the loss function over validation data. (include this in the caption of the scatter plot.)

Based on the scatter plots, comment on the best number of clusters to use.

Mixtures of Gaussians [16 pt.]

Mixtures of Gaussians (MoG) can be interpreted as a probabilistic version of K-means clus-tering. For each data vector, MoG uses a latent variable z to represent the cluster assign-

ment and uses a joint probability model of the cluster assignment variable and the data vec-tor: P (x; z) = P (z)P (x j z). For N iid training cases, we have P (X; z) = QNn=1 P (xn; zn). The Expectation-Maximization (EM) algorithm is the most commonly used technique to learn a MoG.

Like the standard K-means clustering algorithm, the EM algorithm alternates between updating the cluster assignment variables and the cluster parameters. What makes it di erent is that in-stead of making hard assignments of data vectors to cluster centers (the \min” operation above),

the EM algorithm computes probabilities for di erent cluster centers, P (zjx). These are computed from P (z = kjx) = P (x; z = k)= PKj=1 P (x; z = j).

While the Expectation-Maximization (EM) algorithm is typically the go-to learning algorithm to train MoG and is guaranteed to converge to a local optimum, it su ers from slow convergence. In this assignment, we will explore a di erent learning algorithm that makes use of gradient descent.

2.1 The Gaussian cluster mode [7 pt.]

Each of the K mixture components in the MoG model occurs with probability k = P (z = k). The data model is a multivariate Gaussian distribution centered at the cluster mean (data center) k 2 RD. We will consider a MoG model where it is assumed that for the multivariate Gaussian for cluster k, di erent data dimensions are independent and have the same standard deviation, k.

covariance = 0

Use the K-means distance function distance_func implemented in 1.1 to implement log_gauss_pdf function by computing the log probability density function for cluster k: log N (x ; k; k2)

for all pair of N data points and K clusters. Include the snippets of the Python code.

Write a vectorized Tensor ow Python function that computes the log probability of the cluster variable z given the data vector x: log P (zjx). The log Gaussian pdf function implemented above should come in handy. The implementation should use the function reduce_logsumexp() provided in the helper functions le. Include the snippets of the Python code and comment on why it is important to use the log-sum-exp function instead of using tf.reduce_sum.

2.2 Learning the MoG [9 pt.] 2 MIXTURES OF GAUSSIANS [16 PT.]

2.2 Learning the MoG [9 pt.]

The marginal data likelihood for the MoG model is as follows (here \marginal” refers to summing over the cluster assignment variables):

N K

Y X

P(X) =

P (xn) =

P (zn = k)P (xn j zn = k)

n=1

n=1 k=1

= Yn

Xk

kN (xn ; k; k2)

The loss function we will minimize is the negative log likelihood L( ; ; ) = log P (X). The maximum likelihood estimate (MLE) is a set of the model parameters ; ; that maximize the log likelihood or, equivalently, minimize the negative log likelihood.

Implement the loss function using log-sum-exp function and perform MLE by directly opti-mizing the log likelihood function using gradient descent.

Note that the standard deviation has the constraint of 2 [0; 1). One way to deal with

this constraint is to replace 2 with exp( ) in the math and the software, where is an

unconstrained parameter. In addition, has a simplex constraint, that is

k

k = 1. We

can again replace this constrain with unconstrained parameter through a

softmax function

P

k = exp( k)=

k0

exp( k0 ). A log-softmax function, logsoftmax, is provided for convenience

in the helper

functions le.

P

For the dataset data2D.npy, set K = 3 and report the best model parameters it has learned.

Include a plot of the loss vs the number of updates.

Hold out 1/3 of the data for validation, and for each value of K = 1; 2; 3; 4; 5:

Train a MoG model.

Include a 2D scatter plot of training data points colored by their cluster assignments.

Compute and report the percentage of the training data points belonging to each of the K clusters. (include this information in the legend of the scatter plot.)

Compute and report the loss function over validation data. (include this in the caption of the scatter plot.)

Explain which value of K is best, based on the validation loss.

Run both the K-means and the MoG learning algorithms on data100D.npy for K = f5; 10; 15; 20; 30g (Hold out 1/3 of the data for validation). Comment on how many clusters you think are within the dataset by looking at the MoG validation loss and K-means validation loss. Compare the learnt results of K-means and MoG.

