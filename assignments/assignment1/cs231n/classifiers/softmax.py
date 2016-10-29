import ipdb
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        # With numerical problems
        #loss = np.exp(correct_class_score) / np.sum(np.exp(scores))

        # Without numerical problems
        logC = -np.max(scores)
        p = np.exp(correct_class_score + logC) / np.sum(np.exp(scores + logC))
        loss += -np.log(p)

        # Compute the margins without numerical problems
        margins = np.exp(scores + logC) / np.sum(np.exp(scores + logC))

        # Compute dScores with derivative trick
        dScores = margins - (range(num_classes) == y[i])

        # Compute gradient on dScores
        # Bit of python syntax: Either we transpose
        #dW += (dScores[:,np.newaxis]*X[i]).T
        # or we use [:,newaxis]
        dW += dScores * X[i][:, np.newaxis]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    #######################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #######################################################################

    # Calculate scores
    scores = X.dot(W)

    # Calculate correct class score for each sample
    correct_class_scores = scores[range(num_train),y]

    # Calculate loss without numerical problems
    logC = -np.max(scores, axis=1)
    p = np.exp(correct_class_scores + logC) / np.sum(np.exp(scores + logC[:,np.newaxis]), axis=1)
    loss = np.sum(-np.log(p))

    # Compute margins without numerical problems
    margins = np.exp(scores + logC[:,np.newaxis]) / np.sum(np.exp(scores + logC[:,np.newaxis]), axis=1)[:,np.newaxis]

    # Compute dScores with derivative trick
    correct_label_matrix = np.zeros(margins.shape)
    # All zeros, except where y = k
    correct_label_matrix[range(num_train),y] = 1
    dScores = (margins - correct_label_matrix).T

    # Calculate gradient on scores
    dW = dScores.dot(X).T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

#######################################################################
#                          END OF YOUR CODE                                 #
#######################################################################

    return loss, dW

# Test softmax.py
# loss, dw = softmax_loss_vectorized(np.array([[1, 2, 0, 1], [3, 1, 1, 0], [
#     0, 0, 0, -2]], dtype='float').T, np.array([[1, 2, 1, 2], [3, 1, 0, 1], [4, 3, 2, 1], [5, 6, 7, 8], [0, 1, 2, 5]]), np.array([1, 0, 2, 0, 1]), 0.1)
# print(loss)
# print(dw)
