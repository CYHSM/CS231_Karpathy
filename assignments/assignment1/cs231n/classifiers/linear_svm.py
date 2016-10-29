import ipdb
import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    delta = 1
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        dmargins = np.zeros(num_classes)
        for j in xrange(num_classes):
            margin = scores[j] - correct_class_score + delta  # note delta = 1
            dmargins[j] = margin
            if margin > 0:
                loss += margin

        # All positive margins are increased while the real class is decreased
        # if there are positive margins
        dW[:, dmargins > 0] += X[i][:, np.newaxis]
        dW[:, y[i]] -= np.sum(dmargins > 0) * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    loss -= delta
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    ##########################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    ##########################################################################
    dW /= num_train
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    delta = 1
    num_classes = W.shape[1]
    num_train = X.shape[0]

    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    ##########################################################################

    scores = X.dot(W)
    # Without list comprehension making use of the fact that y is aligned with
    # X
    correct_class_scores = scores[np.arange(num_train), y]
    # Calculate margin vectorized
    margins = scores - correct_class_scores[:, np.newaxis]
    # Add delta
    margins += delta
    # Add margins > 0
    loss += np.sum(margins[margins > 0])
    # Subtract real classes if higher zero
    margins_real_class = margins[np.arange(num_train), y]
    loss -= np.sum(margins_real_class[margins_real_class > 0])
    # Divide by number of trains and add regularization
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    # Calculate gradient
    pos_part = np.zeros(margins.shape)
    neg_part = np.zeros(margins.shape)
    #Get all positive margins to add to dW
    pos_part[margins>0] = 1
    #Get all class margins to subtract from dW
    col_sum = np.sum(pos_part, axis=1)
    neg_part[range(num_train), y] = -col_sum[range(num_train)]

    #neg_part[margins==delta] = 1
    combined_mask = pos_part - (-1)*neg_part
    #Calc dw based on dot product of combined mask and input X
    dW = X.T.dot(combined_mask)
    dW /= num_train
    #Add regularization
    dW += reg * W

    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    ##########################################################################
    pass
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    return loss, dW

# loss, dw = svm_loss_vectorized(np.array([[1, 2, 0, 1], [3, 1, 1, 0], [
#                                0, 0, 0, -2]]).T, np.array([[1, 2, 1, 2], [3, 1, 0, 1], [4, 3, 2, 1], [5, 6, 7, 8], [0,1,2,5]]), np.array([1, 0, 2, 0, 1]), 0.1)
# print(loss)
# print(dw)
