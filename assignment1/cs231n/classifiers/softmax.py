from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores=X.dot(W)
    scores=scores-np.max(scores)
    scores=np.exp(scores)
    scoresForGrad=np.zeros_like(scores)
    for i in range(X.shape[0]):
        scores[i]=scores[i]/np.sum(scores[i])
        scoresForGrad[i]=scores[i]
        scores[i]=-np.log(scores[i])
    for i in range(X.shape[0]):
        loss=loss+scores[i][y[i]]
    loss/=X.shape[0]
    loss+=reg*np.sum(W*W)
    for i in range(X.shape[0]):
        scoresForGrad[i][y[i]]-=1
        dW+=(X[i].reshape(X.shape[1],1)).dot(scoresForGrad[i].reshape(1,W.shape[1]))
    dW/=X.shape[0]
    dW+=2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores=X.dot(W)
    scores=scores-np.max(scores)
    scores=np.exp(scores)
    sumScores=np.sum(scores,axis=1).reshape(X.shape[0],1)
    scores/=sumScores

    matrixForGrad=np.zeros_like(scores)
    matrixForGrad[np.arange(X.shape[0]),y]=1
    matrixForGrad=scores-matrixForGrad
    dW=(X.T).dot(matrixForGrad)

    lossVec=-np.log(scores[np.arange(scores.shape[0]),y])
    loss=np.sum(lossVec)
    loss/=X.shape[0]
    dW/=X.shape[0]
    loss+=reg*np.sum(W*W)
    dW+=2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
