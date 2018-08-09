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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes=W.shape[1]
  num_train=X.shape[0]
  for i in range(num_train):
    scores=np.exp(X[i].dot(W))
    correct_scores=scores[y[i]]
    loss-=np.log(correct_scores/np.sum(scores))
    dW[:,y[i]] -=X[i].T
    for j in range(num_classes):
      dW[:,j]+=(1/np.sum(scores))*scores[j]*X[i].T
    
  loss /= num_train
  dW /=num_train

  
  pass
  #############################################################################
  # 
  loss += reg * np.sum(W * W) 
  dW +=reg*W
  
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores=np.exp(X.dot(W))
  num_classes=W.shape[1]
  num_train=X.shape[0]

  correct_scores=scores[np.arange(num_train),y]
  scores_sum=np.sum(scores,axis=1)
  loss=-np.sum(np.log(np.reshape(correct_scores,(scores_sum.shape))/scores_sum))
  loss /= num_train
  margin=np.zeros([num_train,num_classes])
  margin[np.arange(num_train),y]=-1

  scores_sum=scores_sum.reshape(num_train,1)
  scores_sumx=scores_sum.repeat(num_classes,axis=1)

  margin+=scores/scores_sumx
  dW=X.T.dot(margin)
  dW /=num_train





  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  loss += reg * np.sum(W * W) 
  dW +=reg*W
  #############################################################################

  return loss, dW

