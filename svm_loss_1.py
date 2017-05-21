import numpy as np

def svm_loss(W, X, y, C, margin=1):
    """the svm loss function of multi_linear classifier
    * W means linear_model
    * X is our training data
    * y is label vector
    * C is the nomalization coefficient
    * margin is the svm margin parameter
    """
    answer = W.dot(X)
    correct = answer[y, np.arange(X.shape[1])]
    mat = answer - correct + margin
    mat[y, np.arange(X.shape[1])] = 0
    thresh = np.maximum(np.zeros_like(mat), mat)
    loss = np.sum(thresh)
    loss /= X.shape[1]
    loss += 0.5 * C * np.sum(W * W) #normalization
    return loss
    

