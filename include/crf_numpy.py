import numpy as np
from include import permutohedral_lattice as phl

"""
Computes energy as a sum of unary and pairwise energy. 
This implementation uses numpy and an implementation of the Permutohedral Lattice
to perform efficient message passing
"""
def applyCRFNumpy(image, unary, sigma_alpha, sigma_beta, sigma_gamma, w1, w2, iterations=10):
    """
    :param image: original image as numpy array of shape (w,h,c)
    :param unary: unary likelihoods of label assignment per pixel as numpy array of shape (w,h,L)
    :param sigma_alpha: size of the gaussian kernel for pixel distance of the first filter
    :param sigma_beta: size of the gaussian kernel for image intensity of the first filter
    :param sigma_gamma: size of the gaussian kernel for pixel distance of the second filter
    :param w1: weight of the first filter
    :param w2: weight of the second filter
    :param iterations: number of iterations. Default: 10 
    :return: numpy array containing energies with shape (w,h,L)
    """
    w = image.shape[0]
    h = image.shape[1]
    
    grid = np.transpose(np.array(np.meshgrid(np.arange(w),np.arange(h))),axes=(2,1,0))
    
    # Filter 1
    # Initialize higher dimensional representation 
    p1 = np.zeros((w,h,5))
    p1[:,:,:2] = grid / sigma_alpha
    p1[:,:,2:] = image / sigma_beta
    
    # Filter 2
    # Initialize higher dimensional representation 
    p2 = grid / sigma_gamma
    
    # Initialize Q
    Q = np.exp(-unary)
    
    # Normalize Q
    Q = (Q.reshape(w*h,-1).T / Q.sum(2).reshape(w*h).T).T.reshape(w,h,-1)
    
    for i in range(iterations):
        print("CRF iteration %d/%d" % (i,iterations))
        # Efficient Message passing using the Permutohedral lattice
        Q_tilde1 = phl.PermutohedralLattice.filter(Q,p1)
        Q_tilde2 = phl.PermutohedralLattice.filter(Q,p2)

        # Combine filters
        Q_tilde = (w1 * Q_tilde1 + w2 * Q_tilde2)


        # Compatibility transform
        Q_hat = np.zeros(Q.shape)
        for l in range(Q.shape[2]):
            Q_hat[:,:,l] += np.sum(Q_tilde[:,:,:l], 2) + np.sum(Q_tilde[:,:,(l+1):], 2)
        
        # Local update
        Q = np.exp(-unary - Q_hat)
    
        # Normalize
        Q = (Q.reshape(w*h,-1).T / Q.sum(2).reshape(w*h).T).T.reshape(w,h,-1)

    return Q

