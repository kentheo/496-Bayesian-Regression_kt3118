# -*- coding: utf-8 -*-

"""
Use this file for your answers.

This file should been in the root of the repository
(do not move it or change the file name)

"""

import numpy as np

def lml(alpha, beta, Phi, Y):
    """
    4 marks

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: the log marginal likelihood, a scalar
    """
    D = Y.shape[0]
    constant = -D/2 * np.log(2*np.pi)
    term = alpha * np.dot(Phi, Phi.T) + beta * np.identity(N)

    return constant - (0.5 * np.log(np.linalg.det(term))) - (0.5 * np.dot(np.dot(Y.T,np.linalg.inv(term)), Y))

def grad_lml(alpha, beta, Phi, Y):
    """
    8 marks (4 for each component)

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: array of shape (2,). The components of this array are the gradients
    (d_lml_d_alpha, d_lml_d_beta), the gradients of lml with respect to alpha and beta respectively.
    """
    pass
