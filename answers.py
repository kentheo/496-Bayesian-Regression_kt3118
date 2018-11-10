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
    constant = -0.5 * D * np.log(2.0*np.pi)
    term = (alpha * np.dot(Phi, Phi.T)) + (beta * np.identity(D))

    result = constant - (0.5 * np.log(np.linalg.det(term))) - (0.5 * np.dot(np.dot(Y.T, np.linalg.inv(term)), Y))

    return result[0][0]

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
    grad = np.zeros((1,2))

    D = Y.shape[0]
    term = (alpha * np.dot(Phi, Phi.T)) + (beta * np.identity(D))
    inv_term = np.linalg.inv(term)
    Phi_Phi_T = np.dot(Phi, Phi.T)

    term2 = np.dot(inv_term, np.dot(Phi_Phi_T, inv_term))

    grad[0,0] = -0.5 * np.matrix.trace(np.dot(inv_term, Phi_Phi_T)) + 0.5 * np.dot(Y.T, np.dot(term2, Y))

    inv_inv = np.dot(inv_term, inv_term)
    grad[0,1] = -0.5 * np.matrix.trace(inv_term) + 0.5 * np.dot(Y.T, np.dot(inv_inv, Y))

    return grad
