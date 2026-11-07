from enum import Enum


class SVMKernel(Enum):
    """
    Kernel function of a one-class SVM.

    Attributes
    ----------
    LINEAR
        Linear kernel.
    POLYNOMIAL
        Polynomial kernel.
    RBF
        Radial basis function kernel.
    SIGMOID
        Sigmoid kernel.
    """

    LINEAR = "linear"
    POLYNOMIAL = "poly"
    RBF = "rbf"
    SIGMOID = "sigmoid"
