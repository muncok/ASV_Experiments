import numpy as np
from scipy.stats import multivariate_normal as gaussian
from .optimizer import optimize_maximum_likelihood
from .optimizer import calc_scatter_matrices

def transform_X_to_U(data, inv_A, m):
    return np.matmul(data - m, inv_A.T)

class Verifier:
    def __init__(self, model=None):
        self.m = None
        self.A = None
        self.Psi = None
        self.inv_A = None

        if model is not None:
            self.m = model.m
            self.A = model.A
            self.Psi = model.Psi
            self.inv_A = model.inv_A

    def fit(self, data, labels):
        assert len(data.shape) == 2
        assert len(labels) == data.shape[0]

        S_b, S_w = calc_scatter_matrices(data, labels)
        matrix_rank = np.linalg.matrix_rank(S_w)
        assert(matrix_rank == data.shape[-1])

        X = data
        self.m, self.A, self.Psi, self.relevant_U_dims, self.inv_A = \
            optimize_maximum_likelihood(X, labels)

        assert len(self.relevant_U_dims) == data.shape[1]

    def calc_llr(self, enr_X, tst_X):
        enr_U = transform_X_to_U(enr_X, self.inv_A, self.m)
        tst_U = transform_X_to_U(tst_X, self.inv_A, self.m)

        n_enr = len(enr_U)
        enr_var = np.var(enr_U, 0)
        cov_diag = np.diag(self.Psi / (n_enr * self.Psi + 1))
        mean = n_enr*enr_U.mean(axis=0)*cov_diag
        logp_no_class = gaussian(np.zeros(tst_U.shape[1]), np.diag(self.Psi+1)).logpdf(tst_U)
        logp_same_class = gaussian(mean, n_enr*cov_diag+1+enr_var).logpdf(tst_U)

        llr = logp_same_class - logp_no_class

        return llr
