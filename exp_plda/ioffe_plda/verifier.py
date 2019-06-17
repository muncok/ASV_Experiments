import numpy as np
from scipy.stats import multivariate_normal as gaussian
from scipy.spatial.distance import mahalanobis
from .optimizer import optimize_maximum_likelihood
from .optimizer import calc_scatter_matrices
from  scipy.misc import logsumexp

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

    def multi_sess(self, enr_X, tst_X, weights=None, n_enr=None, cov_scaling=False, cov_adapt=False):
        if enr_X.ndim == 1:
            enr_X = enr_X.reshape(1, -1)
        if tst_X.ndim == 1:
            tst_X = tst_X.reshape(1, -1)
            
        enr_U = transform_X_to_U(enr_X, self.inv_A, self.m)
        tst_U = transform_X_to_U(tst_X, self.inv_A, self.m)

        logp_no_class = gaussian(np.zeros(tst_U.shape[1]), np.diag(self.Psi+1)).logpdf(tst_U)
        
        
        if n_enr is None:
            n_enr = len(enr_U)
        cov_diag = np.diag(self.Psi / (n_enr * self.Psi + 1))
        
        if weights is None:
            multi_mean = n_enr*cov_diag*enr_U.mean(0)
        else:
            enr_mean = (weights.reshape(-1, 1) * enr_U).sum(0) / weights.sum()
            multi_mean = n_enr * cov_diag * enr_mean
        
        if cov_scaling:
            cov_diag = n_enr*cov_diag
        
        if cov_adapt:
            enr_var = np.mean(np.square(enr_U - multi_mean), axis=0)
            cov_diag = cov_diag + enr_var
        
        logp_same_class = gaussian(multi_mean, cov_diag+1).logpdf(tst_U)
        llr = logp_same_class - logp_no_class
        
        return np.round(llr.reshape(1, -1), 5)
    
    def score_avg(self, enr_X, tst_X):
        if enr_X.ndim == 1:
            enr_X = enr_X.reshape(1, -1)
        if tst_X.ndim == 1:
            tst_X = tst_X.reshape(1, -1)
            
        enr_U = transform_X_to_U(enr_X, self.inv_A, self.m)
        tst_U = transform_X_to_U(tst_X, self.inv_A, self.m)

        # shared across enr_U
        logp_no_class = gaussian(np.zeros(tst_U.shape[1]), np.diag(self.Psi+1)).logpdf(tst_U)
        
        llr_list = []
        cov_diag = np.diag(self.Psi / (self.Psi + 1))
        for i in range(len(enr_U)):
            logp_same_class = gaussian(enr_U[i]*cov_diag, cov_diag+1).logpdf(tst_U)
            llr = logp_same_class - logp_no_class
            llr_list.append(llr)
        
        llr_arr = np.round(llr_list, 5)
        if llr_arr.ndim == 1:
            llr_arr = llr_arr.reshape(1, -1)

        return llr_arr
    
    def vector_avg(self, enr_X, tst_X):
        # averaging vertors befor scoring
        enr_X = enr_X.mean(0, keepdims=True)
        enr_U = transform_X_to_U(enr_X, self.inv_A, self.m)
        tst_U = transform_X_to_U(tst_X, self.inv_A, self.m)

        logp_no_class = gaussian(np.zeros(tst_U.shape[1]), np.diag(self.Psi+1)).logpdf(tst_U)
        
        cov_diag = np.diag(self.Psi / (self.Psi + 1))
        mean = enr_U[0]*cov_diag
        logp_same_class = gaussian(mean, cov_diag+1).logpdf(tst_U)

        llr = logp_same_class - logp_no_class

        return np.round(llr.reshape(1, -1), 5)
