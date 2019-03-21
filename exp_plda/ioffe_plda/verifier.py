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

    def multi_sess(self, enr_X, tst_X,  cov_scaling=False, cov_adapt=False):
        enr_U = transform_X_to_U(enr_X, self.inv_A, self.m)
        tst_U = transform_X_to_U(tst_X, self.inv_A, self.m)

        logp_no_class = gaussian(np.zeros(tst_U.shape[1]), np.diag(self.Psi+1)).logpdf(tst_U)
        
        n_enr = len(enr_U)
        cov_diag = np.diag(self.Psi / (n_enr * self.Psi + 1))
        multi_mean = n_enr*cov_diag*enr_U.mean(0)
        
        if cov_scaling:
            cov_diag = n_enr*cov_diag
        
        if cov_adapt:
#             import ipdb
#             ipdb.set_trace()
            enr_var = np.mean(np.square(enr_U - multi_mean), axis=0)
            cov_diag = cov_diag + enr_var
        
        logp_same_class = gaussian(multi_mean, cov_diag+1).logpdf(tst_U)
        llr = logp_same_class - logp_no_class
        
        return np.round(llr.reshape(1, -1), 5)

    def multi_sess_cluster(self, enr_X, tst_X,  cov_scaling=False, cov_adapt=False, n_cluster=3):
        from sklearn.cluster import KMeans
        enr_U = transform_X_to_U(enr_X, self.inv_A, self.m)
        tst_U = transform_X_to_U(tst_X, self.inv_A, self.m)

        logp_no_class = gaussian(np.zeros(tst_U.shape[1]), np.diag(self.Psi+1)).logpdf(tst_U)
        
        kmeans = KMeans(n_clusters=n_cluster).fit(enr_U)
        cluster_labels = kmeans.labels_
        
        llr_list = []
        tot_n_enr = 0
        for l in np.unique(cluster_labels):
            n_enr = np.count_nonzero(cluster_labels==l)
            tot_n_enr += n_enr
            cov_diag = np.diag(self.Psi / (n_enr * self.Psi + 1))
            enr_U_ = enr_U[cluster_labels==l]
            multi_mean = n_enr*cov_diag*enr_U_.mean(0)
            
            if cov_scaling:
                cov_diag = n_enr*cov_diag

            if cov_adapt:
                enr_var = np.sum(np.square(enr_U_ - multi_mean), axis=0)/n_enr
                cov_diag = cov_diag + enr_var
                
            logp_same_class = gaussian(multi_mean, cov_diag+1).logpdf(tst_U)
            llr = logp_same_class - logp_no_class
            llr_list.append(llr*n_enr)
        llr_list = np.array(llr_list)
        llr = np.round(llr_list.sum(0)/tot_n_enr, 5)
        
        return llr.reshape(1, -1)
            
    def multi_sess_select(self, enr_X, tst_X,  
                         cov_scaling=False, cov_adapt=False, score_avg=False,
                         init_n_enr=3, r_adapt=0.5, adapt_T=30):
        enr_U = transform_X_to_U(enr_X, self.inv_A, self.m)
        tst_U = transform_X_to_U(tst_X, self.inv_A, self.m)
        
        init_enr_U = enr_U[:init_n_enr] 
        init_cov = self.Psi / (init_n_enr * self.Psi + 1)
        init_cov_diag = np.diag(init_cov)
        init_multi_mean = init_cov_diag*enr_U[:init_n_enr].sum(axis=0)
        
        adapt_n_enr = len(enr_U) - init_n_enr
        adapt_enr_U = enr_U[init_n_enr:]
        
        # mahalonobis distance
        dist_from_mean = [mahalanobis(U, init_multi_mean, np.linalg.inv(init_cov)) for U in adapt_enr_U]
        # euclidean distance
#         dist_from_mean = np.linalg.norm(adapt_enr_U - init_multi_mean, axis=1)
#         print(np.sort(dist_from_mean))

        sorted_adapt_enr_U = adapt_enr_U[np.argsort(dist_from_mean)]
        act_n_adapt = int(adapt_n_enr * r_adapt)
        n_enr = init_n_enr + act_n_adapt
        enr_U = np.concatenate([init_enr_U, sorted_adapt_enr_U[:act_n_adapt]])
        
#         adapt_idx = np.nonzero(adapt_enr_U[dist_from_mean > adapt_T])[0]
#         n_enr = init_n_enr + len(adapt_idx)
#         enr_U = np.concatenate([init_enr_U, adapt_enr_U[adapt_idx]])
        
        cov_diag = np.diag(self.Psi / (n_enr * self.Psi + 1))
        multi_mean = cov_diag*enr_U.sum(axis=0)
        
        if cov_scaling:
            cov_diag = n_enr*cov_diag
        
        if cov_adapt:
            enr_var = np.sum(np.square(enr_U - multi_mean), axis=0)/(n_enr-1)
            cov_diag = cov_diag + enr_var
            
        logp_no_class = gaussian(np.zeros(tst_U.shape[1]), np.diag(self.Psi+1)).logpdf(tst_U)
        
        if score_avg:
            llr_list = []
            for i in range(len(enr_U)):
                logp_same_class = gaussian(enr_U[i], cov_diag+1).logpdf(tst_U)
                llr = logp_same_class - logp_no_class
                llr_list.append(llr)
            llr = np.mean(llr_list, axis=0) 
        else:
            logp_same_class = gaussian(multi_mean, cov_diag+1).logpdf(tst_U)
            llr = logp_same_class - logp_no_class

        return np.round(llr.reshape(1, -1), 5)
    
    def score_avg(self, enr_X, tst_X):
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

        return np.round(llr_list, 5)
    
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
