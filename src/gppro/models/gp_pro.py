#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:14:25 2023.

@author: yhong.
"""

import numpy as np
import torch
import gpytorch
#from gpytorch.constraints.constraints import Interval
#from gpytorch.distributions import MultivariateNormal
#from gpytorch.kernels import MaternKernel, ScaleKernel
#from gpytorch.likelihoods import GaussianLikelihood
#from gpytorch.means import ConstantMean, ZeroMean
#from gpytorch.mlls import ExactMarginalLogLikelihood
#from gpytorch.models import ExactGP
import scipy
from sklearn.neighbors import BallTree
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from .gp_base import GPBase


class GPPro:
    """
    A product-of-experts Gaussian process model using GPBase.

    Args:
        points_per_experts: Maximum number of training points allocated to each
                            expert.
        partition_type: Method to partition the training points:
                        random / clustering / balltree.

    """

    def __init__(self, points_per_experts: int=200, partition_type: str='balltree') -> None:
        """
        Initialise a product-of-experts Gaussian process model.
        
        """
        self.partition_type = partition_type
        self.points_per_experts = points_per_experts
        self.weighting = 'diff_entr'

    
    def _partition_random(self, x: np.ndarray) -> None:
        """
        Partition training points using random method.

        Args:
            x: The input training points.

        """
        self.partition = []
        ls_all_idx = range(x.shape[0])
        len_all = len(ls_all_idx)
        ls_idx_rd = np.random.choice( np.array(ls_all_idx), len_all, 
                                     replace=False)
        for i in range(self.M):
            start_idx = i * self.N
            end_idx = start_idx + self.N
            if end_idx <= len_all:
                self.partition.append(np.array(ls_idx_rd[start_idx:end_idx]))
                
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        
        """ 
        Initiate the individual experts and fit their shared hyperparameters by
        minimizing the sum of negative log marginal likelihoods
        
        Inputs : 
                -- X, dimension: n_train_points x dim_x : Training inputs
                -- Y, dimension: n_train_points x 1 : Training Labels
                
        """
        
        # Compute number of experts 
        self.M = int(np.max([int(X.shape[0]) / self.points_per_experts, 1]))
        
        # Compute number of points experts 
        self.N = int(X.shape[0] / self.M)
        
        print("M: ", self.M, ", N: ", self.N)
        
        # If random partition, assign random subsets of data to each expert
        if self.partition_type == 'random':
            self._partition_random(X)
            #self.partition = np.random.choice(X.shape[0], size=(self.M, self.N), replace=False)
        
        # If clustering partition, assign fit a K_means to the train data and assign a cluster to each expert
        if self.partition_type == 'clustering':
            
            ls_num = []
            
            self.partition = []
            centroid, label = scipy.cluster.vq.kmeans2(X, self.M)
            # How many points are in each cluster?
            #counts = np.bincount(label)
            
            #total_len = 0
            num_cluster = 0
            for i in range(self.M):
                ls_data_idx = [j for j, lb in enumerate(label) if lb == i]
                num_data = len(ls_data_idx)
                if (num_data > 3):
                    self.partition.append( np.array(ls_data_idx) )
                    ls_num.append(len(self.partition[-1]))
                    num_cluster = num_cluster + 1
                
            if num_cluster == 0:#(num_data < 5):
                print(" !!! clustering failed !!! revert to random partition ")
                self.partition = []
                self._partition_random(X)
                #self.partition = np.random.choice(X.shape[0], size=(self.M, self.N), replace=False)
            elif num_cluster < self.M:
                self.M = num_cluster
                print("self.M updated: ", self.M )
                    
            
            print("clustering: ls_num: ", ls_num)
            
        
                
        """ Partition training data points using balltree assignment method. """
        
        
        if self.partition_type == 'balltree':
            
            self.partition = []
            # For a specified leaf_size, a leaf node is guaranteed to satisfy leaf_size <= n_points <= 2 * leaf_size
           
            leaf_size = self.N  #math.ceil(self.N * 0.8)
            #print("leaf_size: ", leaf_size)
            tree = BallTree(X, leaf_size=leaf_size )
            
            (
                _,
                idx_array,
                node_data,
                node_bounds,
            ) = tree.get_arrays()
            
            #node_num = len(node_data)
            
            ls_join = []
            
            # start from leaf
            num_from_leaf = 0
            num_from_parent = 0
            #for idx, node in enumerate(node_data):
            for node in node_data[::-1]:
                #if idx >= self.M: break
                if len(self.partition) >= self.M: break
                idx_start, idx_end, is_leaf, radius = node
                #ls_idx_rd = [randint(idx_start, idx_end) for p in range(self.points_per_experts)]
                #ls_idx_rd = np.random.choice( np.arange(idx_start, idx_end), self.points_per_experts, replace=False)
                if (is_leaf):
                    #ls_idx_rd = np.random.choice( np.arange(idx_start, idx_end), self.N, replace=False)
                    ls_idx_rd = np.arange(idx_start, idx_start+self.N)
                    #ls_idx_rd = np.arange(idx_end-self.N, idx_end)
                    ls_join = ls_join + ls_idx_rd.tolist()
                    ls_idx = idx_array[ls_idx_rd] 
                    #print("ls_idx: ", ls_idx)
                    self.partition.append(np.array(ls_idx))
                    num_from_leaf = num_from_leaf + 1
                
            num_from_parent = 0
            idx_start, idx_end, is_leaf, radius = node_data[0]  # parent node
            #print("idx_start: ", idx_start, ", idx_end: ", idx_end)
            ls_all_idx = np.arange(idx_start, idx_end).tolist()
            ls_not = [idx for idx in ls_all_idx if idx not in ls_join]
            len_ls_not = len(ls_not)
            #print("len ls_not: ", len_ls_not)
            #ls_idx_rd = np.random.choice( np.array(ls_not), len_ls_not, replace=False)
            i = 0
            while len(self.partition) < self.M:
                start_idx = i * self.N
                end_idx = start_idx + self.N
                if end_idx <= len_ls_not:
                    ls_idx_rd_ = np.array(ls_not[start_idx:end_idx])
                    ls_join = ls_join + ls_idx_rd_.tolist()
                    ls_idx = idx_array[ls_idx_rd_] 
                    self.partition.append(np.array(ls_idx))
                    num_from_parent = num_from_parent + 1
                else:
                    break
                i = i + 1
            #print("num from leaf: ", num_from_leaf, ", num_from_parent: ", num_from_parent)
            
            
            
            # update self.M, because the number of leaf might be fewer than the original M
            self.M = len(self.partition)
            #print("self.M updated: ", self.M )
            self.n_gp = self.M
            
        # check if there is any left out
        ls_join = []
        for i in range(self.M):
            ls_join = ls_join + self.partition[i].tolist()
        ls_all_idx = range(X.shape[0]) #np.arange(idx_start, idx_end).tolist()
        ls_not = [idx for idx in ls_all_idx if idx not in ls_join]
        #print("!!! ls_not len: ", len(ls_not))
        if len(ls_not) >= self.N:
            self.partition.append(np.array(ls_not))
        elif len(ls_not) > 0:
            self.partition[-1] = np.hstack((self.partition[-1], np.array(ls_not)))  
        
        # update self.M, because the number of leaf might be fewer than the original M
        self.M = len(self.partition)
        #print("self.M updated: ", self.M )
        self.n_gp = self.M
        
        
        ls_size_ = [] 
        for i in range(self.M):
            ls_size_.append(len(self.partition[i]))
        #print("ls_size_: ", ls_size_)
        
        train_X = X
        train_fX = Y
        likelihoods = []
        models = []
        for i in range(self.M):
            sub_train_X = torch.tensor( train_X[self.partition[i]] ).contiguous()
            sub_train_fX = torch.tensor( np.ravel(train_fX[self.partition[i]]) ).contiguous()
            
            #noise_constraint = Interval(5e-4, 0.2)
            likelihood_ = gpytorch.likelihoods.GaussianLikelihood()#noise_constraint=noise_constraint)
            
            model_ = GPBase(sub_train_X, sub_train_fX, likelihood_)
            models.append(model_)
            likelihoods.append(model_.likelihood)
            
        self.ls_model = gpytorch.models.IndependentModelList(*models)
        self.ls_likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods)
        
        mll = SumMarginalLogLikelihood(self.ls_likelihood, self.ls_model)
        
        # Find optimal model hyperparameters
        training_iterations = 50
        self.ls_model.train()
        self.ls_likelihood.train()
        
        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.ls_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self.ls_model(*self.ls_model.train_inputs)
            #print("output shape: ", len(output))
            #print("self.ls_model.train_targets shape: ", len(self.ls_model.train_targets))
            #print("output: ", output)
            #print("self.ls_model.train_targets: ", self.ls_model.train_targets)
            loss = -mll(output, self.ls_model.train_targets)
            #print("loss len: ", len(loss))
            #loss = sum(loss)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
            
        
        # Set into eval mode
        self.ls_model.eval()
        self.ls_likelihood.eval()
        
        self.ls_noise = []
        self.ls_prior_mean = []
        self.ls_prior_var = []
        #for model in self.ls_model.models:
        for idx, model in enumerate(self.ls_model.models):
            #x_0_torch = torch.tensor(model.train_inputs[0])
            #prior_var = model.covar_module(x_0_torch, x_0_torch)
            
            #ls_prior_var.append( prior_var[0,0].detach().numpy().ravel() )
            gp_noise =  model.likelihood.noise.cpu().detach().numpy().ravel() # this is noise variance
            self.ls_noise.append(gp_noise )
            prior_var = output[idx].variance.detach().numpy()
            self.ls_prior_var.append( prior_var[0].ravel() + gp_noise ) # + 1e-6)
            
            #self.ls_prior_var.append( prior_var[0,0].detach().numpy().ravel() + gp_noise ) # + 1e-6)
            
            #print("prior_var: ", prior_var[0,0].detach().numpy().ravel(), ", var: ", var)
            prior_mean = model.mean_module.constant
            self.ls_prior_mean.append( prior_mean.detach().numpy().ravel() )
            
            #del x_0_torch
    
    
    def predict(self, xt_s):
        """Predicting aggregated mean and variance for all test points
        
        Inputs : 
                -- xt_s, dimension: n_test_points x dim_x : Test points 
                -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
                -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
                
        Output : 
                -- mu, dimension: n_test_points x 1 : aggregated predictive mean
                -- var, dimension: n_test_points x 1 : aggregated predictive variance
                
        """
        mu_s, var_s = self.gather_predictions(xt_s)
        return self.prediction_aggregation(xt_s, mu_s, var_s) #, self.model, power=self.power, weighting=self.weighting)

    
    
    def gather_predictions(self, xt_s: np.ndarray):
        """Gathering the predictive means and variances of all local experts at all test points
        
        Inputs : 
                -- xt_s, dimension: n_test_points x dim_x : Test points 
                
        Output : 
                -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
                -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
        """
        
        # Gather the predictive means and variances of each experts 
        #                  (a list with the means and variances of each expert - len(list)=num_experts )
        
        
        #print("xt_s: ", xt_s)
        X_cand_torch = torch.tensor(xt_s).contiguous() #.to(device=self.device, dtype=self.dtype)
        #ls_x_cand = []
        #for i in range(self.M):
        #    ls_x_cand.append(X_cand_torch)
        
        # Set into eval mode
        self.ls_model.eval()
        self.ls_likelihood.eval()
        
        
        # Make predictions (use the same test points)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #test_x = torch.linspace(0, 1, 51)
            # This contains predictions for both outcomes as a list
            predictions = self.ls_likelihood(*[m(X_cand_torch) for m in self.ls_model.models])
            #predictions = [m(X_cand_torch) for m in self.ls_model.models]
            #predictions = likelihood(*model(test_x, test_x))
        
        ls_gp_y_cand_pred_mean = []
        ls_gp_y_cand_pred_var = []
        for submodel, prediction in zip(self.ls_model.models, predictions):
            #gp_noise =  submodel.likelihood.noise.cpu().detach().numpy().ravel() # this is noise variance
            mean = prediction.mean
            var = prediction.variance #+ gp_noise
            ls_gp_y_cand_pred_mean.append(mean.numpy())
            ls_gp_y_cand_pred_var.append(var.numpy())
            
            
        
        #Stacking so that mu_s and var_s are tf tensors of dim n_expert x n_test_points 
        mu_s = np.stack(ls_gp_y_cand_pred_mean)#[:, :, 0] #tf.stack(mu_s)[:, :, 0]
        var_s = np.stack(ls_gp_y_cand_pred_var)#[:, :, 0]
        
        
        return mu_s, var_s
    
    
    
    def prediction_aggregation(self, xt_s, mu_s, var_s, power=8): #, method='PoE', weighting='wass', power=8):

        """ Aggregation of predictive means and variances of local experts
        
        Inputs : 
                -- xt_s, dimension: n_test_points x dim_x : Test points 
                -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
                -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
                -- method, str : aggregation method (PoE/gPoE/BCM/rBCM/bar)
                -- weighting, str : weighting method (variance/wass/uniform/diff_entr/no_weights)
                -- power, dimension : 1x1 : Softmax scaling
                
        Output : 
                -- mu, dimension: n_test_points x 1 : aggregated predictive mean
                -- var, dimension: n_test_points x 1 : aggregated predictive variance
        """
        # Compute prior variance (shared between all experts)
        #prior_var = self.kern(xt_s[0], xt_s[0])
        
        
        ls_prior_mean = self.ls_prior_mean
        ls_prior_var = self.ls_prior_var
        ls_noise = self.ls_noise
        
        # Compute individual precisions - dim: n_experts x n_test_points
        prec_s = 1/var_s
        
            
        # local cc
        ratio_matrix = self.compute_calibration(var_s, ls_noise, np.array(ls_prior_var)) #, ls_ig_train)
        #print("ratio_matrix: ", ratio_matrix)
        var_s_c = ratio_matrix * var_s
        
        
        # Compute weight matrix - dim: n_experts x n_test_points
        weight_matrix = self.compute_weights(mu_s, var_s, power, self.weighting, 
                                              prior_mean=ls_prior_mean, prior_var=ls_prior_var, 
                                              ratio_cc=ratio_matrix)
          
        
        weight_matrix = self.normalize_weights(weight_matrix)
        
        # Compute individual precisions - dim: n_experts x n_test_points
        prec_s = 1 / var_s_c
        #print("prec_s.shape: ", prec_s.shape)
        prec = np.sum(weight_matrix * prec_s, axis=0)
        var = 1 / prec
        mu = var * np.sum(weight_matrix * prec_s * mu_s, axis=0) 
            
        
        min_var = 1e-10
        var_torch = torch.tensor(var)
        var_torch = var_torch.clamp_min(min_var)
        var = var_torch.detach().numpy() 
        
        
        mu = np.reshape(mu, (-1, 1)) #tf.reshape(mu, (-1, 1))
        var = np.reshape(var, (-1, 1)) #tf.reshape(var, (-1, 1))
        
        return mu, var
    
    
    def compute_calibration(self, m_var, ls_noise, ls_prior_var, ls_ig_train=None):
        """ Compute reduced variances at each local GP by applying local coservativeness control. 
        Inputs : 
                -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
                -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
                -- ls_noise: n_expert x 1 : noise hyperparameter of each expert
                -- ls_ig_train_max: n_expert x 1 
                
        Output : 
                -- ratio_matrix, dimension: n_expert x n_test_points : conservativeness ratio of ith expert at jth test point
        
        
        """
        
        ls_noise_numpy = np.array(ls_noise)
        m_ig_cand = 0.5 * np.log( 1 + ( m_var / ls_noise_numpy ) )
        #m_ig_cand_lb = m_ig_cand * ( (np.exp(1) - 1) / np.exp(1) )
        #print("\n m_ig_cand: ", m_ig_cand)
        
        # normalise all ig to 1
        ls_expert_ig_ub_from = 0.5 * np.log( 1 + ( np.array(ls_prior_var) / np.array(ls_noise) ) )
        #print("\n ls_expert_ig_ub_from: ", ls_expert_ig_ub_from)
        
        ig_from = max(ls_expert_ig_ub_from)
        ig_from = max(1, ig_from)
        ig_to = 1
        
        
        m_ig_cand_norm = self.normalise_ig_ig(m_ig_cand, ig_from, ig_to)
        
        
        m_ratio = np.ones(m_ig_cand.shape)
        
        lb = (np.exp(1) - 1) / np.exp(1) # 2.0
          
        for i in range(m_ig_cand.shape[0]):
            
            ##ig_cand_cur = m_ig_cand_norm[i,:] *  lb
            ##ig_cand_cur = m_ig_cand_norm[i,:] * ( lb**(1 / np.mean( m_ig_cand_norm[i,:]) ) )
            ig_cand_cur = m_ig_cand_norm[i,:] * ( lb**(1 / np.median( m_ig_cand_norm[i,:]) ) )
            #ig_cand_cur = m_ig_cand_norm[i,:] * ig_cand_lb_norm[i,:]
            
            m_ratio[i,:] = np.exp(-1 * ig_cand_cur) 
            
        
        m_ratio = np.where(m_ratio < 1.0, m_ratio, 1.0)
        #print("\n m_ratio: ", m_ratio)
        
        return m_ratio
    
    
    def normalise_ig_ig(self, from_ig2, from_ig1, to_ig1):
        term1 =  np.exp( from_ig2  ) - 1
        term2 =  np.exp( from_ig1  ) - 1
        term3 =  np.exp( to_ig1  ) - 1
        to_ig2 = (term1 / term2) * term3
        to_ig2 = np.where(to_ig2 > 0, to_ig2, 0)
        to_ig2 = np.log(1 + to_ig2)
        
        #to_ig2 = (from_ig2 / from_ig1) * to_ig1
        return to_ig2


    def compute_weights(self, mu_s, var_s, power, weighting, prior_mean=None, prior_var=None, ratio_cc=None, softmax=False):
        
        """ Compute unnormalized weight matrix
        Inputs : 
                -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
                -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
                -- power, dimension : 1x1 : Softmax scaling
                -- weighting, str : weighting method (variance/wass/uniform/diff_entr/no_weights)
                -- prior_var, dimension: n_expert xx1 : shared prior variance of expert GPs
                -- soft_max_wass : logical : whether to use softmax scaling or fraction scaling
                
        Output : 
                -- weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of ith expert at jth test point
        """
        #print("prior_var: ", prior_var)
        #prior_mean_numpy = np.array(prior_mean)
        prior_var_numpy = np.array(prior_var)
        #prior_var_max = np.max(prior_var_numpy)

        
        if weighting == 'variance':
            weight_matrix = np.exp(-power * var_s) 
            
        if weighting == 'variance_t1':
            weight_matrix = np.exp(-1 * var_s) 
            
        if weighting == 'variance_cc':
            weight_matrix = np.exp(-ratio_cc * var_s) 
        
        if weighting == 'variance_diff':
            weight_matrix = ( prior_var_numpy ) - ( var_s )
            
        if weighting == 'std_dev':
            weight_matrix = ( np.sqrt(prior_var_numpy) ) - ( np.sqrt(var_s) )
        
        if weighting == 'uniform':
            weight_matrix = np.ones(mu_s.shape, dtype = np.float64) / mu_s.shape[0] #tf.ones(mu_s.shape, dtype = tf.float64) / mu_s.shape[0]

        if weighting == 'diff_entr':
            weight_matrix = 0.5 * (np.log(prior_var_numpy) - np.log(var_s)) 
            
        if weighting == 'no_weights':
            #weight_matrix = 1
            weight_matrix = np.ones(mu_s.shape, dtype = np.float64)
            

        return weight_matrix   


    
    def normalize_weights(self, weight_matrix):
        """ Compute unnormalized weight matrix
        Inputs : 
                -- weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of ith expert at jth test point
                
                
        Output : 
                -- weight_matrix, dimension: n_expert x n_test_points : normalized weight of ith expert at jth test point
        """
        
        sum_weights = np.sum(weight_matrix, axis=0) #tf.reduce_sum(weight_matrix, axis=0)
        weight_matrix = weight_matrix / sum_weights
        
        return weight_matrix

