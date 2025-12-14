#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:14:25 2023.

@author: yhong.
"""

import numpy as np
import torch
import gpytorch
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

    def __init__(self, points_per_experts: int=200, 
                 partition_type: str='balltree') -> None:
        """ Initialise a product-of-experts Gaussian process model. """
        self.partition_type = partition_type
        self.points_per_experts = points_per_experts
        self.weighting = 'diff_entr'
        self.min_point_in_cluster = 3

    
    def _partition_random(self, x: np.ndarray) -> None:
        """
        Partition training points using random method.

        Args:
            x: The input training points.

        """
        self.partition = []
        ls_all_idx = range(x.shape[0])
        len_all = len(ls_all_idx)
        ls_idx_rd = np.random.Generator( np.array(ls_all_idx), len_all, 
                                     replace=False)
        for i in range(self.M):
            start_idx = i * self.N
            end_idx = start_idx + self.N
            if end_idx <= len_all:
                self.partition.append(np.array(ls_idx_rd[start_idx:end_idx]))
                
    
    def _partition_clustering(self, x: np.ndarray) -> None:
        """
        Partition training points using clustering method.

        Args:
            x: The input training points.

        """
        ls_num = []
        self.partition = []
        centroid, label = scipy.cluster.vq.kmeans2(x, self.M)
        # How many points are in each cluster? counts = np.bincount(label)
        
        num_cluster = 0
        for i in range(self.M):
            ls_data_idx = [j for j, lb in enumerate(label) if lb == i]
            num_data = len(ls_data_idx)
            if (num_data > self.min_point_in_cluster):
                self.partition.append( np.array(ls_data_idx) )
                ls_num.append(len(self.partition[-1]))
                num_cluster = num_cluster + 1
            
        if num_cluster == 0:
            self.partition = []
            self._partition_random(x)
        elif num_cluster < self.M:
            self.M = num_cluster
            
        
    def _partition_balltree(self, x: np.ndarray) -> None:
        """
        Partition training points using balltree method.

        Args:
            x: The input training points.

        """
        self.partition = []
        # For a specified leaf_size, a leaf node is guaranteed to
        # satisfy leaf_size <= n_points <= 2 * leaf_size
        leaf_size = self.N  
        tree = BallTree(x, leaf_size=leaf_size )
        (
            _,
            idx_array,
            node_data,
            node_bounds,
        ) = tree.get_arrays()
        
        ls_join = []
        
        # start from leaf
        num_from_leaf = 0
        num_from_parent = 0
        for node in node_data[::-1]:
            if len(self.partition) >= self.M: 
                break
            idx_start, idx_end, is_leaf, radius = node
            if (is_leaf):
                ls_idx_rd = np.arange(idx_start, idx_start+self.N)
                ls_join = ls_join + ls_idx_rd.tolist()
                ls_idx = idx_array[ls_idx_rd] 
                self.partition.append(np.array(ls_idx))
                num_from_leaf = num_from_leaf + 1
            
        num_from_parent = 0
        idx_start, idx_end, is_leaf, radius = node_data[0]  # parent node
        ls_all_idx = np.arange(idx_start, idx_end).tolist()
        ls_not = [idx for idx in ls_all_idx if idx not in ls_join]
        len_ls_not = len(ls_not)
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
        
        # update self.M, because the number of leaf might be fewer than 
        # the original M
        self.M = len(self.partition)
        self.n_gp = self.M
        
        
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initiate the individual experts and fit their hyperparameters.

        Args: 
            x: dimension: n_train_points x dim_x : Training inputs.
            y: dimension: n_train_points x 1 : Training labels.

        """
        
        # Compute number of experts 
        self.M = int(np.max([int(x.shape[0]) / self.points_per_experts, 1]))
        
        # Compute number of points experts 
        self.N = int(x.shape[0] / self.M)
        
        print("M: ", self.M, ", N: ", self.N)
        
        # If random partition, assign random subsets of data to each expert
        if self.partition_type == 'random':
            self._partition_random(x)
            
        # If clustering partition, assign fit a K_means to the train data and 
        # assign a cluster to each expert
        if self.partition_type == 'clustering':
            self._partition_clustering(x)
                
               
        # Partition training data points using balltree assignment method. 
        if self.partition_type == 'balltree':
            self._partition_balltree(x)
            
            
        # check if there is any left out
        ls_join = []
        for i in range(self.M):
            ls_join = ls_join + self.partition[i].tolist()
        ls_all_idx = range(x.shape[0]) 
        ls_not = [idx for idx in ls_all_idx if idx not in ls_join]
        if len(ls_not) >= self.N:
            self.partition.append(np.array(ls_not))
        elif len(ls_not) > 0:
            self.partition[-1] = np.hstack((self.partition[-1], np.array(ls_not)))  
        
        # update self.M, because the number of leaf might be fewer than 
        # the original M
        self.M = len(self.partition)
        self.n_gp = self.M
        
        self._fit_model(x, y)
        
            
    
    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the individual experts' hyperparameters.

        The hyperparameters are fit by minimizing the sum of negative log
        marginal likelihoods.

        Args: 
            x: dimension: n_train_points x dim_x : Training inputs.
            y: dimension: n_train_points x 1 : Training labels.

        """
        
        train_x = x
        train_fx = y
        likelihoods = []
        models = []
        for i in range(self.M):
            sub_train_x = torch.tensor( train_x[self.partition[i]] ).contiguous()
            sub_train_fx = torch.tensor( 
                        np.ravel(train_fx[self.partition[i]]) ).contiguous()
            
            likelihood_ = gpytorch.likelihoods.GaussianLikelihood()
            
            model_ = GPBase(sub_train_x, sub_train_fx, likelihood_)
            models.append(model_)
            likelihoods.append(model_.likelihood)
            
        self.ls_model = gpytorch.models.IndependentModelList(*models)
        self.ls_likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods)
        
        mll = SumMarginalLogLikelihood(self.ls_likelihood, self.ls_model)
        
        # Find optimal model hyperparameters
        training_iterations = 50
        self.ls_model.train()
        self.ls_likelihood.train()
        
        # Use the Adam optimizer, # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(self.ls_model.parameters(), lr=0.1)  
        
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self.ls_model(*self.ls_model.train_inputs)
            loss = -mll(output, self.ls_model.train_targets)
            loss.backward()
            print(
                f"Iter {i + 1}/{training_iterations} - Loss: {loss.item():.3f}"
            )
            optimizer.step()
         
        # Set into eval mode
        self.ls_model.eval()
        self.ls_likelihood.eval()
        
        self.ls_noise = []
        self.ls_prior_mean = []
        self.ls_prior_var = []
        for idx, model in enumerate(self.ls_model.models):
            # this is noise variance
            gp_noise =  model.likelihood.noise.cpu().detach().numpy().ravel() 
            self.ls_noise.append(gp_noise )
            prior_var = output[idx].variance.detach().numpy()
            self.ls_prior_var.append( prior_var[0].ravel() + gp_noise ) 
            prior_mean = model.mean_module.constant
            self.ls_prior_mean.append( prior_mean.detach().numpy().ravel() )
            
        
    def predict(self, xt_s: np.ndarray) -> np.ndarray:
        
        """
        Predicting aggregated mean and variance for all test points.

        Args: 
            xt_s: dimension: n_test_points x dim_x : Test points.

        Returns: 
            mu_s: dimension: n_test_points x 1 : aggregated predictive mean.
            var_s: dimension: n_test_points x 1 : aggregated predictive variance

        """
        mu_s, var_s = self.gather_predictions(xt_s)
        return self.prediction_aggregation(mu_s, var_s)

    
    
    def gather_predictions(self, xt_s: np.ndarray) -> np.ndarray:
        """
        Gathering the predictive means and variances of all local experts.

        Args: 
            xt_s: dimension: n_test_points x dim_x : Test points.

        Returns: 
            mu_s: dimension: n_expert x n_test_points :
                  predictive mean of each expert at each test point.
            var_s: dimension: n_expert x n_test_points :
                   predictive variance of each expert at each test point.

        """
        # Gather the predictive means and variances of each experts 
        # (a list with the means and variances of each expert - 
        # len(list)=num_experts )
        
        x_cand_torch = torch.tensor(xt_s).contiguous() 
        
        # Set into eval mode
        self.ls_model.eval()
        self.ls_likelihood.eval()
        
        # Make predictions (use the same test points)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # This contains predictions for both outcomes as a list
            predictions = self.ls_likelihood(
                            *[m(x_cand_torch) for m in self.ls_model.models])
        
        ls_gp_y_cand_pred_mean = []
        ls_gp_y_cand_pred_var = []
        for prediction in predictions:
            mean = prediction.mean
            var = prediction.variance #+ gp_noise
            ls_gp_y_cand_pred_mean.append(mean.numpy())
            ls_gp_y_cand_pred_var.append(var.numpy())
        
        #Stacking so that mu_s and var_s are tf tensors of dim 
        # n_expert x n_test_points 
        mu_s = np.stack(ls_gp_y_cand_pred_mean)
        var_s = np.stack(ls_gp_y_cand_pred_var)
        
        
        return mu_s, var_s
    
    
    
    def prediction_aggregation(self, mu_s: np.ndarray, 
                               var_s: np.ndarray, power: int=8) -> np.ndarray:
        """
        Aggregation of predictive means and variances of local experts.

        Args: 
            mu_s: dimension: n_expert x n_test_points :
                  predictive mean of each expert at each test point.
            var_s: dimension: n_expert x n_test_points :
                   predictive variance of each expert at each test point.
            power: dimension : 1x1 : Softmax scaling.

        Returns: 
            mu: dimension: n_test_points x 1 : aggregated predictive mean.
            var: dimension: n_test_points x 1 : aggregated predictive variance.

        """
        
        ls_prior_var = self.ls_prior_var
        ls_noise = self.ls_noise
        
        # Compute individual precisions - dim: n_experts x n_test_points
        prec_s = 1/var_s
            
        # local cc
        ratio_matrix = self.compute_calibration(var_s, ls_noise, 
                                                np.array(ls_prior_var))
        var_s_c = ratio_matrix * var_s
        
        # Compute weight matrix - dim: n_experts x n_test_points
        weight_matrix = self.compute_weights(mu_s, var_s, power, self.weighting, 
                                              prior_var=ls_prior_var, 
                                              )
        
        weight_matrix = self.normalize_weights(weight_matrix)
        
        # Compute individual precisions - dim: n_experts x n_test_points
        prec_s = 1 / var_s_c
        prec = np.sum(weight_matrix * prec_s, axis=0)
        var = 1 / prec
        mu = var * np.sum(weight_matrix * prec_s * mu_s, axis=0) 
        
        min_var = 1e-10
        var_torch = torch.tensor(var)
        var_torch = var_torch.clamp_min(min_var)
        var = var_torch.detach().numpy() 
        
        mu = np.reshape(mu, (-1, 1)) 
        var = np.reshape(var, (-1, 1)) 
        
        return mu, var
    
    
    def compute_calibration(self, m_var: np.ndarray, ls_noise: list, 
                            ls_prior_var: list) -> np.ndarray:
        """ 
        Compute calibration ratio at each local GP.

        Args: 
            m_var: dimension: n_expert x n_test_points :
                   predictive variance of each expert at each test point.
            ls_noise: n_expert x 1 : noise hyperparameter of each expert.
            ls_prior_var: n_expert x 1 : prior variance of each expert.

        Returns: 
            Calibration ratio of ith expert at jth test point.

        """
        
        ls_noise_numpy = np.array(ls_noise)
        m_ig_cand = 0.5 * np.log( 1 + ( m_var / ls_noise_numpy ) )
        
        # normalise all ig to 1
        ls_expert_ig_ub_from = 0.5 * np.log(
                1 + (np.array(ls_prior_var) / np.array(ls_noise))
            )
        
        ig_from = max(ls_expert_ig_ub_from)
        ig_from = max(1, ig_from)
        ig_to = 1
        m_ig_cand_norm = self.normalise_ig_ig(m_ig_cand, ig_from, ig_to)
        
        m_ratio = np.ones(m_ig_cand.shape)
        lb = (np.exp(1) - 1) / np.exp(1) 
          
        for i in range(m_ig_cand.shape[0]):
            ig_cand_cur = m_ig_cand_norm[i,:] * ( 
                lb**(1 / np.median( m_ig_cand_norm[i,:]) ) 
                )
            m_ratio[i,:] = np.exp(-1 * ig_cand_cur) 
        
        return np.where(m_ratio < 1.0, m_ratio, 1.0)
    
    
    def normalise_ig_ig(self, from_ig1: np.ndarray, from_ig2: np.ndarray, 
                        to_ig1: np.ndarray) -> np.ndarray:
        """ 
        Normalise information gain.

        Args: 
            from_ig1: Numerator.
            from_ig2: Denominator.
            to_ig1: Target.

        Returns: 
            Normalised value.

        """
        term1 =  np.exp( from_ig1  ) - 1
        term2 =  np.exp( from_ig2  ) - 1
        term3 =  np.exp( to_ig1  ) - 1
        to_ig2 = (term1 / term2) * term3
        to_ig2 = np.where(to_ig2 > 0, to_ig2, 0)
        
        return np.log(1 + to_ig2)


    def compute_weights(self, mu_s: np.ndarray, var_s: np.ndarray, 
                        power: np.ndarray, weighting: np.ndarray, 
                        prior_var: np.ndarray=None) -> np.ndarray:
        """ 
        Compute unnormalized weight matrix.

        Args:
            mu_s: dimension: n_expert x n_test_points :
                  predictive mean of each expert at each test point.
            var_s: dimension: n_expert x n_test_points :
                   predictive variance of each expert at each test point.
            power: dimension : 1x1 : Softmax scaling.
            weighting: str : weighting method (variance/uniform/diff_entr/no_weights).
            prior_var: dimension: n_expert xx1 : prior variance of expert GPs.

        Returns: 
            weight_matrix: dimension: n_expert x n_test_points :
                           unnormalized weight of ith expert at jth test point.

        """
        prior_var_numpy = np.array(prior_var)
        
        if weighting == 'variance':
            weight_matrix = np.exp(-power * var_s) 
        
        if weighting == 'variance_diff':
            weight_matrix = ( prior_var_numpy ) - ( var_s )
            
        if weighting == 'std_dev':
            weight_matrix = ( np.sqrt(prior_var_numpy) ) - ( np.sqrt(var_s) )
        
        if weighting == 'uniform':
            weight_matrix = np.ones(mu_s.shape, dtype = np.float64) / mu_s.shape[0] 

        if weighting == 'diff_entr':
            weight_matrix = 0.5 * (np.log(prior_var_numpy) - np.log(var_s)) 
            
        if weighting == 'no_weights':
            #weight_matrix = 1
            weight_matrix = np.ones(mu_s.shape, dtype = np.float64)
            

        return weight_matrix   


    
    def normalize_weights(self, weight_matrix: np.ndarray) -> np.ndarray:
        """ 
        Compute unnormalized weight matrix.
        
        Args: 
            weight_matrix: dimension: n_expert x n_test_points : 
                           unnormalized weight of ith expert at jth test point.
       
        Returns: 
            Normalised weight of ith expert at jth test point.

        """
        
        sum_weights = np.sum(weight_matrix, axis=0) 
        return weight_matrix / sum_weights

