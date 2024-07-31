from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # base case
        alpha[:, 0] = np.multiply(self.pi, self.B[:, self.obs_dict[Osequence[0]]])
        
        # recursive formula
        for i in range(1, L):
        	val = np.matmul(alpha[:, i-1], self.A)
        	alpha[:, i] = np.multiply(self.B[:, self.obs_dict[Osequence[i]]], val)
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # base case
        beta[:, L-1] = 1

        # recursive formula
        for i in range(L-2, -1, -1):
            beta[:, i] = np.matmul(self.A, np.multiply(self.B[:, self.obs_dict[Osequence[i+1]]], beta[:, i+1]))
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        alpha = self.forward(Osequence)
        prob = sum(alpha[:, len(Osequence)-1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)

        Oprob = self.sequence_prob(Osequence)
        prob = np.divide(np.multiply(alpha, beta), Oprob)
        ###################################################
        return prob

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        Oprob = self.sequence_prob(Osequence)
        
        for i in range(L-1):
            for j in range(S):
                for k in range(S):
                    prob[k][j][i] = (alpha[k][i] * self.A[k][j] * self.B[j][self.obs_dict[Osequence[i+1]]] * beta[j][i+1])/Oprob
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        prev = -np.ones([S, L])

        obs_indices = []
        for t in Osequence:
            try:
                idx = self.obs_dict[t]
            except:
                idx = len(self.obs_dict)
                self.obs_dict[t] = idx
                self.B = np.hstack((self.B, np.ones((S, 1)) * 1e-6))
            obs_indices.append(idx)

        delta[:, 0] = np.multiply(self.pi, self.B[:, obs_indices[0]])

        AT = self.A.T
        for t in range(1, L):
            delta[:, t] = np.multiply(np.max(np.multiply(AT, delta[:, t - 1]), axis=1), self.B[:, obs_indices[t]])
            prev[:, t] = np.argmax(np.multiply(AT, delta[:, t - 1]), axis=1)
        
        path_ind = [0] * L
        path_ind[-1] = np.argmax(delta[:, -1])
        for t in range(L - 2, -1, -1):
            path_ind[t] = prev[int(path_ind[t + 1]), t + 1]

        key_list = list(self.state_dict.keys())
        val_list = list(self.state_dict.values())

        path = [key_list[val_list.index(i)] for i in path_ind]
        ###################################################
        return path
