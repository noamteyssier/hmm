#!/usr/bin/env python3

import numpy as np
import random
import argparse

class HMM:
    def __init__(self, transitions, emissions, initial, h_states, o_states):
        self.a = transitions
        self.b = emissions
        self.pi = initial
        self.h_states = h_states
        self.o_states = o_states

        self.N = len(h_states)
        self.M = len(o_states)

        self.alpha = None
        self.beta = None
        self.beta_v = None

    def index_obs(self, o_symbol):
        """return index of observed symbol in class"""
        return self.o_states.index(o_symbol)
    def forward_backward(self, obs):
        self.forward(obs)
        self.backward(obs)
        return self.result_forward()
    def forward(self, obs):
        """calculate forward probabilities"""
        T = len(obs)

        # alpha[t][i]
        self.alpha = np.array([
            [0.0 for _ in range(self.N)] for _ in range(T)
        ])

        # alpha_1_i = pi_i * emission_i_O1
        # alpha_t+1_j = sum(alpha_t_i * transmission_i_j) * emission_b_Ot+1
        for t in range(T - 1):
            o = self.index_obs(obs[t])
            ot1 = self.index_obs(obs[t+1])

            for i in range(self.N):

                # initialization
                if t == 0 :
                    self.alpha[t][i] = np.product(
                        [self.pi[i], self.b[i][o]]
                    )

                for j in range(self.N):
                    self.alpha[t+1][j] = np.product(
                        [
                            self.alpha[t][i] * self.a[i][j],
                            self.b[j][ot1]
                        ]
                    )
    def backward(self, obs):
        """calculate backwards probabilities"""
        T = len(obs)

        # indexed beta[t][i]
        self.beta = np.array([
            [0.0 for _ in range(self.N)] for _ in range(T)
        ])

        # indexed beta_v[t][j][v]
        self.beta_v = np.array([
            [[{v : 0} for v in self.h_states] for _ in range(self.N)] for _ in range(T)
        ])


        for t in range(T-1, -1, -1):
            for i in range(self.N):

                # initialization
                if t == (T-1) :
                    self.beta[t][i] = 1

                # induction
                else:
                    ot1 = self.index_obs(obs[t+1])
                    for j in range(self.N):
                        beta_tij = np.product([
                            self.a[i][j],
                            self.b[j][ot1],
                            self.beta[t+1][j]
                        ])
                        self.beta[t][i] += beta_tij
                        self.beta_v[t][j][ot1] = beta_tij
    def result_forward(self):
        """sum final nodes of forward trellis"""
        return sum(self.alpha[-1][i] for i in range(self.N))
    def baum_welch(self, obs, iter = 1000, overwrite=False):
        T = len(obs)

        # calculate initial alpha and beta
        self.forward_backward(obs)
        self.get_epsilon(obs)
        self.get_gamma(obs)

        for i in range(iter):
            self.get_epsilon(obs)
            self.get_gamma(obs)
            new_a = self.update_a(T, overwrite=overwrite)
            new_b = self.update_b(obs, overwrite=overwrite)
            new_pi = self.update_pi(overwrite=overwrite)

            # recalculate alpha and beta
            self.forward_backward(obs)
            self.get_epsilon(obs)
            self.get_gamma(obs)

        return [new_a, new_b, new_pi]
    def get_epsilon(self, obs):
        """calculate epsilon matrix"""
        T = len(obs)

        # index epsilon[t][i][j]
        epsilon= np.array([
            [[0.0 for _ in range(self.N)]
                for _ in range(self.N)]
                    for _ in range(T)
        ])

        epsilon_v = np.array([
            [[[{v_m} for v_m in range(self.M)]
                for _ in range(self.N)]
                for _ in range(self.N)]
                for _ in range(T)
        ])

        # calculate epsilon matrix
        for t in range(T-1):
            ot1 = self.index_obs(obs[t+1])
            for i in range(self.N):
                for j in range(self.N):
                    epsilon[t][i][j] = np.product([
                        self.alpha[t][i],
                        self.a[i][j],
                        self.b[j][ot1],
                        self.beta[t+1][j]
                    ])

        # calculate denominator
        for t in range(T-1):
            total = sum(sum(epsilon[t]))
            epsilon[t] = epsilon[t] / total


        self.epsilon = epsilon.copy()
        return epsilon
    def get_gamma(self, obs):
        """calculate gamma matrix"""
        T = len(obs)

        gamma = np.array([
            [0.0 for _ in range(self.N)] for _ in range(T)
        ])

        for t in range(T):
            ot = self.index_obs(obs[t])
            for i in range(self.N):
                gamma[t][i] = sum(
                    [self.epsilon[t][i][j] for j in range(self.N)]
                )

        self.gamma = gamma.copy()
        return gamma
    def update_a(self, T, overwrite=False):
        """update transmission matrix """
        new_a = self.a.copy()

        for i in range(self.N):
            for j in range(self.N):
                new_a[i][j] = \
                    sum([self.epsilon[t][i][j] for t in range(T)]) / \
                    sum([self.gamma[t][i] for t in range(T)])

        if overwrite == True:
            self.a = new_a

        return new_a
    def update_b(self, obs, overwrite=False):
        """update emissions"""
        T = len(obs)

        new_b = self.b.copy()

        for k in self.o_states:
            k_timepoints = [i for i,x in enumerate(obs) if x == k]
            for i in range(self.N):
                num = sum([self.gamma[t][i] for t in range(T) if t in k_timepoints])
                denom = sum([self.gamma[t][i] for t in range(T)])
                new_b[i][self.index_obs(k)] = num/denom

        if overwrite == True:
            self.b = new_b

        return new_b
    def update_pi(self, overwrite=False):
        """update initial states"""
        new_pi = [i for i in self.gamma[0]]

        if overwrite == True:
            self.pi = new_pi

        return new_pi
    def train_multiple(self, X):

        new_a = np.array([
            [{'num' : 0, 'denom' : 0 } for _ in range(self.N)] for _ in range(self.N)
        ])

        new_b = np.array([
            [{'num' : 0, 'denom' : 0 } for _ in range(self.M)] for _ in range(self.N)
        ])

        for x in X:
            self.forward_backward(x)
            self.update_a_multiple(x, new_a)
            self.update_b_multiple(x, new_b)
            self.update_multiple(new_a, new_b)
    def update_a_multiple(self, obs, new_a):
        T = len(obs)

        for i in range(self.N):

            denom = np.sum(
                [self.alpha[t][i] * self.beta[t][i] for t in range(T-1)]
            )

            for j in range(self.N):
                new_a[i][j]['num'] += np.sum(
                    [
                        np.product([
                            self.alpha[t][i],
                            self.a[i][j],
                            self.b[j][self.index_obs(obs[t+1])],
                            self.beta[t+1][j]
                        ]) for t in range(T-1)
                    ]
                )
                new_a[i][j]['denom'] += denom

        return new_a
    def update_b_multiple(self, obs, new_b):
        T = len(obs)

        for k in range(self.M):
            vk_indices = [i for i,v in enumerate(obs) if self.index_obs(v) == k]
            for i in range(self.N):
                new_b[i][k]['num'] += np.sum([
                    [self.alpha[t][i] * self.beta[t][i]] for t in range(T-1) if t in vk_indices
                ])
                new_b[i][k]['denom'] += np.sum([
                    [self.alpha[t][i] * self.beta[t][i]] for t in range(T-1)
                ])
        return new_b

            # break
    def update_multiple(self, new_a, new_b):

        for i in range(self.N):
            for j in range(self.N):
                self.a[i][j] = new_a[i][j]['num'] / new_a[i][j]['denom']

        for i in range(self.N):
            for k in range(self.M):
                self.b[i][k] = new_b[i][k]['num'] / new_b[i][k]['denom']
    def viterbi(self, obs):
        pass










def generate_random_A(states):
    '''generate a random transition matrix to optimize'''
    n = len(states)
    random_a = np.array([
        [random.random() for i in range(n)] for j in range(n)
    ])
    norm_a = random_a.copy()
    for i in range(n):
        total = random_a[i].sum()
        for j in range(n):
            random_a[i][j] = random_a[i][j] / total

    return random_a
def generate_random_B(states, obs):
    '''generate a random emission matrix to optimize'''
    n = len(states)
    k = len(obs)

    random_b = np.array([
        [random.random() for v in range(k)] for i in range(n)
    ])

    norm_b = random_b.copy()

    for i in range(n):
        total = random_b[i].sum()
        for v in range(k):
            norm_b[i][v] = norm_b[i][v] / total

    return norm_b
def generate_random_pi(states, equal_odds=False):
    '''create a random vector of initial states'''
    n = len(states)

    if not equal_odds:
        pi = np.array([
            random.random() for i in range(n)
        ])
    else:
        val = random.random()
        pi = np.array([
            val for _ in range(n)
        ])

    norm_pi = [p/sum(pi) for p in pi]

    return norm_pi
def read_faust(ifn):
    f = open(ifn, 'r')
    for line in f:
        yield [k for k in next(f).strip('\n')]




def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', help = 'faust filename', required=True)
    p.add_argument('-n', '--num_iter', help = 'num of iterations [10]', required = False)
    args = p.parse_args()

    states = ['i', 'u']
    obs = ['T', 'F']

    transitions = generate_random_A(states)
    emissions = generate_random_B(states, obs)
    initial = generate_random_pi(states, equal_odds = False)

    observations = [x for x in read_faust(args.input)]

    hmm = HMM(transitions, emissions, initial, states, obs)

    obs = ['T', 'T', 'F', 'T', 'F', 'F', 'F', 'F', 'T', 'T', 'T']

    hmm.train_multiple(observations)

    hmm.viterbi(obs)








if __name__ == '__main__':
    random.seed(41)
    main()
