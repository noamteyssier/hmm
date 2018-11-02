#!/usr/bin/env python3

import random
import numpy as np
import sys
import argparse


class HMM:
    def __init__(self, states, obs, A, B, pi):
        self.states = states
        self.obs = obs

        self.N = len(self.states)
        self.M = len(self.obs)

        self.A = A      # index[i][j]
        self.B = B      # index[i][o]
        self.pi = pi

        self.f_nodes = None         # the forward likelihoods of each node (NxT)
        self.r_nodes = None         # the reverse likelihoods of each node (NxT)
        self.g_mat = None           # the multiplied forward and reverse likelihoods (NxT)
        self.norm_g_mat = None      # the normalized gamma matrix of probabilities across states (NxT)
    def forward_backward(self, observation, normalize = False):
        self.forward(observation)
        self.backward(observation)

        if normalize:
            self.normalize_alpha()
            self.normalize_beta()

        self.gamma_matrix(len(observation))
        return self.argmax_gamma(len(observation))
    def forward(self, observation):
        '''forward propagation along trellis diagram'''

        # representation of trellis diagram (x-axis = t, y-axis = probability)
        #  indexed nodes[i][t]
        self.f_nodes = np.array(
            [[0.0 for _ in range(len(observation))] for _ in range(self.N)]
        )

        # iterate across timepoint~state
        for t, obs in enumerate(observation):
            for i in range(self.N):

                # initialization
                if t == 0:
                    t_m1 = self.pi[i]
                # induction
                else:
                    t_m1 = sum(self.f_nodes[s][t-1] * self.A[s][i] for s in range(self.N))

                # product of previous timepoint and current emission
                # common term of induction and initialization
                self.f_nodes[i][t] = t_m1 * self.B[i][self.index_obs(obs)]

        # termination
        self.p_forward = sum(sum(self.f_nodes))

        return self.f_nodes
    def backward(self, observation):
        '''backward propagation along trellis diagram'''

        self.b_nodes = np.array(
            [[0.0 for _ in range(len(observation))] for _ in range(self.N)]
        )

        # iterate through observation list backwards and staggered by one
        for t, obs in enumerate(reversed(observation[1:] + [None])):
            for i in range(self.N):

                # initialization
                if t == 0:
                    self.b_nodes[i][-1-t] = 1

                # induction
                else:
                    self.b_nodes[i][-1-t] = sum(
                            self.A[i][s] * \
                            self.B[s][self.index_obs(obs)] * \
                            self.b_nodes[s][-1-t+1] \
                        for s in range(self.N)
                    )
        return self.b_nodes
    def normalize_alpha(self):
        '''alpha_hat = alpha normalized across i at each t'''
        for t in range(self.f_nodes.shape[1]):
            total = sum([self.f_nodes[i][t] for i in range(self.N)])
            for i in range(self.N):
                self.f_nodes[i][t] = self.f_nodes[i][t] / total
    def normalize_beta(self):
        '''beta_hat = beta normalized at each t < T across i'''
        for t in range(self.b_nodes.shape[1] - 1):
            total = sum([self.b_nodes[i][t] for i in range(self.N)])
            for i in range(self.N):
                self.b_nodes[i][t] = self.b_nodes[i][t] / total
    def probability_observation(self, observation):
        return sum([self.f_nodes[i][-1] for i in range(self.N)])
    def index_obs(self, o):
        '''return index position in observations of given observation'''
        return(self.obs.index(o))
    def gamma_matrix(self, T):
        '''return normalized gamma_matrix'''
        self.g_mat = np.array([
            [self.f_nodes[i][t] * self.b_nodes[i][t]
                for t in range(T)]
                    for i in range(self.N)
        ])
        self.norm_g_mat = self.g_mat.copy()

        # normalize gamma matrix across each state
        for i in range(self.N):
            i_sum = sum(self.g_mat[i])
            for t in range(T):
                self.norm_g_mat[i][t] = self.norm_g_mat[i][t] / i_sum
    def argmax_gamma(self, T):
        '''take maximum state probability at each timepoint and return string'''
        max_indices = []
        for t in range(T):
            position_probabilities = [self.norm_g_mat[i][t] for i in range(self.N)]
            max_indices.append(position_probabilities.index(max(position_probabilities)))
        return [self.states[i] for i in max_indices]
    def viterbi(self, observation):
        '''
        - forward propagate each timepoint using
            -delta  (maximum t-1)
            -psi    (maximum_i t-1)
        - backtrack through nodes and return
            - highest probability delta and follow psi to t=0
        '''
        T = len(observation)
        self.viterbi_forward(T, observation)
        return self.viterbi_max(T)
    def viterbi_forward(self, T, observation):
        '''populate the the state-timepoint lattice using the viterbi algorithm'''

        # initialize trellis with starting probabilities of states and previous
        # indexed [i][t] for state-timepoint
        trellis = np.array([
            [{"prob" : 0.0 if t > 0 else self.pi[i], "prev" : None } for t in range(T)] for i in range(self.N)
        ])


        # iterate through all timepoint~states
        for t in range(1, T):
            for i in range(self.N):

                # calculate all transition probabilities
                transition_probabilities = [trellis[s][t-1]['prob'] * self.A[s][i] for s in range(self.N)]

                # take maximum
                max_transition = max(transition_probabilities)

                # calculate psi_t_j
                psi = np.argmax(transition_probabilities)

                # calculate delta_t_j
                delta = transition_probabilities[psi] * self.B[i][self.index_obs(observation[t])]

                # add delta and psi to trellis
                trellis[i][t]['prob'], trellis[i][t]['prev'] = delta, psi

        self.viterbi_trellis = trellis
    def viterbi_max(self, T):
        '''return maximum likelihood sequence through backtracking forward probabilities'''

        # highest probable end state
        p_star = max(self.viterbi_trellis[i][-1]['prob'] for i in range(self.N))
        qt_star = [i for i in range(self.N) if (self.viterbi_trellis[i][-1]['prob'] == p_star)][0]
        backtrack = self.viterbi_trellis[qt_star][-1]['prev']

        # list to hold optimal state pattern
        optimal = [qt_star]


        # follow the backtrack
        for t in range(T - 2, -1, -1):
            optimal.insert(0, self.viterbi_trellis[backtrack][t+1]['prev'])
            backtrack = self.viterbi_trellis[backtrack][t+1]['prev']


        return [self.states[o] for o in optimal]
    def train(self, observation, iter = 10):
        T = len(observation)
        epMat = self.epsilon_matrix(observation, T)
        gamma = self.gamma_via_epsilon(T, epMat)

        new_pi = self.pi.copy()
        new_A = self.A.copy()
        new_B = self.B.copy()

        for _ in range(iter):
            new_pi = self.update_pi(gamma)
            new_A = self.update_A(T, gamma, epMat)
            new_B = self.update_B(T, gamma, observation)

        self.A = new_A
        self.B = new_B
        self.pi = new_pi
    def epsilon_matrix(self, observation, T):
        '''
        ep[t][i][j] = P(S_i at q_t & S_j at q_t+1 | model, observation)

        -calculate epsilon matrix for each s_ij for each timepoint t < T
        -normalize at each timepoint
        '''

        # indexed Timepoint ~ i ~ j (epMat[t][i][j])
        epMat = np.array(
            [[[0.0 for j in range(self.N)] for i in range(self.N)] for t in range(T-1)]
        )

        alpha = self.forward(observation)
        beta = self.backward(observation)

        # populate epsilon matrix for probabilities of change S_ij for each t < T
        total = [[] for _ in range(T-1)]
        for t in range(T-1):
            for i in range(self.N):
                for j in range(self.N):
                    eps_tij = alpha[i][t] * \
                        self.A[i][j] * \
                        self.B[j][self.index_obs(observation[t+1])] * \
                        beta[j][t+1]

                    # populate position
                    epMat[t][i][j] = eps_tij

                    # append to timepoint specfic total_list
                    total[t].append(eps_tij)



        # normalize epsilon matrix by timepoint_totals
        normalized_epMat = epMat.copy()
        for t in range(T-1):
            for i in range(self.N):
                for j in range(self.N):
                    normalized_epMat[t][i][j] = normalized_epMat[t][i][j] / sum(total[t])


        return normalized_epMat
    def gamma_via_epsilon(self, T, epMat):
        '''
        calculate expected number of transitions from s_i across all T
        '''

        # expected n_transitions from S_i (gamma[t][i])
        gamma = np.array(
            [[0.0 for i in range(self.N)] for t in range(T)]
        )

        # populate gamma using epsilon
        for t in range(T-1):
            for i in range(self.N):
                gamma[t][i] = sum(epMat[t][i][j] for j in range(self.N))

        return gamma
    def update_pi(self, gamma):
        '''update pi with gamma_i1'''
        return [gamma[0][i] for i in range(self.N)]
    def update_A(self, T, gamma, epMat):
        '''create new transition matrix from epsilon and gamma'''

        # plateholder of same size of original transition matrix
        new_A = self.A.copy()


        # for each transition S_ij divide the sums of eps[tij] and gamma[ti]
        for i in range(self.N):
            for j in range(self.N):
                num = sum([epMat[t][i][j] for t in range(T-1)])
                denom = sum([gamma[t][i] for t in range(T-1)])
                new_A[i][j] = num / denom

        return new_A
    def update_B(self, T, gamma, observation):
        ''''''

        new_B = self.B.copy()

        for i in range(self.N):
            for o in self.obs:
                num = sum([gamma[t][i] for t in range(T) if observation[t] == o])
                denom = sum([gamma[t][i] for t in range(T)])
                new_B[i][self.index_obs(o)] = num / denom

        return new_B
    def train_multiple(self, observation_matrix):

        arr_o = np.array([
            {'P_o' : None, 'alpha_hat' : None, 'beta_hat' : None} for _ in observation_matrix
        ])


        for o in observation_matrix:
            self.forward_backward(o, normalize = True)




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
def random_observations(obs, n, hmm):
    '''
    create a series of random observations that have higher probabilities
        of fitting in the model
    '''

    attempts = set()

    for i in range(1000):
        o = [random.choice(obs) for i in range(n)]
        if ''.join(o) not in attempts:
            hmm.forward_backward(o)
            po = hmm.probability_observation(o)
            attempts.add(''.join(o))
            att = [po, ''.join(o), ''.join(hmm.viterbi(o))]
            if po > 0.002:
                print('\t'.join(str(i) for i in att))
        # break
def read_file(fn):
    for line in open(fn,'r'):
        yield([c for c in line.strip('\n')])
def main():

    states = ['r','c','s']
    obs = ['u','n']

    # transitions of states S{1..n}
    A = np.array([
        [0.4,0.3,0.3],
        [0.2,0.6,0.2],
        [0.1,0.1,0.8]
    ])

    # emissions of states S{1..n}+O{1..m}
    B = np.array([
        [0.9,0.1],
        [0.6,0.4],
        [0.1,0.9]
    ])

    # initial probability of S{1..n}
    pi = np.array(
        [0.3, 0.4, 0.3]
    )

    p = argparse.ArgumentParser()
    p.add_argument('-i','--input',help='observations')
    p.add_argument('-n','--num_iter',help='iterations')
    args = p.parse_args()

    if args.input != None:
        observation = [o for o in args.input]
    else:
        observation = ['u','u','n','n']

    if args.num_iter != None:
        iterations = int(args.num_iter)
    else:
        iterations = 10


    random.seed(42)

    #
    # num_obs = 10
    # random_observations(obs, num_obs, hmm)


    # predictions with initial probabilities
    hmm = HMM(states, obs, A, B, pi)
    fb = hmm.forward_backward(observation)
    vit = hmm.viterbi(observation)


    # predictions with random starting probabilities
    random_a = generate_random_A(states)
    random_b = generate_random_B(states, obs)
    random_pi = generate_random_pi(states)


    new_hmm = HMM(states, obs, random_a, random_b, random_pi)
    new_hmm.forward_backward(observation)
    new_hmm.train(observation, iter = iterations)
    new_fb = new_hmm.forward_backward(observation)
    new_vit = new_hmm.viterbi(observation)


    print('prediction   : ', ''.join(fb))
    print('trained_pr   : ', ''.join(new_fb))
    print('viterbi_max  : ', ''.join(vit))
    print('trained_max  : ', ''.join(new_vit))
    print('observation  : ', ''.join(observation))
def ama1():
    p = argparse.ArgumentParser()
    p.add_argument('-i','--input',help='input file to train on',required=True)
    args = p.parse_args()

    states = ['i','u']
    obs = ['t','f']

    transitions = generate_random_A(states)
    emissions = generate_random_B(states, obs)
    pi = generate_random_pi(states, equal_odds=True)
    hmm = HMM(states,obs, transitions, emissions, pi)


    obs_matrix = [o for o in read_file(args.input)]


    hmm.train_multiple(obs_matrix)





if __name__ == '__main__':
    random.seed(42)
    # main()
    ama1()
