#!/usr/bin/env python3

import numpy as np
import itertools
import argparse
import collections

def states_of_Y(x, states):
    """all possible states of Y to evaluate likelihood"""
    n_observations = len(x)
    y = list(itertools.product(states, repeat = n_observations))
    return y
def transArray(arr):
    """return the transitions of an array"""
    return [(arr[i],arr[i+1]) for i in range(len(arr) - 1)]
def print_out(y_scores):
    """print scores in a sorted order"""
    for y, score in sorted(y_scores.items(), key=lambda x: x[1]):
        print("{}\t{}".format(score, y))

def main(inputString):
    states = ['r', 's']
    obs = ['w', 'h', 'c']

    transitions = {
        ('r', 'r') : 0.7,
        ('r', 's') : 0.3,
        ('s', 'r') : 0.4,
        ('s', 's') : 0.6
    }

    emissions = {
        ('r', 'w') : 0.1,
        ('r', 'h') : 0.4,
        ('r', 'c') : 0.5,
        ('s', 'w') : 0.6,
        ('s', 'h') : 0.3,
        ('s', 'c') : 0.1
    }



    x = list(inputString)
    Y = states_of_Y(x, states)

    y_scores = dict()

    # iterate over all possible seqStates of Y
    for y in Y:

        # transistion states of sequence
        trans = transArray(y)

        # store log-likelihood values of each state
        logs = []

        # iterate across the nodes of the sequence
        for i in range(len(y)):

            # begin at node t1 and create start probability from first emission
            if i == 1:
                p_past = emissions[(y[i-1], x[i])]
                p_trans = transitions[trans[i-1]]
                p_current = emissions[(y[i], x[i])]

                # sum of log-likelihood function for node
                logs.append(
                    sum([np.log(p) for p in [p_past, p_trans, p_current]])
                )

            # from node t2 on draw p_past from logs list
            if i > 1:
                p_past = logs[i - 2]
                p_trans = transitions[trans[i-1]]
                p_current = emissions[(y[i], x[i])]

                logs.append(
                    sum(
                        [p_past] +  # already in log form
                        [np.log(p) for p in [p_trans, p_current]])
                )

        # summation of log-likelihoods of each node for each y
        y_scores[y] = sum(logs)


    print_out(y_scores)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', help = 'string of [whc] to predict states of weather', required=True)
    args = p.parse_args()
    main(args.input)
