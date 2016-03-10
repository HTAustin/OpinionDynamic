# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Create an G{n,m} random graph with n nodes and m edges
and report some properties.

This graph is sometimes called the Erdős-Rényi graph
but is different from G{n,p} or binomial_graph which is also
sometimes called the Erdős-Rényi graph.
"""
__author__ = """Haotian Zhang"""
__credits__ = """"""
#    Copyright (C) 2004-2015 by 
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
import matplotlib.pyplot as plt
from networkx import *
import networkx as nx
import sys
# from __future__ import division, print_function

import numpy as np
import numpy.random as rand
import random as stdrand
import scipy.sparse as sparse
from numpy.linalg import norm, inv
import scipy.stats as stats

from datetime import datetime
from tqdm import trange

from util import *
from viz import *


def preprocessArgs(s, max_rounds):
    '''Argument processing common for most models.

    Returns:
        N, z, max_rounds
    '''

    N = np.size(s)
    max_rounds = int(max_rounds) + 1  # Round 0 contains the initial opinions
    z = s.copy()

    return N, z, max_rounds

def trustFunc(s, i, j,h):
    
    trust =  np.exp( - (s[i] - s[j])**2 / h )
    return trust



def deGroot(A, s, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulates the DeGroot Model.

    Runs a maximum of max_rounds rounds of the DeGroot model. If the model
    converges sooner, the function returns.

    Args:
        A (NxN numpy array): Adjacency matrix

        s (1xN numpy array): Initial opinions vector

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in trange(1, max_rounds):

        z = A.dot(z)


        opinions[t, :] = z

        # print z

        # print norm(opinions[t - 1, :] - opinions[t, :], np.inf)
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('DeGroot converged after {t} rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'dg' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s,
                      opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]




if __name__=="__main__":  
    n=30 # 10 nodes
    # m=8 # 20 edges
    max_rounds=1000000


    lower = 0
    upper = 1
    mu = 0.5
    sigma = 0.3
    h=0.3


    # s = np.random.normal(mu, sigma, n)
    #Generate initial opinions for each agent
    s = stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=n)

    print "Initial opinions:"
    print s






    G = nx.erdos_renyi_graph(n, .2)

    A = nx.adjacency_matrix(G).todense()
    A = np.squeeze(np.asarray(A))
    # A = row_stochastic(A)
    print "Erdős-Rényi graph:"
    sA = sparse.coo_matrix(A)


    
    newG=nx.Graph()
    for i,j,v in zip(sA.row, sA.col, sA.data):
        
        # v = rand.random()
        trust = trustFunc(s,i,j,h)
        newG.add_edge(i,j,weight=trust)
        # print "(%d, %d), %s" % (i,j,v)

    newA = nx.adjacency_matrix(newG).todense()
    newA = np.squeeze(np.asarray(newA))
    print newA

    plot_weighted_graph(newG)

    
    # plot_network(A, s, k=0.2, node_size=n, iterations=max_rounds, cmap=plt.cm.cool)

    # A=gnp(n,0.3, rand_weights=True, verbose=True)
    # print "Random graph:"
    # print A.shape; print type(A),A

    # opinionsIterations=deGroot(newA, s, max_rounds, eps=1e-6, conv_stop=True, save=True)
    # plot_opinions(opinionsIterations, title='', dcolor=False, interp=True,cmap=plt.cm.cool, linewidth=1.0)



