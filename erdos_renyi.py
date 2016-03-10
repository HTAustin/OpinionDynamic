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


def preprocessArgs(s, max_rounds):
    '''Argument processing common for most models.

    Returns:
        N, z, max_rounds
    '''

    N = np.size(s)
    max_rounds = int(max_rounds) + 1  # Round 0 contains the initial opinions
    z = s.copy()

    return N, z, max_rounds



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

        print z

        print norm(opinions[t - 1, :] - opinions[t, :], np.inf)
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
	n=10 # 10 nodes
	m=8 # 20 edges
	max_rounds=10


	lower = 0
	upper = 1
	mu = 0.5
	sigma = 0.3


	# s = np.random.normal(mu, sigma, n)
	s = stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=n)


	# print s

	# G=gnm_random_graph(n,m)

	# # some properties
	# print("node degree clustering")
	# for v in nodes(G):
	#     print('%s %d %f' % (v,degree(G,v),clustering(G,v)))

	# # print the adjacency list to terminal 
	# try:
	#     write_adjlist(G,sys.stdout)
	# except TypeError: # Python 3.x
	#     write_adjlist(G,sys.stdout.buffer)

	# print G

	# nx.draw(G)
	# # nx.draw_networkx_nodes(G,],node_size=2000,nodelist=[4])
	# # nx.draw_networkx_nodes(G,pos,node_size=3000,nodelist=[0,1,2,3],node_color='b')
	# # nx.draw_networkx_edges(G,pos,alpha=0.5,width=6)
	# plt.axis('off')
	# # plt.savefig("house_with_colors.png") # save as png
	# plt.show() # display

	A=gnp(n, 0.2)

	print A

	# s=

	deGroot(A, s, max_rounds, eps=1e-3, conv_stop=True, save=True)



