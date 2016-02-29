#!/usr/bin/env python
# -*- coding: UTF-8  -*-

"""docstring
"""

__revision__ = '0.1'

import sys,os
import getopt
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages

def usage():
    print """python yourFile.py
    --help
    --baseline=baseline dump
    --new=new dump
    --topic=topic id
    --maxn=max N
    """

def error():
    usage()
    sys.exit(-1)

def cmdProcess(argv):
    myArgs={
        "defaulArgument1":"",
    }
    try:
        opts, args = getopt.getopt(argv,"h",["help","run="])
    except getopt.GetoptError:
        error()
    for opt, arg in opts:
        if opt in ("--help","-h"):
            usage()
            sys.exit()
        else:
            opt="".join(opt[2:])
            myArgs[opt]=arg
    return myArgs


# Use the first experiment to judge stubborness, and validate on second and third
def generateData(dataFile):

    gainList=[]

    dataF=open(dataFile,'r')

    for line in dataF:
        line= line.strip()
        userID, firstInit, secondInit, thirdInit, firstFinal, secondFinal, thirdFinal, firstNeighbour, secondNeighbour, thirdNeighbour = line.split("\t")
        print firstNeighbour
        print secondNeighbour
        print thirdNeighbour




if __name__=="__main__":

    argvNum=1
    if len(sys.argv)<=argvNum:
        error()
    myArgs=cmdProcess(sys.argv[1:])


    runF=myArgs['run']

    generateData(runF)


    

