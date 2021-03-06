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
    print """python modelNeighbour.py --run=Data/dotssurvey-extrememodemedian.txt 
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

    userData={}

    dataF=open(dataFile,'r')

    count=0

    for line in dataF:
        if count == 0:
            count =count +1
            continue
        line= line.strip()
        terms = line.split("\t")
        userID, firstInit, secondInit, thirdInit, firstFinal, secondFinal, thirdFinal, firstNeighbour, secondNeighbour, thirdNeighbour = (None,)*10
        if len(terms) == 7:
            userID, firstInit, secondInit, thirdInit, firstFinal, secondFinal, thirdFinal = terms
        elif len(terms) == 8:
            userID, firstInit, secondInit, thirdInit, firstFinal, secondFinal, thirdFinal, firstNeighbour = terms
        elif len(terms) == 9:
            userID, firstInit, secondInit, thirdInit, firstFinal, secondFinal, thirdFinal, firstNeighbour, secondNeighbour = terms
        elif len(terms) == 10:
            userID, firstInit, secondInit, thirdInit, firstFinal, secondFinal, thirdFinal, firstNeighbour, secondNeighbour, thirdNeighbour = terms
        
        userData[userID]=[firstInit, secondInit, thirdInit, firstFinal, secondFinal, thirdFinal, firstNeighbour, secondNeighbour, thirdNeighbour]
        
        #userID, firstInit, secondInit, thirdInit, firstFinal, secondFinal, thirdFinal
        # firstNeighbour, secondNeighbour, thirdNeighbour
        # print allNeighbour
        # print firstNeighbour
        # print secondNeighbour
        # print thirdNeighbour
    return userData



if __name__=="__main__":

    argvNum=1
    if len(sys.argv)<=argvNum:
        error()
    myArgs=cmdProcess(sys.argv[1:])


    runF=myArgs['run']

    userData=generateData(runF)

    initOpinions=[]
    finalOpinions=[]

    for user in userData:
        initOpinions.append(int(userData[user][0]))
        # print int(userData[user][0])
        finalOpinions.append(int(userData[user][3]))

    initOpinions.sort()
    finalOpinions.sort()
    values_A_to_plot = [200001 if i > 200000 else i for i in initOpinions]
    values_B_to_plot = [200001 if i > 200000 else i for i in finalOpinions]


    # fit = stats.norm.pdf(initOpinions, np.mean(initOpinions), np.std(initOpinions))  #this is a fitting indeed
    
    # plt.plot(initOpinions,fit,'-o')

    binwidth=(max(values_A_to_plot)-min(values_A_to_plot))/40




    plt.hist([values_A_to_plot,values_B_to_plot],bins=range(min(initOpinions), max(initOpinions) + binwidth, binwidth),normed=False,alpha=0.5, label='Inital Opinion')


    # plt.hist(finalOpinions,bins=range(min(finalOpinions), max(finalOpinions) + binwidth, binwidth),normed=False,alpha=0.5,label='Final Opinion')  
    #use this to draw histogram of your data

    plt.ylabel('#Users', fontsize=18)

    plt.xlabel('Initial Opinion', fontsize=18)
    plt.legend(loc='upper right')

    plt.show()

    

