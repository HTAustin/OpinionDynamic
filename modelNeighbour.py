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
        opts, args = getopt.getopt(argv,"h",["help","run=","record="])
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


def generatePoints(dataFile,topic):
    gainList=[]

    dataF=open(dataFile,'r')

    for line in dataF:
        line= line.strip()
        cTopic,sample,pid,cluster,gain = line.split()
        if cTopic == topic:
            gainList.append(float(gain))

    return gainList





if __name__=="__main__":

    argvNum=1
    if len(sys.argv)<=argvNum:
        error()
    myArgs=cmdProcess(sys.argv[1:])

    topicList=["310","336","362","367","383","426","427","436"]

    run=myArgs['run']

    record = myArgs['record']

    pp = PdfPages(run.split('/')[1]+'.pdf')

    recordF = open(record,"a")


    for topic in topicList:
        h = generatePoints(run,topic)
        h.sort()

        assert len(h)==10000

        fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

        plt.plot(h,fit,'-o')

        plt.hist(h,normed=True, color='g')      #use this to draw histogram of your data

        plt.ylabel('#Samples', fontsize=18)
        plt.xlabel('Gain', fontsize=18)

        mean=np.asarray(h).mean()

        plt.axvline(np.asarray(h).mean(), color='r', linestyle='dashed', label='Mean', linewidth=2)
        # plt.text(np.asarray(h).mean(),0,'mean',rotation=90)

        median = np.median(np.asarray(h))

        plt.axvline(np.median(np.asarray(h)), color='y', linestyle='dashed', label='Median',linewidth=2)
        # plt.text(np.median(np.asarray(h)),0,'median',rotation=90)

        lower90 = np.percentile(np.asarray(h), 10)

        plt.axvline(lower90, color='c', linestyle='-.', label='10%', linewidth=3)

        upper90= np.percentile(np.asarray(h), 90)

        plt.axvline(upper90, color='b', linestyle='-.', label='90%', linewidth=3)

        plt.legend()

        plt.title(run.split('/')[1] +' - '+topic)
        # plt.show() 
        plt.savefig(pp, format='pdf')
        
        plt.close()

        recordF.write(run+"\t"+topic+"\t"+str(lower90)+"\t"+str(upper90)+"\t"+str(median)+"\t"+str(mean)+"\n")

    recordF.close()

    pp.close()

    

    # plt.plot(gainList,label='UW')

    # plt.legend()
    


    # fig = plt.figure(1, (7,4))
    # ax = fig.add_subplot(1,1,1)


    # plt.show()
    # plt.savefig(topic+'_effort_'+str(maxN)+'.png')
