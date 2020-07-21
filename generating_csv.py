import os
import csv
import pandas as pd
import numpy as np
import patsy
import statsmodels.formula.api as smf
import scipy.stats as stats
import warnings
"""
This function is platform dependent! defult is for unix-like system! 
"""
def generate_csv(n: int, k : int, rho: float, iterations :int, sigma_x = 1.0, sigma_beta = 1.0, sigma_mu = 1.0, maxfile = 100, platform = 'u'):
    nd = iterations//maxfile + 1 if iterations%maxfile != 0 else iterations//maxfile # numbers of directories
    ns  = str(n)+'_'+str(k)+'_'+str(rho)+'_'      # n_k_rho_
    # ---------- platform dependent ---------- #
    if platform == 'u':   # for unix-like
        s = '/'
    elif platform == 'w': # for windows
        s = '\\'
    # ------------ cd to test_ds ------------- #
    if  not os.path.exists("test_ds"):
        os.mkdir("test_ds")
    os.chdir("test_ds")
    # ---------------------------------------- #
    os.mkdir(str(n)+'_'+str(k)+'_'+str(rho))
    os.chdir(str(n)+'_'+str(k)+'_'+str(rho))
    for i in range(nd):
        os.mkdir(ns+str(i+1))
        os.chdir(ns+str(i+1))
        for j in range( iterations%maxfile if i == nd-1 and iterations%maxfile != 0 else maxfile):
            head     = [ "y" if i == k+1 else"x"+str(i) for i in range(1,k+2)]   # x1, x2, x3... y
            tmp_x    = np.random.normal(0,sigma_x,np.array((n,k)))
            tmp_beta = np.random.normal(0,sigma_beta,np.array((k,1)))          # beta_1, beta_2...
            tmp_mu   = np.random.normal(0,sigma_mu,np.array((n+1,1)))          # n+1
            epsilon  = np.array([ rho*tmp_mu[i]+tmp_mu[i+1] for i in range(len(tmp_mu)-1) ])
            beta_0   = np.random.normal(0,sigma_beta) 
            y = np.dot(tmp_x,tmp_beta)+beta_0+epsilon
            tmp_data = np.hstack((tmp_x,y))
            tmp_df = pd.DataFrame(tmp_data,columns = head)
            tmp_df.to_csv( ns+str(i+1)+'_'+str(j+1+maxfile*i)+".csv" ,mode = 'a',header = True,index = False)
        os.chdir(".."+s)
    os.chdir(".."+s)
    os.chdir(".."+s)

"""
nlist = [10,20,50,100,200,500,1000]
klist = [1,2,3,5,10]
rholist = [0,0.1,0.3,0.5]
it = 5000

# generating .csv

for i in nlist:
    for j in klist:
        for k in rholist:
            generate_csv(i,j,k,it)
"""