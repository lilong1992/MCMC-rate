# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 23:37:07 2019

@author: lilon
"""

import numpy as np
import matplotlib.pyplot as plt

costheta=np.array([0.9302,0.8547,0.9413,0.8756])
counts=np.array([3.37,2.57,3.93,3.30])
countserr=np.array([0.14,0.078,0.14,0.085])
currentscale=np.array([0,0,1,1]) #whether to do the current scaling
labely=r"counts/detector/$\mu$Ah" #raw string literal

plt.figure(1)
plt.errorbar(costheta, counts, yerr=countserr, fmt=".k", capsize=0)
plt.xlim(0.84, 0.95)
plt.xticks(np.linspace(0.84,0.95,12))
plt.rc('text', usetex=True)
plt.xlabel("Cos(scattering angles) in COM")
plt.ylabel(labely)
#plt.show()
plt.savefig("original data.pdf")

def dcs1p3(x):
    return 0.368791+0.144415*x+0.21951*x**2+0.489389*x**3+0.28405*x**4+\
    0.0319314*x**5+0.00863931*x**6-0.00011481*x**7-0.0000445186*x**8

def log_prior(paras):
    f,cf=paras
    if 1.99<f<2.87 and -0.4<cf<0.3:
        return 0.0
    return -np.inf

def log_likelihood(paras,x,y,yerr,ys):
    f,cf=paras
    model=f*dcs1p3(x)
    sigma2=(yerr*(1+ys*cf))**2
    yp=y*(1+ys*cf)#add corrections from current scaling factor
    return -0.5*np.sum((yp-model)**2/sigma2+np.log(sigma2))

def log_probability(paras,x,y,yerr,ys):
    lp=log_prior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp+log_likelihood(paras,x,y,yerr,ys)

import emcee
np.random.seed(1)
nwalkers,ndim=32,2
f=np.random.uniform(1.99,2.87,nwalkers)
cf=np.random.uniform(-0.4,0.3,nwalkers)
paras=np.stack((f,cf))
paras=paras.T #transpose the 2d array
print(paras.shape)
sampler=emcee.EnsembleSampler(nwalkers,ndim,log_probability,args=(costheta,counts,countserr,currentscale))
sampler.run_mcmc(paras,5000,progress=True) #5000 steps

#plt.figure(2)
fig, axes = plt.subplots(2, figsize=(10, 5), sharex=True)
samples = sampler.get_chain()
labels = ["f", "cf"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:,:,i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
#plt.show()
plt.savefig("MC chain plot.pdf")

tau = sampler.get_autocorr_time()
print(tau) #get steps to forget the start

flat_samples = sampler.get_chain(discard=100,thin=10,flat=True)
print(flat_samples.shape)

import corner
#plt.figure(3)
cornerfig = corner.corner(
    flat_samples, labels=["f","cf"],quantiles=[0.16,0.5,0.84],show_titles=True
)
#plt.show()
plt.savefig("cornerplot.pdf")

plt.figure(4)
x0=np.linspace(0.84,0.95,100)
""" #overlapped plots
inds = np.random.randint(len(flat_samples),size=500)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0,sample[0]*dcs1p3(x0),"C1",alpha=0.1)#draw lines with different f
"""
meanpara=np.mean(flat_samples,axis=0)
plt.plot(x0,meanpara[0]*dcs1p3(x0),"r-",alpha=1)
plt.plot(x0,(meanpara[0]+0.06)*dcs1p3(x0),"r--",alpha=1)
plt.plot(x0,(meanpara[0]-0.06)*dcs1p3(x0),"r--",alpha=1)
meancf=meanpara[1]#get the mean cf
for i in range(4):
    counts[i]=counts[i]*(1+meancf*currentscale[i])
    countserr[i]=countserr[i]*(1+meancf*currentscale[i])

plt.errorbar(costheta, counts, yerr=countserr, fmt=".k", capsize=0)
plt.xlim(0.84, 0.95)
plt.xticks(np.linspace(0.84,0.95,12))
plt.rc('text', usetex=True)
plt.xlabel("Cos(scattering angles) in COM")
plt.ylabel(labely)
#plt.show()
plt.savefig("corrected data and MC fit line.pdf")




