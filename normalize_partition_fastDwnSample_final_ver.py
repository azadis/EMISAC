#!/usr/bin/python

"This code is developed by Samaneh Azadi in Prof. Pieter Abbeel's lab at UC Berkeley with the colaboration of Jeremy Maitin-Shepard."

import numpy as np
import scipy as sp
import getopt
import math
import scipy.sparse as scsp
from scipy.optimize import fmin_l_bfgs_b
import h5py
import scipy.sparse as sparse
from itertools import *
import time
from numpy.lib.stride_tricks import as_strided as ast
from skimage.morphology import disk
from skimage.morphology import rectangle
import skimage.filter.rank as rank
from joblib import Parallel, delayed
import os
import sys

def usage():
	print "Usage : main_normalization.py [options] <I> <w_x> <w_y> <par_x> <par_y> <n_cpu> <output_path>\n"
	print "This function generates a set of evolutionary related networks based on"
	print "inputs:"
	print "<I>    	  The path to 3D input image on which you want to do normalization."
	print "			 (it should be an image file of .h5 or .tif formats)"
	print "<stp_x>	 The sampling rate of the input image in the x coordinate. "
	print "<stp_y>	 The sampling rate of the input image in the y coordinate. "
	print "			  NOTE: w_x should be dividable by stp_x, and w_y should be dividable by stp_y."
	print "<w_x>  	  The minimum length of the artifacts you want to remove from the "
	print "			  image in X-Z direction."
	print "<w_y>     The minimum length of the artifacts you want to remove from the"
	print "			  image in Y-Z direction."
	print "			  NOTE: Not to deface the image in the borders, you should use the"
	print "			  values for w_x and w_y to which the lenght of your input image "
	print "			  in x and y directions are dividable respectively!!"
	print "			  EXAMPLE: if the size of your image is [1024,1020,100], "
	print "			  you can use w_x=32, w_y=30."
	print "<par_x>   A paramter that shows you can divide your image in x-direction in "
	print "			  'par_x' partitions to speed up the algorithm."
	print "			  A higher value causes a faster process, but it should not be too large"
	print "<par_y>   A paramter that shows you can divide your image in y-direction in "
	print "			  par_y' partitions to speed up the algorithm."
	print "			  A higher value causes a faster process, but it should not be too large"
	print "			  You should consider the following note..."
	print "			  NOTE:(size_of_x_dim/w_x) should be dividable to par_x (and the same for y-dim)..."
	print "<n_cpu>   Defines the number of CPUs used for parallel computation on the partitions."
	print "           This param depends on the number of the CPUs of your computer"
	print "			  and the available memory.max(num_parallel)= # of CPUs without considering"
	print "			  The limitations on memory."
	print "			  If your input image is too large, you should decrese the value of "
	print "			  'n_cpu' parameter. A lower value of this parameter decreases the speed,"
	print "			  but can save memory."
	print "			  NOTE: (n_cpu < = par_x*par_y)."
	print "<output_path> Output directory where the generated normalized files will be placed." 
	print "options: "
	print "--grp		STRING		The group name of your input file if it is .h5, and the group name of"
	print "							the output files. [Default: 'stack']"
	print " --lm        FLOAT       The paramter that determines the relation of normaliztion along "
	print "							x-dir and y-dir (regularization term) to the normalization along "
	print "							z-dir [Default=100]."
	print "	--beta_bnd_l  FLOAT		"
	print "	--beta_bnd_u  FLOAT		The parameters that determine the lower and upper bounds for the scaling"
	print "							factors which are multiplied by the pixel values. [Default: (0.5,100)]"	
	print "	--alpha_bnd_l FLOAT"
	print "	--alpha_bnd_u FLOAT		The parameters that determine the lower and upper bounds for the offset "
	print "							which are added to the pixel values. [Default: (-100,100)]"	
	print "	--cnst 		FLOAT		It should be a value greater than or equal to |alpha_bnd_l|. [Default: 100]"
	print " --beta_val_l FLOAT		"
	print " --beta_val_u FLOAT      Define the [beta_val_l, beta_val_u] interval around the greyvalue of" 
	print "							the center of scaling factors to be considered for computing the value"
	print "							 and smoothing over various windows [Default: [4000,4000]]."
	print " --alpha_val_l FLOAT 	"
	print " --alpha_val_u FLOAT     The same application as the previous line, except that these are used"
	print "							in smoothing over the offset values [Default=[4000,4000]]."
	print " --fact		FLOAT		Determines the accuracy of the optimization process.[Default: 1e10]"
	print "							Typical values for fact are: 1e12 for low accuracy; 1e7 for moderate "
	print "							accuracy; 10.0 for extremely high accuracy. The lower the 'fact' value,"
	print "							the higher the running time. The values in [1e8,1e10] produce good results."			
	print "		-h					Help. Print Usage."
opts, args = getopt.getopt(sys.argv[1:], "h",["grp=","lm=","beta_bnd_l=","beta_bnd_u=","alpha_bnd_l=","alpha_bnd_u=","cnst=","fact=","beta_val_l=","beta_val_u=","alpha_val_l=","alpha_val_u="])

#---------------------------------------------------------------
#reading the inputs and optional paramteres---------------------

for o, a in opts:
	if o == "--grp":
		group_name=str(a)
	elif o == "--lm":
		 lmbda= float(a)
	elif o=="--beta_bnd_l":
		lower_bnd_beta=float(a)
	elif o=="--beta_bnd_u":
		upper_bnd_beta=float(a)
	elif o=="--alpha_bnd_l":
		lower_bnd_alpha=float(a)
	elif o=="--alpha_bnd_u":
		upper_bnd_alpha=float(a)
	elif o=="--fact":
		fact=float(a)
	elif o=="--beta_val_l":
		lower_val_beta=float(a)
	elif o=="--beta_val_u":
		upper_val_beta=float(a)
	elif o=="--alpha_val_l":
		lower_val_alpha=float(a)
	elif o=="--alpha_val_u":
		upper_val_alpha=float(a)
	elif o=="--cnst":
		cnst=float(a)
	else:
		usage()
		sys.exit()
		
img_path=args[0]
d=h5py.File(img_path,'r')
img=d[group_name].value
partnum=int(args[1])
window_size_x=int(args[2])
window_size_y=int(args[3])
num_partitions_x=int(args[4])
num_partitions_y=int(args[5])
num_parallel=int(args[6])
step=int(args[7])
output_path=args[8]

#-------------------------------------------------------------
#------------------------------------------------------------
		
def func(x,sign=1,lmbda=lmbda):
	"""objective function"""
	fx=0
	CX=correction_mat*x
	GXa=0
	GXb=0
	LXa=0
	LXb=0
	global CXr
	
	CXr=np.reshape(CX,(r,c,s),order='F')
	t1=time.time()
	FCX=np.diff(CXr,axis=2)
	FCX[:,:,-1]=FCX[:,:,-1]*math.sqrt(2)
	
	br=np.reshape(x[0:len_x0_h],(partition_num_block_r,partition_num_block_c,s),order='F')
	ar=np.reshape(x[len_x0_h:],(partition_num_block_r,partition_num_block_c,s),order='F')
	if partition_num_block_c>1:
		GXb=np.diff(br,axis=1)
		GXb[:,-1,:]=GXb[:,-1,:]*math.sqrt(2)
		GXa=np.diff(ar,axis=1)
		GXa[:,-1,:]=GXa[:,-1,:]*math.sqrt(2)
	if partition_num_block_r>1:
		LXb=np.diff(br,axis=0)
		LXb[-1,:,:]=LXb[-1,:,:]*math.sqrt(2)
		LXa=np.diff(ar,axis=0)
		LXa[-1,:,:]=LXa[-1,:,:]*math.sqrt(2)
	
	fx=np.sum(FCX**2)+lmbda*(np.sum((GXa**2+GXb**2))+np.sum((LXb**2+LXa**2)))
	return fx
	
def func_deriv(x,sign=1.0,lmbda=lmbda):
	""" Derivative of  the objective function """
	derv_alpha_G3=np.zeros((partition_num_rc*s,1))
	derv_alpha_L3=np.zeros((partition_num_rc*s,1))
	derv_beta_G3=np.zeros((partition_num_rc*s,1))
	derv_beta_L3=np.zeros((partition_num_rc*s,1))
	derv_beta_G=np.reshape(x[0:len_x0_h],(partition_num_block_r,partition_num_block_c,s),order='F')
	derv_alpha_G=np.reshape(x[len_x0_h:],(partition_num_block_r,partition_num_block_c,s),order='F')
	if partition_num_block_c>1:
		
		derv_beta_G2=-np.diff(derv_beta_G,axis=1)
		derv_beta_G3=2*np.hstack((derv_beta_G2[:,0,:][:,np.newaxis,:],np.diff(derv_beta_G2,axis=1)))
		derv_beta_G3[:,-1,:]=derv_beta_G3[:,-1,:]+2*derv_beta_G2[:,-1,:]
		derv_beta_G3=np.hstack((derv_beta_G3,-2*derv_beta_G2[:,-1,:][:,np.newaxis,:]))

		derv_alpha_G2=-np.diff(derv_alpha_G,axis=1)
		derv_alpha_G3=2*np.hstack((derv_alpha_G2[:,0,:][:,np.newaxis,:],np.diff(derv_alpha_G2,axis=1)))
		derv_alpha_G3[:,-1,:]=derv_alpha_G3[:,-1,:]+2*derv_alpha_G2[:,-1,:]
		derv_alpha_G3=np.hstack((derv_alpha_G3,-2*derv_alpha_G2[:,-1,:][:,np.newaxis,:]))
	if partition_num_block_r>1:
		derv_beta_L2=-np.diff(derv_beta_G,axis=0)
		derv_beta_L3=2*np.vstack((derv_beta_L2[0,:,:][np.newaxis,:,:],np.diff(derv_beta_L2,axis=0)))
		derv_beta_L3[-1,:,:]=derv_beta_L3[-1,:,:]+2*derv_beta_L2[-1,:,:]
		derv_beta_L3=np.vstack((derv_beta_L3,-2*derv_beta_L2[-1,:,:][np.newaxis,:,:]))
	
		derv_alpha_L2=-np.diff(derv_alpha_G,axis=0)
		derv_alpha_L3=2*np.vstack((derv_alpha_L2[0,:,:][np.newaxis,:,:],np.diff(derv_alpha_L2,axis=0)))
		derv_alpha_L3[-1,:,:]=derv_alpha_L3[-1,:,:]+2*derv_alpha_L2[-1,:,:]
		derv_alpha_L3=np.vstack((derv_alpha_L3,-2*derv_alpha_L2[-1,:,:][np.newaxis,:,:]))
	
	bb=np.reshape(derv_beta_G3+derv_beta_L3,(partition_num_rc*s,1),order='F')
	aa=np.reshape(derv_alpha_G3+derv_alpha_L3,(partition_num_rc*s,1),order='F')
	derv_lambda=lmbda*np.vstack((bb,aa))

	derv_F2=-np.diff(CXr,axis=2)
	derv_F3=2*np.dstack((derv_F2[:,:,0][:,:,np.newaxis],np.diff(derv_F2,axis=2)))
	derv_F3[:,:,-1]=derv_F3[:,:,-1]+2*derv_F2[:,:,-1]
	derv_F3=np.dstack((derv_F3,-2*derv_F2[:,:,-1][:,:,np.newaxis]))
	
	ff=np.reshape(derv_F3,(r*c*s,1),order='F').T
	derv_ff=(ff*correction_mat).T
	
	df=sign*(derv_ff+derv_lambda)[:,0]
	return df

def block_view(A, block= (3, 3)):
	"""Provide a 2D block view to 2D array. No error checking made.
	Therefore meaningful (as implemented) only for blocks strictly
	compatible with the shape of A."""

	shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
	strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
	return ast(A, shape= shape, strides= strides)

#-------------------------------------------------------------
#the image proeprties----------------------------------------
[r,c,s]=img.shape
partition_num_block_r=(r/window_size_x)
partition_num_block_c=(c/window_size_y)
partition_num_rc=partition_num_block_c*partition_num_block_r
Dimg=np.reshape(img,(r*c*s,),order='F')

#-------------------------------------------------------------
#initial guess for optimization paramters---------------------
if step==1 or step==3:
	x0=np.zeros((1,partition_num_rc*s))
	x0=np.append((np.ones((1,partition_num_rc*s))),x0).T
	
elif step==2 or step==4:
	g=h5py.File('%s/parameters_final_partition%s_%s.h5'%(output_path,step-1,partnum),'r')
	x0=g[group_name].value
	g.close()
	bnds=list(repeat((lower_bnd_beta,upper_bnd_beta),int(partition_num_rc*s)))+list(repeat((lower_bnd_alpha,upper_bnd_alpha),int(partition_num_rc*s)))


len_x0_h=partition_num_rc*s

#-------------------------------------------------------------
#calculating the matrices to formulate the problem as a convex optimization problem----
tstart0=time.time()
cnt=0
correction_mat_list=np.zeros((r*c*s,3))
for i in np.arange(s):
	""" cb:coloumn block
		rb: row block"""
	bb=block_view(img[0:partition_num_block_r*window_size_x,0:partition_num_block_c*window_size_y,i],(window_size_x,window_size_y))
	for cb in np.arange(partition_num_block_c):
		for rb in np.arange(partition_num_block_r):
			blck=bb[rb,cb]
			correction_mat1_block=np.reshape(blck,(window_size_x*window_size_y,),order='F')
			for w in np.arange(window_size_y):
				correction_mat_list[((w*window_size_x)+cnt*r*c/partition_num_rc):((w+1)*window_size_x+cnt*r*c/partition_num_rc),0]=np.arange(((cb*r*window_size_y+w*r)+rb*window_size_x+i*r*c),((cb*r*window_size_y+w*r)+rb*window_size_x+window_size_x+i*r*c))
			correction_mat_list[(cnt*(window_size_x*window_size_y)):((cnt+1)*window_size_x*window_size_y),1]=cnt
			correction_mat_list[(cnt*(window_size_x*window_size_y)):((cnt+1)*window_size_x*window_size_y),2]=correction_mat1_block
			cnt+=1
correction_mat=scsp.coo_matrix((correction_mat_list[:,2],(correction_mat_list[:,0],correction_mat_list[:,1])),shape=(r*c*s,partition_num_rc*s))

del bb
del blck

correction_mat2_block=np.ones((window_size_x*window_size_y,))
correction_mat2_block_x=np.ones(((r%window_size_x)*window_size_y),)

cnt=0
correction_mat2_list=np.zeros((r*c*s,3))
for i in np.arange(s):
	""" cb:coloumn block
		rb: row block"""
	for cb in np.arange(partition_num_block_c):
			for rb in np.arange(partition_num_block_r):
				for w in np.arange(window_size_y):
					correction_mat2_list[((w*window_size_x)+cnt*r*c/partition_num_rc):((w+1)*window_size_x+cnt*r*c/partition_num_rc),0]=np.arange(((cb*r*window_size_y+w*r)+rb*window_size_x+i*r*c),((cb*r*window_size_y+w*r)+rb*window_size_x+window_size_x+i*r*c))
				correction_mat2_list[(cnt*(window_size_x*window_size_y)):((cnt+1)*window_size_x*window_size_y),1]=cnt
				correction_mat2_list[(cnt*(window_size_x*window_size_y)):((cnt+1)*window_size_x*window_size_y),2]=correction_mat2_block
				cnt+=1
correction_mat2=scsp.coo_matrix((correction_mat2_list[:,2],(correction_mat2_list[:,0],correction_mat2_list[:,1])),shape=(r*c*s,partition_num_rc*s))
correction_mat=scsp.hstack([correction_mat,correction_mat2])
correction_mat=correction_mat.tocsc()

GXa=0
GXb=0
LXa=0
LXb=0

#-----------------------------------------------------
#saving memory----------------------------------------
del img
del correction_mat1_block
del correction_mat2_block
del correction_mat2_block_x
del correction_mat2_list
del correction_mat_list
del correction_mat2
#-----------------------------------------------------

#Calling the optimization function--------------------

bnds=list(repeat((lower_bnd_beta,upper_bnd_beta),int(partition_num_rc*s)))+list(repeat((lower_bnd_alpha,upper_bnd_alpha),int(partition_num_rc*s)))

tt1=time.time()
res=fmin_l_bfgs_b(func, x0,fprime=func_deriv,bounds=bnds,factr=fact)

tt2=time.time()-tt1

f_x=h5py.File('%s/parameters_final_partition%s_%s.h5' %(output_path,step,partnum),'w')
dset=f_x.create_dataset(group_name,data=res[0])
f_x.close()

#------------------------------------------------------

