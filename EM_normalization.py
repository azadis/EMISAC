#!/usr/bin/python

"This code is developed by Samaneh Azadi in Prof. Pieter Abbeel's lab at UC Berkeley with the colaboration of Jeremy Maitin-Shepard."

"The main code"

""" Copyright (C) {2014}  {Samaneh Azadi, Jeremy Maitin-Shepard, Pieter Abbeel}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>."""

import os
import sys
import getopt
import numpy as np
import scipy as sp
import scipy.sparse as scsp
import scipy.misc as scimg
import h5py
import time
from itertools import *
from skimage.morphology import rectangle
import skimage.filter.rank as rank
from joblib import Parallel, delayed
from os.path import split as split_path, join as join_path
from fnmatch import filter as fnfilter
import Image
from numpy import squeeze, array,newaxis,concatenate
from scipy.stats.mstats import mquantiles
import scipy.ndimage.filters
import multiprocessing, threading, ctypes
import scipy.ndimage


def usage():
	print "Help:"
	print "Usage : Normalization_command.py [options] <I> <w_x> <w_y> <par_x> <par_y> <n_cpu> <output-path>\n"
	print "<I>		STRING	The path to 3D input image on which you want to do normalization."
	print "			(it should be an image file of .h5 or .tif formats)"
	print "<stp_x>		INT The sampling rate of the input image in the x coordinate. "
	print "<stp_y>		INT The sampling rate of the input image in the y coordinate. "
	print "			NOTE: w_x should be dividable by stp_x, and w_y should be dividable by stp_y."
	print "<w_x>		INT	The minimum length of the artifacts you want to remove from the "
	print "			image in X-Z direction (It is recommended to choose it around 32)."
	print "<w_y>		INT	The minimum length of the artifacts you want to remove from the"
	print "			image in Y-Z direction(It is recommended to choose it around 32)."
	print "			NOTE: Not to deface the image in the borders, you should use the"
	print "			values for w_x and w_y to which both the lenght of your input image  "
	print "			and downsampled input image in x and y directions are dividable respectively!!"
	print "			EXAMPLE: if the size of your image is [1024,1020,100], "
	print "			you can use stp_x=4, stp_y=2, w_x=32, w_y=30."
	print "<par_x>  	INT A paramter that shows you can divide your image in x-direction in "
	print "			'par_x' partitions to speed up the algorithm."
	print "			A higher value causes a faster process, but it should not be too large"
	print "<par_y>		INT	A paramter that shows you can divide your image in y-direction in "
	print "			'par_y' partitions to speed up the algorithm."
	print "			A higher value causes a faster process, but it should not be too large"
	print "			You should consider the following note..."
	print "			NOTE:(size_of_x_dim/w_x) & (size_of_downsampled_x_dim)should be dividable "
	print "			to par_x (and the same for y-dim)..."
	print "<n_cpu>   	INT	Defines the number of CPUs used for parallel computation on the partitions. "
	print "			This param depends on the number of the CPUs of your computer."
	print "			NOTE: max(num_parallel)= (# of CPUs). However, if your memory"
	print "			is too low, you should not use all of the CPUs"
	print "<output_path> STRING	Output directory where the generated normalized files will be placed." 
	print "options: "
	print "--grp		STRING	The group name of your input file if it is .h5, and the group name of"
	print "			the output files. [Default: 'stack']"
	print "--up_fg		0/1	If you do not want to upsample your input image in the z-direction"
	print "			not to have a better resolution, you should put the 'up_fg=0'.[Default:1]"
	print "--up_z		INT	If you decide to upsample the input image, 'up_z' is the parameter"
	print "			that shows the size of the upsmapled image is how many times larger "
	print "			than the original image in z-dir.[Default: 4]"			
	print " --fact	      FLOAT	Determines the accuracy of the optimization process.[Default: 1e10]"
	print "			Typical values for fact are: 1e12 for low accuracy; 1e7 for moderate "
	print "			accuracy; 10.0 for extremely high accuracy. The lower the 'fact' value,"
	print "			the higher the running time. The values in [1e8,1e10] produce good results."
	print "	--cns_fg	0/1	If you want to increase the contrast of the output image, set the "
	print "			cns_fg=1. [Default=0]"
	print " --cns_low	FLOAT "
	print " --cns_high	FLOAT	If you have set cns_fg=1, you can choose which low and high quantiles"
	print "			be removed. [Default=(0.00001,0.99999)]"	
	print " --lm    	FLOAT	Determines the ratio of the regularization term to the loss function."
	print "-h			Help. Print Usage."

#-----------------------------------------------------------
# Auto-detect file format----------------------------------
def read_image_stack(fn):
	
    """Read a 3D volume of images in .tif or .h5 formats into a numpy.ndarray.
    This function attempts to automatically determine input file types and
    wraps specific image-reading functions.
    Adapted from gala.imio (https://github.com/janelia-flyem/gala)
    """
    if os.path.isdir(fn):
        fn += '/'
    d, fn = split_path(os.path.expanduser(fn))
    if len(d) == 0: d = '.'
    fns = fnfilter(os.listdir(d), fn)
    if len(fns) == 1 and fns[0].endswith('.tif'):
		stack = read_multi_page_tif(join_path(d,fns[0]))
    elif fn.endswith('.h5'):
		data=h5py.File(join_path(d,fn),'r')
		stack=data[group_name].value
    return squeeze(stack)

    
def pil_to_numpy(img):
	
    """Convert an Image object to a numpy array.
    Adapted from gala.imio (https://github.com/janelia-flyem/gala)
    """
    
    ar = squeeze(array(img.getdata()).reshape((img.size[1], img.size[0], -1)))
    return ar
       
def read_multi_page_tif(fn, crop=[None]*6):
   
    """Read a multi-page tif file into a numpy array.
   Currently, only grayscale images are supported.
   Adapted from gala.imio (https://github.com/janelia-flyem/gala)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = crop
    img = Image.open(fn)
    pages = []
    if zmin is not None and zmin > 0:
        img.seek(zmin)
    eof = False
    while not eof and img.tell() != zmax:
        pages.append(pil_to_numpy(img)[...,newaxis])
        try:
            img.seek(img.tell()+1)
        except EOFError:
            eof = True
    return concatenate(pages, axis=-1)
    
#----------------------------------------------------------
#----------------------------------------------------------	

def producer_parallel():
	for parnum in range(num_partitions_x*num_partitions_y):
		print('Optimizing partition%s' % parnum)
		partnum_list.append(parnum)
	return partnum_list
	
			
def make_shared_farray(shape):
    import multiprocessing, ctypes
    size = np.prod(shape)
    arr = multiprocessing.Array(ctypes.c_float, size)
    x = np.frombuffer(arr.get_obj(), dtype = np.float32)
    return (x.reshape(shape), arr)			

		
def smoothing_func(lower_val_beta,upper_val_beta,lower_val_alpha,upper_val_alpha):
	
	"""
	This function eliminates the blocking effects appeared from the previous stages.
	The input arguments are explained in the image_normalization function.
	"""
	print 'The parameters are found successfully, and they are saved.'
	print 'Please wait to do smoothing over the parameters...'
	smooth_ngbh_x=5
	smooth_ngbh_y=5
	selem=rectangle(smooth_ngbh_x,smooth_ngbh_y)
	accuracy1=0.0025
	accuracy2=.05
	r=r_org/num_partitions_x
	c=c_org/num_partitions_y
	max_shared_array_size = 33000000

	def do_bilateral():
		
		num_threads = num_parallel
		volume_size = np.prod([r_org,c_org,s])
		volume_parts = np.ceil(np.float(volume_size)/ (100*max_shared_array_size))
		value_total = np.zeros((r_org,c_org,s))
		for z_division in np.arange(volume_parts):
			s_portion = int(s/volume_parts)
			s_portion_prev = s_portion
			if z_division==volume_parts:
				s_portion = s - volume_parts*s_portion_prev
			value_portion, value_portion_arr = make_shared_farray([r_org,c_org,s_portion])
			v_queue = multiprocessing.Queue()
			finished_count = multiprocessing.Value(ctypes.c_long, 0)
			buckets=s_portion
			for bucket in range(buckets):
				v_queue.put(bucket)

			def thread_proc():
				while True:
					try:
						bucket = v_queue.get_nowait()
					except:
						break
					
					beta_mat_block=beta[:,:,(z_division*s_portion)+bucket]
					alpha_mat_block=alpha[:,:,(z_division*s_portion)+bucket]
		
					correction_mat_pixel=scsp.diags(Dimg_org[((z_division*s_portion)+bucket)*r_org*c_org:((z_division*s_portion)+bucket+1)*r_org*c_org],0)
					correction_mat2_pixel=scsp.eye(r_org*c_org,r_org*c_org)
					correction_mat_pixel=scsp.hstack([correction_mat_pixel,correction_mat2_pixel])
					del correction_mat2_pixel
		
					beta_mat_block=np.repeat(beta_mat_block,window_size_x,axis=0)	
					beta_mat=np.repeat(beta_mat_block,window_size_y,axis=1)	
					del beta_mat_block
					beta_mat=np.uint16((float(1)/float(accuracy1))*beta_mat)
					beta_changed=rank.mean_bilateral(beta_mat,selem=selem,s0=lower_val_beta,s1=upper_val_beta)
					beta_changed=(np.float32(beta_changed))*float(accuracy1)
					del beta_mat
					beta_vec=np.reshape(beta_changed,(r_org*c_org,1),order='F')
					alpha_mat_block=np.repeat(alpha_mat_block,window_size_x,axis=0)	
					alpha_mat=np.repeat(alpha_mat_block,window_size_y,axis=1)	
					del alpha_mat_block
					alpha_mat=alpha_mat+cnst
					alpha_mat=np.uint16((float(1)/float(accuracy2))*alpha_mat)
					alpha_changed=rank.mean_bilateral(alpha_mat,selem=selem,s0=lower_val_alpha,s1=upper_val_alpha)
					alpha_changed=((np.float32(alpha_changed))*float(accuracy2))-cnst
					del alpha_mat
					alpha_vec=np.reshape(alpha_changed,(r_org*c_org,1),order='F')
					param_vec=np.vstack([beta_vec,alpha_vec])
					corrected_data_smoothed=correction_mat_pixel*param_vec
					corrected_data_smoothed=np.reshape(corrected_data_smoothed,(r_org,c_org),order='F')
			
					num_finished = 0
					with finished_count.get_lock():
						finished_count.value += 1
						num_finished = finished_count.value
					#print '  Finished %d/%d' % (num_finished, buckets)
					with value_portion_arr.get_lock():
						value_portion[...][:,:,bucket]=corrected_data_smoothed
				
	
			procs = []
			for i in range(num_threads):
				p = multiprocessing.Process(target = thread_proc)
				p.start()
				procs.append(p)
			for p in procs:
				p.join()
			value_total[:,:,(z_division*s_portion_prev):(z_division*s_portion_prev+s_portion)]=value_portion
		return value_total

	alpha=np.zeros((r_org/window_size_x,c_org/window_size_y,s))
	beta=np.zeros((r_org/window_size_x,c_org/window_size_y,s))
	#---------------------------------------------------------------------------
	for partnum_y in np.arange(num_partitions_y):
		for partnum_x in np.arange(num_partitions_x):
			partnum=num_partitions_x*partnum_y+partnum_x
			f=h5py.File('%s/parameters_final_partition%s_%s.h5' %(output_path,step,partnum),'r')
			param=f[group_name].value
			alpha_part=np.reshape(param[len(param)/2:],(r/window_size_x,c/window_size_y,s),order='F')
			beta_part=np.reshape(param[0:len(param)/2],(r/window_size_x,c/window_size_y,s),order='F')
			alpha[(partnum_x*r/window_size_x):((partnum_x+1)*r/window_size_x),(partnum_y*c/window_size_y):((partnum_y+1)*c/window_size_y),:]=alpha_part
			beta[(partnum_x*r/window_size_x):((partnum_x+1)*r/window_size_x),(partnum_y*c/window_size_y):((partnum_y+1)*c/window_size_y),:]=beta_part
			
			
	Corrected_data_smoothed=np.zeros((r_org,c_org,s))
	#---------------------------------------------------
	del alpha_part
	del beta_part
	Corrected_data_smoothed=do_bilateral()
	
	var=np.max(Corrected_data_smoothed)-np.min(Corrected_data_smoothed)
	Corrected_data_smoothed=(Corrected_data_smoothed-np.min(Corrected_data_smoothed))*float(255)/float(var)
	
	f=h5py.File('%s/normalized_data_final.h5'%output_path,'w')
	dset=f.create_dataset(group_name,data=Corrected_data_smoothed)
	f.close()
	
	
	
	return Corrected_data_smoothed
	
def AutoContrast(img, quantile_low,quantile_high):
	""" This function can change the contrast of the input image by cutting the proper quantiles.
	"""
	print "Please wait to change the contrast..."
	[r,c,s]=img.shape
	print r,c,s
	img=np.reshape(img,(r*c*s,1),order='F')
	q=mquantiles(img,[quantile_low,quantile_high])
	img=(img-q[0])*255/float(q[1]-q[0])
	img[img<0]=0
	img[img>255]=255
	img=np.uint8(img)
	img=np.float32(img)
	img=np.reshape(img,(r,c,s),order='F')
	img=np.uint8(img)
	print img.shape
	f=h5py.File('%s/higher_contrast_normalized_data.h5'%output_path,'w')
	dset=f.create_dataset(group_name,data=img)
	f.close()
	return img
	
	
def minimizer_downsampled(partnum,step):
	""" This function divides the image into several partitions and normalizes each partition"""
	img=img_down[((partnum%num_partitions_x)*r_down/num_partitions_x):((partnum%num_partitions_x)*r_down/num_partitions_x+r_down/num_partitions_x),(np.floor(partnum/num_partitions_x)*c_down/num_partitions_y):(np.floor(partnum/num_partitions_x)*c_down/num_partitions_y+c_down/num_partitions_y),:]
	da=h5py.File('%s/img_partition%s.h5'%(output_path,partnum),'w')
	imgData=da.create_dataset(group_name,data=img)
	da.close()
	img_path='%s/img_partition%s.h5'%(output_path,partnum)
	"""call for the main normalization function--------------"""
	os.system("./normalize_partition.py --grp %s --lm %f --beta_bnd_l %f --beta_bnd_u %f --alpha_bnd_l %f --alpha_bnd_u %f --cnst %f --fact %f --beta_val_l %f --beta_val_u %f --alpha_val_l %f --alpha_val_u %f %s %d %d %d %d %d %d %d %s"%(group_name,lmbda,lower_bnd_beta,upper_bnd_beta,lower_bnd_alpha,upper_bnd_alpha,cnst,fact,lower_val_beta,upper_val_beta,lower_val_alpha,upper_val_alpha,img_path,partnum,window_size_x/step_x,window_size_y/step_y,num_partitions_x,num_partitions_y,num_parallel,step,output_path))
	
#----------------------------------------------------------	
def minimizer(partnum,step):
	""" This function divides the image into several partitions and normalizes each partition"""
	img=img_org[((partnum%num_partitions_x)*r_org/num_partitions_x):((partnum%num_partitions_x)*r_org/num_partitions_x+r_org/num_partitions_x),(np.floor(partnum/num_partitions_x)*c_org/num_partitions_y):(np.floor(partnum/num_partitions_x)*c_org/num_partitions_y+c_org/num_partitions_y),:]
	da=h5py.File('%s/img_partition%s.h5'%(output_path,partnum),'w')
	imgData=da.create_dataset(group_name,data=img)
	da.close()
	img_path='%s/img_partition%s.h5'%(output_path,partnum)
	"""call for the main normalization function--------------"""
	os.system("./normalize_partition.py --grp %s --lm %f --beta_bnd_l %f --beta_bnd_u %f --alpha_bnd_l %f --alpha_bnd_u %f --cnst %f --fact %f --beta_val_l %f --beta_val_u %f --alpha_val_l %f --alpha_val_u %f %s %d %d %d %d %d %d %d %s"%(group_name,lmbda,lower_bnd_beta,upper_bnd_beta,lower_bnd_alpha,upper_bnd_alpha,cnst,fact,lower_val_beta,upper_val_beta,lower_val_alpha,upper_val_alpha,img_path,partnum,window_size_x,window_size_y,num_partitions_x,num_partitions_y,num_parallel,step,output_path))
#----------------------------------------------------------			
		
	
opts, args = getopt.getopt(sys.argv[1:], "h",["grp=","up_fg=","up_z=","cnst=","fact=","cns_fg=","cns_low=","cns_high=","lm="]) 

#setting the defualt parameters:------------------------
tt0=time.time()
group_name='stack'
upsample_flag=1
upsample_z_param=2
lower_bnd_beta=0.5
upper_bnd_beta=10000
lower_bnd_alpha=-100
upper_bnd_alpha=10000
fact=1e10
fact1=fact
lower_val_beta=4000
upper_val_beta=4000
lower_val_alpha=4000
upper_val_alpha=4000
cnst=abs(lower_bnd_alpha)
contrast_flag=0
quantile_low=0.00001
quantile_high=0.99999
lm_cns=100
#--------------------------------------------------------
#setting the optional parameters as the values entered in the command line-----
if len(args) != 9 :
		print "Error:"
		print "you have not entered the whole needed inputs"
		usage()
		sys.exit()
if len(opts)>0:
	for o, a in opts:
		if o == "--grp":
			group_name=str(a)
		elif o == "--up_fg":
			upsample_flag = int(a)
		elif o == "--up_z":
			upsample_z_param = int(a)
		elif o=="--fact":
			fact1=float(a)
		elif o=="--cns_fg":
			contrast_flag=int(a)
		elif o=="--cns_low":
			quantile_low=float(a)
		elif o=="--cns_high":
			quantile_high=float(a)
		elif o=="--lm":
			lm_cns=float(a)
		else:
			usage()
			sys.exit()

    
#-----------------------------------------------------------
#read the inputs--------------------------------------------
print "Reading the Inputs..."
img_org=read_image_stack(args[0])
#-------------------------------------

img_path=args[0]
step_x=int(args[1])
step_y=int(args[2])
window_size_x=int(args[3])
window_size_y=int(args[4])
num_partitions_x=int(args[5])
num_partitions_y=int(args[6])
num_parallel=int(args[7])
output_path=args[8]
lmbda=lm_cns/float((32/float(window_size_x))**2)

#--------------------------------------------------------
img_org=np.float32(img_org)
[r_org,c_org,s]=img_org.shape
for i in np.arange(s):
	img_org[:,:,i]=(img_org[:,:,i]-np.min(img_org[:,:,i]))*255/float(np.max(img_org[:,:,i])-np.min(img_org[:,:,i]))
print r_org,c_org,s
Dimg_org=np.reshape(img_org,(r_org*c_org*s,),order='F')

#------------------------------------
#upsampling the image--------------------------------------
print "Upsampling the image..."
if upsample_flag==1:
	output_path=args[8]
	resampled_img=np.zeros((r_org,c_org,s*upsample_z_param))
	img_org=np.uint8(img_org)
	for i in np.arange(c_org):
		resampled_img[:,i,:]=scimg.imresize(img_org[:,i,:],(r_org,upsample_z_param*s),interp='cubic')
	img_org=np.float32(resampled_img)
	print img_org.shape
	del resampled_img
	[r_org,c_org,s]=img_org.shape
	g=h5py.File('%s/upsampled_train_input.h5' %output_path,'w')
	dset=g.create_dataset(group_name,data=img_org)
	g.close()
	Dimg_org=np.reshape(img_org,(r_org*c_org*s,),order='F')
	 
#----------------------------------------------------------
# printing the parameters
print "These are the parameters used in the optimization process"
print "--grp %s --up_fg %d --up_z %d --lm %f --fact %f --cns_fg %d --cns_low %f --cns_high %f %s %d %d %d %d %d %s"%(group_name,upsample_flag,upsample_z_param,lmbda,fact,contrast_flag,quantile_low,quantile_high,img_path,window_size_x,window_size_y,num_partitions_x,num_partitions_y,num_parallel,output_path)

#----------------------------------------------------------
#Starting from the downsampled version of the input image
step=1
lmbda=lm_cns/float((step_x*step_y)*(32/float(window_size_x))**2)
print 'step',step
img_down=img_org[0::step_x,0::step_y,:]
[r_down,c_down,s]=img_down.shape
partnum_list=[]
Dimg_down=np.reshape(img_down,(r_down*c_down*s,),order='F')
res= Parallel(n_jobs=num_parallel,verbose=100)(delayed(minimizer_downsampled)(partnum,step) for partnum in producer_parallel())


#----------------------------------------------------------
#parallel processing---------------------------------------
step=2
lmbda=lm_cns/float((32/float(window_size_x))**2)
print 'step',step
partnum_list=[]
fact=fact1
res= Parallel(n_jobs=num_parallel,verbose=100)(delayed(minimizer)(partnum,step) for partnum in producer_parallel())


#----------------------------------------------------------
#smoothing the image to remove the blocking effect---------
del img_down
del img_org
img_org=smoothing_func(lower_val_beta,upper_val_beta,lower_val_alpha,upper_val_alpha)
[r_org,c_org,s]=img_org.shape
Dimg_org=np.reshape(img_org,(r_org*c_org*s,),order='F')

#----------------------------------------------------------
window_size_x=window_size_x/2
window_size_y=window_size_y/2
#----------------------------------------------------------
step=3
lmbda=lm_cns/float((32/float(window_size_x))**2)
print 'step',step
partnum_list=[]
res= Parallel(n_jobs=num_parallel,verbose=100)(delayed(minimizer)(partnum,step) for partnum in producer_parallel())
#----------------------------------------------------------
del img_org
img_org=smoothing_func(lower_val_beta,upper_val_beta,lower_val_alpha,upper_val_alpha)
#----------------------------------------------------------

#Changing the contrast of the output image----------------
if contrast_flag==1:
	data=AutoContrast(img_org,quantile_low,quantile_high)
	
print "Total time:", time.time()-tt0
print "Finished Successfully."
