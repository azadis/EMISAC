EMISAC
======

Optimization-Based Artifact Correction for Electron Microscopy Image Stacks

---------------------------------------------------------------------------------------------

This code is developed in python to eliminate the artifacts in the electron microscopy image
stacks most notably along the section axes,and produce an image with a higher quality. 
These artifacts are the result of variations in  section thickness, sample deformations,
and variations in imaging conditions.

Another application of this code is in time-lapse photography where the light correction 
is needed.

The first version 1.0 (June 2014) is developed by Samaneh Azadi in a colaboration with 
Jeremy Maitin-Shepard in the Robotic Learning Lab, EECS department, UC Berkeley under the 
supervision of Prof. Pieter Abbeel.

For any questions about the algorithm or the code, feel free to email sazadi156@gmail.com .

------------------------------------------------------------------------------------------
This code gets the original 3D image stack and several parameters as the input and produces 
the normalized image as an output which will be saved in the path specified by the user.
A short description of the method is as follows:

1. The 3D input image is downsampled in x- and y- directions as a coarse estimation in the 
optimization algorithm.

2. The 3D image is divided into the equal sized 3D partitions, and these partitions 
are considered as seperate images optimized in parallel. The number of these partitions 
in x,y are two of the input parameters.

3. For each 3D partition, we consider non-overlapping smaller 2D blocks. For each block,
a scaling factor(beta) and offset(alpha) parameters are supposed which should be multiplied
by and added to each voxel respectively, through an affine transformation,
(x_2=beta*x_1+alpha). The size of these blocks in the x,y directions are two other
input parameters. These two parameters determine the minimum size of the local intensity
discrepencies which should be eliminated.
									

4. Difference between the affine transformations of the aside blocks along the z-direction is 
optimized by the L-BFGS-B algorithm. Parameters in this optimization process are the 
mentioned scaling factors and offsets which should be bounded. Also, a regularization
term is added to the objective function to prevent defacing the image in the x,y directions.

5. All of the steps 1-4 are repeated for the actual 3D image without any downsmapling. 
The initial values for the optimization parmaters in this step are the outputs of the 
previous steps.

6. All of the steps 1-4 are repeated for the actual image but by cinsidering smaller blocks.
(new_block_size=old_block_size/2). The initial values for the optimization parameters in this
step are the outputs of step 5.

7. The optimized parameters generated in step 6 are smoothed with an averaging filter 
to remove the blocking effect produced in the first and second steps.

#Required python modules:
	numpy
	scipy
	h5py
	skimage
	joblib
	
#Usage:
	./normalization_command.py [options] <I> <stp_x> <stp_y> <w_x> <w_y> <par_x> <par_y> <n_cpu> <output-path>
	
#Example:
	As an example, the input image is the snemi3d image available in http://brainiac2.mit.edu/SNEMI3D/.
	The size of the image is (1024,1024,100).
	./normalization-commad.py --up_fg 0 ./train-input.h5 4 4 32 32 4 4 8 /home/normalized_files
	or
	./normalization-commad.py ./train-input.h5 4 4 32 32 4 4 8 /home/normalized_files
	
#Description:
	<I>			STRING		The path to 3D input image on which you want to do normalization.
							(it should be an image file of .h5 or .tif formats)
	<stp_x>		INT 		The sampling rate of the input image in the x coordinate.
	<stp_y>		INT			The sampling rate of the input image in the y coordinate.
							NOTE: w_x should be dividable by stp_x, and w_y should be dividable by stp_y.
	<w_x>		INT			The minimum length of the artifacts you want to remove from the 
							image in X-Z direction.
	<w_y>		INT			The minimum length of the artifacts you want to remove from the
							image in Y-Z direction.
							NOTE: Not to deface the image in the borders, you should use the
							values for w_x and w_y to which the lenght of your input image 
							in x and y directions are dividable respectively!! Also, w_x and w_y
							should be even numbers. 
							EXAMPLE: if the size of your image is [1024,1020,100], 
							you can use w_x=32, w_y=30.
	<par_x>   	INT			A paramter that shows you can divide your image in x-direction in 
							'par_x' partitions to speed up the algorithm.
							A higher value causes a faster process, but it should not be too large. 
	<par_y> 	INT			A paramter that shows you can divide your image in y-direction in 
							'par_y' partitions to speed up the algorithm.
							A higher value causes a faster process, but it should not be too large.
							You should consider the following notes...
							NOTE:(size_of_x_dim/w_x) should be dividable to par_x (and the same for y-dim),
							NOTE:(The values for par_x and par_y should be set such that the quotient of
							(size_of_x_dim/par_x)/w_x and (size_of_y_dim/par_y)/w_y be integer numbers
							near 8.0 or larger for EM images, so that we have 8 blocks in each partition in 						
							both x,y directions.) For the time-lapse photography correction it might be better 					
							to set par_x and par_y equal to 1.
							EXAMPLE: Consider the snemi3d image. If you divide it to par_x,par_y=4 
							partitions in both x,y, each partition would be of size(256,256,100).
							The length of 256 can be divided to 8 windows of size 32.
	<n_cpu>  	INT			Defines the number of CPUs used for parallel computation on the partitions. 
							This parameter depends on the number of the CPUs of your computer.
							NOTE: max(n_cpu)= (# of CPUs). However, if your memory
							is too low, you should not use all of the CPUs.
	<output_path> STRING    Output directory where the generated normalized files will be placed. 


	options:
	--grp		STRING		The group name of your input file if it is .h5, and the group name of the output
							files.[Default: 'stack'] 
	--up_fg		0/1			If you do not want to upsample your input image in the z-direction
							not to have a better resolution, you should put the 'up_fg=0'.[Default:1]
	--up_z		INT			If you decide to upsample the input image, 'up_z' is the parameter
							that shows the ratio of upsampling in the z-dir.[Default: 4]			
	--fact		FLOAT		Determines the accuracy of the optimization process.[Default: 1e10]	
							Typical values for fact are: 1e12 for low accuracy; 1e7 for moderate accuracy; 
							10.0 for extremely high accuracy. The lower the 'fact' value, the higher the 
							running time. The values in [1e8,1e10] produce good results.
	--cns_fg	0/1			If you want to increase the contrast of the output image, set the cns_fg=1.
							[Default:0]
	--cns_low	FLOAT 
	--cns_high	FLOAT		If you have set cns_fg=1, you can choose which low and high quantiles be removed.
							[Default:(0.00001,0.99999)]
	--lm 		FLOAT 		Determines the ratio of the regularization term to the loss function. Choosing a
							very large number may prevent your code from removing the artifacts. On the other
							hand, a very small value may result in some kinds of memory in the normalized
 							image. Its value depends on the application.[Default: 100]
							Example: For electron microscpy datasets the default value is preferred. For 
							time-lapse photography, it can be in the order of 10^6.
	-h						Help. Print Usage.

-------------------------------------------------------------------------------------------------
Version History

1.0 06/28/2014
	-- source code released
	
	
	
	
	
	



