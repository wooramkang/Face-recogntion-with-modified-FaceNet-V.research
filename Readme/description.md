


Face recognition research


1. environment && Dependency about research

OS : ubuntu 16.04
language : python 3.5
package : 
	a. CUDA : 9.0
	b. cuDNN : 7.1.4
	c. opencv
	d. tensorflow : 1.9.0
	e. dlib : 19.15.0
	f. numpy
	g. sklearn
	...


2. Dataset for training and testing
location : /Dataset_Image_Face

	a. LFW http://vis-www.cs.umass.edu/lfw/
	Labeled Faces in the Wild
	
	13233 images
	5749 people
	1680 people with two or more


	b. VGG VGGFace2 http://zeus.robots.ox.ac.uk/vgg_face2/
	Visual Geometry group

	3.31 Million images
	9131 people
	87 - 850 people with two or more

	
	c. VGG VGGFace1 http://www.robots.ox.ac.uk/~vgg/data/vgg_face/
	Visual Geometry group

	2622 people

	cf. tiny version of VGGFace2

	caution. each group has diffenrt kinds of image'size for each. so make sure the sizes of groups to be as same as one's size of those.
		 
		 mainly, a. LFW & b. VGGFace2 be used


3. trial algorithms	 A by A

ToDo list :

	a. detection + recognition

		1. Deep Face

		2. Open Face

		3. VGG __face recognition

		4. Openbr

		5. facenet
	
		6. face-everthing

	b. detection only
	
		1. tiny face

		2. MTCNN_Face_detection

	c. recognition only

		1. Fisherfaces

		2. shanren7_ real time face recogntion

		3. LBPH Algorithm
		
		4. insightface

	d. preprocessing

		1. remove light on LAB colour system

		2. CLAHE

		3. gamma correction

    d-prime. preprocessing with autoencoder
    
		1. VAE

		2. denosing AE

Done list : 

	a. FaceNet testing

	b. FaceNet papers review

	c. preprocessing by vision algorithm

	d. VAE papers review

