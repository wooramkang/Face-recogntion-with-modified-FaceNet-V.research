1. luminance problem

all the results are affected by luminance around here
so i tried to remove it and make robust features by preprocessing

	a. LAB negative way
	removing luminance way
	http://t9t9.com/60

	=> tons of frame down but good performance


	b. CLAHE (Contrast Limited Adaptive Histogram Equalization)
	inhance the contrast by using histogram
	
	=> good for protecting raw data but median performance

	c. CLAHE (Contrast Limited Adaptive Histogram Equalization) + nagative
	inhance the contrast by using histogram
	
	=> good for protecting raw data but median performance

	d. gamma correction
	felt like erase important feature as well

	=> median median but fast

	e. gamma correction + negative
	gamma between 3-4 = the best

	=> if gamma getting higher, correction rate up but detection rate down

2. ksize error

	OpenCV Error: Assertion failed (type == srcB.type() && srcA.size() == srcB.size()) #1057
    

	to solve this kind of error, you should make sure the sizes of images you'd like to use
	and filters paramaters as well 
	
	