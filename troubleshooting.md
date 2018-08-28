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
	
3. on image processing

##mode, rawmode = _fromarray_typemap[typekey]

	-- a/PIL/Image.py
+++ b/PIL/Image.py
@@ -2207,10 +2207,14 @@ _fromarray_typemap = {
     # ((1, 1), "|b1"): ("1", "1"), # broken
     ((1, 1), "|u1"): ("L", "L"),
     ((1, 1), "|i1"): ("I", "I;8"),
-    ((1, 1), "<i2"): ("I", "I;16"),
-    ((1, 1), ">i2"): ("I", "I;16B"),
-    ((1, 1), "<i4"): ("I", "I;32"),
-    ((1, 1), ">i4"): ("I", "I;32B"),
+    ((1, 1), "<u2"): ("I", "I;16"),
+    ((1, 1), ">u2"): ("I", "I;16B"),
+    ((1, 1), "<i2"): ("I", "I;16S"),
+    ((1, 1), ">i2"): ("I", "I;16BS"),
+    ((1, 1), "<u4"): ("I", "I;32"),
+    ((1, 1), ">u4"): ("I", "I;32B"),
+    ((1, 1), "<i4"): ("I", "I;32S"),
+    ((1, 1), ">i4"): ("I", "I;32BS"),
     ((1, 1), "<f4"): ("F", "F;32F"),
     ((1, 1), ">f4"): ("F", "F;32BF"),
     ((1, 1), "<f8"): ("F", "F;64F"),


4. for early stopping

        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        callbacks = [early, lr_reducer, checkpoint]
        SupResolution.fit(x_train,
                    x_train,
                    validation_data=(x_test, x_test),
                    epochs=30,
                    batch_size=batch_size,
                    callbacks=callbacks)