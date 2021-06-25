**********************************
Background
**********************************


Why data augmentation ?
#######################

Even though always larger data sets are now available, the lack of labeled data remains a tremendous issue in many fields of application. Among others, a good example is healthcare where practitioners have to deal most of the time with (very) low sample sizes (think of small patient cohorts) along with very high dimensional data (think of neuroimaging data that are 3D volumes with millions of voxels). Unfortunately, this leads to a very poor representation of a given population and makes classical statistical analyses unreliable. Meanwhile, the remarkable performance of algorithms heavily relying on the deep learning framework has made them extremely attractive and very popular. However, such results are strongly conditioned by the number of training samples since such models usually need to be trained on huge data sets to prevent over-fitting or to give statistically meaningful results. For instance, the easiest way to do this on images is to apply simple transformations such as the addition of Gaussian noise, cropping or padding, and assign the label of the initial image to the created ones. 


Limitations of classic DA
#######################
While such augmentation techniques have revealed very useful, they remain strongly data dependent and limited. Some transformations may indeed be uninformative or even induce bias. 


.. centered::
    |pic1| apply rotation |pic2|



.. |pic1| image:: imgs/nine_digits.png
    :width: 30%


.. |pic2| image:: imgs/nine_digits-rot.png
    :width: 30%


For instance, a digit representing a 6 which gives a 9 when rotated. While assessing the relevance of augmented data may be quite straightforward for simple data sets, it reveals very challenging for complex data and may require the intervention of an *expert* assessing the degree of relevance of the proposed transformations. 

Generative models: A new hope
#######################

The recent rise in performance of generative models such as GAN or VAE has made them very attractive models to perform DA. However, the main limitation to a wider use of these models is that they most of the time produce blurry and fuzzy samples. This undesirable effect is even more emphasized when they are trained with a small number of samples which makes them very hard to use in practice to perform DA in the high dimensional (very) low sample size (HDLSS) setting.


This is why Pyraug was born!