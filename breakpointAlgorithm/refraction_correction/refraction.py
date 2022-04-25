
"""Refraction correction script

This script performs the preprocessing task of correcting for refraction
from the distortion introduced by the glass of the observation section,
using methods outline in "3D visualization of two-phase flow in the
micro-tube by a simple but effective method" by Fu et al.

This file can also be imported as a module and contains the following
functions:

    * refraction_single_image - undistorts the input image and returns the output
    * main - undistort a batch of images and save the outputs
"""

# Imports
from refraction_maths import refraction_maths
import cv2
import numpy as np
import glob

def refraction_single_image(img):
    """Undistorts a single image and saves the output

    Parameters
    ----------
    img: input image to be undistorted

    Returns
    -------
    img_output
        the undistorted input image
    """

    # Constants
    # Refractive indices of mediums
    n0=1
    n1=1.473
    n2=1.33
    # Inner and outer radius of tube section
    ri=4e-3
    ro=6e-3
    # Pixel to mm rescale
    scale=(1e-3)/18 #18 pixels for 1mm

    # Formatting for output image
    rows, cols = img.shape
    img_output = np.zeros(img.shape, dtype=img.dtype)

    # Loop through each pixel, and offset it to correct for refraction
    for i in range(rows):
        for j in range(cols):
            s1=(abs(j-194))*scale
            if(s1<ro):
                s=refraction_maths(s1,ri,ro,n0,n1,n2)
                if(j<194):
                    offset_x=int(round((abs(s1-s))/scale))
                else:
                    offset_x = -int(round((abs(s1 - s)) / scale))
            else:
                offset_x=0

            if j+offset_x < rows:
                img_output[i,j] = img[i,(j+offset_x)%cols]
            else:
                img_output[i,j] = 0
    return img_output

def main():
    """Batch distortion correction

    A batch of images are loaded in, corrected, and saved.
    """

    # Load in input images
    images=[cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob("python_pre/*.bmp")]

    # Loop through and correct images, appending the outputs to images_ref
    images_ref=[] 
    for i in range(len(images)):
        images_ref.append(refraction_single_image(images[i]))
    
    # Save corrected images
    for i in range(len(images)):
        cv2.imwrite("processed_data/ref{}.bmp".format(i),images_ref[i])


if __name__=="__main__":
    main()
