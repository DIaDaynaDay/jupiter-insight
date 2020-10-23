import numpy as np
import os
import cv2 
from glob import glob
from astropy.io import fits
from astropy.visualization import *
from photutils import *

import scipy.misc

#function to get the data from fits files
def get_data(file):
    foo = fits.open(file,ignore_missing_end=True,lazy_load_hdus = False)
    return foo[0].data

#open multiple fits without reading the data
def open_fits(directory):
    files =[]
    for filename in os.listdir(directory):
        if filename.endswith(".fits"):
              foo = fits.open(directory + '/' + filename, ignore_missing_end=True)[0]
              files.append(foo)
    return files

#function to calculate bias average value and standard deviation
#code credit: https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
def bias_calc(directory):
    averages = []
    stds = []
    for filename in os.listdir(directory):
        if filename.endswith(".fits"):
              averages.append(get_data(directory + '/' + filename).mean())
              stds.append(get_data(directory + '/' + filename).std())
    avg = np.array([averages])
    stan = np.array([stds])
    mean = avg.mean()
    std = stan.mean()
    return mean,std

#put all our images in a single array of arrays. Then we can access each individual image using index
def multiple_fits(directory):    
    images = []
    for filename in os.listdir(directory):
        images.append(get_data(directory + '/' + filename))
    data = np.array(images)
    return data

#calculate centroids of the rest of the images and put them in arrays
def calculate_centroids(data):
    xcoor = []
    ycoor = []
    for child in data[1:]:
        x,y = centroid_quadratic(child, fit_boxsize = 1000)
        xcoor.append(x)
        ycoor.append(y)
    xcen = np.array(xcoor)
    ycen = np.array(ycoor)
    return xcen,ycen


#constructing a linear transformation matrices to be applied to source images
def transformation_matrices(xoffset,yoffset,):
    m = []
    for i in range(len(xoffset)):
        foo = np.array([[1, 0, xoffset[i]], [0, 1, yoffset[i]]]) 
        m.append(foo)
    M = np.array(m)
    return M

#shifting the rest of the images by offsets in comparison with the reference image
#code credit https://www.geeksforgeeks.org/image-processing-in-python-scaling-rotating-shifting-and-edge-detection/
def image_registration(data,M,rows,cols):
    res = []
    for i in range(1,len(data)):
        foo = cv2.warpAffine(data[i], M[i-1], (cols, rows)) 
        res.append(foo)
    out = np.array(res)
    return out

#if we need to actually write the files back to fits. Headers are not preserved unfortunately, might solve this later if needed
def write_to_fits(directory,data,name):
    fitslist = []
    for i in range(len(data)):
        foo = fits.PrimaryHDU(data[i],header=None)
        fitslist.append(foo)
    for i in range(len(fitslist)):    
        fitslist[i].writeto(directory + '/' + name + str(i) + '.fits')          
 
           
#code credit https://properimage.readthedocs.io/en/latest/tutorial/Tutorial04.html
#get a image normalisation to z scale and asinh stretch
def norm_zscale_asinh(data):
    norms = []
    for child in data:
        norm = ImageNormalize(child, interval=ZScaleInterval(), stretch=AsinhStretch())
        norms.append(norm)
    return norms

#code credit https://properimage.readthedocs.io/en/latest/tutorial/Tutorial04.html
#get a image normalisation to z scale and asinh stretch
def norm_zscale_asinh(data):
    norms = []
    for child in data:
        norm = ImageNormalize(child, interval=ZScaleInterval(), stretch=AsinhStretch())
        norms.append(norm)
    return norms

#code credit https://photutils.readthedocs.io/en/stable/detection.html
#detects all the moons positions, does not differentiate between them
def find_moons(data,mask,threshold):
    moon_list = []
    for child in data:
        tbl = find_peaks(child, threshold, box_size=11, mask = mask)
        table = tbl.as_array()
        moon_list.append(table[0])
        if len(tbl) == 2:
            moon_list.append(table[1])
        if len(tbl) == 3:
            moon_list.append(table[1])
            moon_list.append(table[2])
        if len(tbl) == 4:
            moon_list.append(table[1])
            moon_list.append(table[2])
            moon_list.append(table[3])
    moons = np.array(moon_list)

    
    return moons  

def find_moons_2(data,mask,threshold):
    moon_list = []
    for child in data:
        segm = detect_souces(child, threshold, mask = mask)
        props = source_properties(data, segm)
        tbl = properties_table(props)
        table = tbl.as_array()
        moon_list.append(table[0])
        if len(tbl) == 2:
            moon_list.append(table[1])
        if len(tbl) == 3:
            moon_list.append(table[1])
            moon_list.append(table[2])
        if len(tbl) == 4:
            moon_list.append(table[1])
            moon_list.append(table[2])
            moon_list.append(table[3])
    moons = np.array(moon_list)

    
    return moons  
        