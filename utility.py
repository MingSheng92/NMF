#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ming Sheng Choo
@E-mail: cming0721@gmail.com
@Github: MingSheng92

"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import random 
from PIL import Image
import random
import warnings
warnings.filterwarnings('ignore')

# function to load in dataset and convert into an numpy array 
def load_data(root='data/CroppedYaleB', reduce=4):
    """ 
    Load ORL (or Extended YaleB) dataset to numpy array.
    
    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.
        
    """ 
    images, labels = [], []
    if root=='data/ORL':
        wdth = 92 
        hght = 112
        left = 0
        right = 92
        top = 10
        bottom = 102
    elif root=='data/CroppedYaleB':
        wdth = 168 
        hght = 192
        left = 0
        right = 168
        top = 12
        bottom = 180

    img_size = None
    
    for i, person in enumerate(sorted(os.listdir(root))):
        
        if not os.path.isdir(os.path.join(root, person)):
            continue
        
        for fname in os.listdir(os.path.join(root, person)):    
            
            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue
            
            if not fname.endswith('.pgm'):
                continue
                
            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L') # grey image.

            # reduce computation complexity.
            if reduce > 1:
                img = img.resize([s//reduce for s in img.size])
                w, h = img.size
                right = w
                gap = (h - w)/2.0
                top = 0 + gap
                bottom = h - gap
            
            # TODO: preprocessing.
            img = img.crop((left, top, right, bottom))
            img_size = img.size
            
            # convert image to numpy array.
            img = np.asarray(img).reshape((-1,1))

            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)
    
    # return image array, labels and the latest image size
    return images, labels, img_size
	
# visualize the first n 
def faceGrid(num, img_arr, img_shape):
	title = "First %d faces of the dataset" % num
	# Let's show some centered faces
	plt.figure(figsize=(20, 2))
	plt.suptitle(title, size=16)
	for i in range(num):
		plt.subplot(1, num, i+1)
		plt.imshow(img_arr[:,i * 10].reshape(img_shape), cmap=plt.cm.gray)
		plt.xticks(())
		plt.yticks(())
		
def ResultGrid(num_person, con_img, rec_img, img_shape):
    plt.figure(figsize=(5, 11))
    plt.suptitle("Noise(left) and denoised face(right)", size=16)

    #p_id = 1
    p_id = random.randrange(con_img.shape[1]) 
	
    for j in range(1, (num_person*2 + 1)):
        plt.subplot(num_person, 2, j)
        if (j % 2) == 0:
            plt.imshow(rec_img[:,p_id].reshape(img_shape), cmap=plt.cm.gray)
            p_id = random.randrange(con_img.shape[1])
        else:
            plt.imshow(con_img[:,p_id].reshape(img_shape), cmap=plt.cm.gray)
	
        plt.xticks(())
        plt.yticks(())

# calculate PSNR based on given noised and clean(original) image
def PSNR(original, noisy, peak=100):
    mse = np.mean((original-noisy)**2)
    return 10*np.log10(peak*peak/mse)

# subsampling the overall data into n% of the original data
def subsample(arr, label, n = 0.9, rndm =False):
    '''
    This function subsamples from a 2D numpy array of images where
    the columns depicts instances of an image and the rows depicts the 
    pixel values.
  
    arr: A 2D Numpy Array
    n: the sampling rate, expressed as a percentage
    rndm: if true, will randomly shuffle the original array column-wise
    '''
    #check for shuffle
    if rndm:
        np.random.shuffle(arr.T)

    #determine a range of valid starting indices 
    interval = int(len(arr[0])*(1-n))

    #determine size of sample based on sampling rate
    bound = int(len(arr[0])*(n))

    #random init of starting column index
    i = np.random.randint(0,interval)

    #slice subarray
    subarr = arr[:,i:(i+bound)]
    sublab = label[i:(i+bound)]

    return subarr, sublab
	
# function to add noise to image block (Matrix)
def AddNoiseToMatrix(img_mtx, noise_typ, img_shape):
    # copy image matrix for further processing
    Noised_mtx = img_mtx.copy()
    nMtx = img_mtx.copy()
    # get row and col count from the matrix
    row, col = Noised_mtx.shape
    
    # loop through the numpy array to apply noise
    for c in range(col):
        # reshape array to orginal image structure 
        # before calling the noise function 
        img = Noised_mtx[:,c].reshape(img_shape[0], img_shape[1])
        # apply noise function
        noised_img, noise = noisy(noise_typ, img)
        # flatten the image 
        noised_img = noised_img.reshape((-1,1))
        # update back to the main image matrix
        Noised_mtx[:, c] = noised_img.T
        if noise_typ == "gauss":
            noise = noise.reshape((-1,1))
            nMtx[:, c] = noise.T
        
    # return the applied noise block 
    return Noised_mtx, nMtx

# Add noise to single image
def noisy(noise_typ, image, blk_size=10):
    noise = None
    # apply gaussian noise to the image
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0.1
        sigma = 35
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        
        return noisy, gauss
    
    # apply salt & pepper noise to the image
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.09
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out, noise
    
    # apply poisson noise to the image
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        vals = 0.3
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy, noise
    
    # apply speckle noise to the image
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col) * 0.5        
        noisy = image + image * gauss
        return noisy, noise
    
    # apply block occlusion noise to the image
    elif noise_typ=="block":
        noisy = image.copy()
        numRows, numCols = image.shape
        # get random starting position for x and y 
        randomColumn = random.randint(0,(numCols-blk_size))
        randomRow = random.randint(0,(numRows-blk_size))

        # loop according to block size
        for i in range(blk_size):  
            for j in range(blk_size):
                noisy[randomRow + i, randomColumn + j]=255
        return noisy, noise
		
def inspect_dictionary(Dict, image_shape, n_cols=5):
    """Inspect the dictionary
    Args:
        Dict: learned dictionary array with shape of [feature size, number of components]
        data: str, 'ORL' or 'EYB' or 'AR'.
        reduce: scale factor.
        n_cols: int, number of images shown in each row.
    """
    nrows = Dict.shape[1] // n_cols
    nrows += 1 if Dict.shape[1] % n_cols else 0
    for i in range(nrows):
        plt.figure(figsize=(16, 9))
        for j in range(n_cols):
            plt.subplot(1, n_cols, j+1)
            plt.imshow(Dict[:, i*n_cols+j].reshape(image_shape), cmap=plt.cm.gray)
            plt.axis('off')
        plt.show()