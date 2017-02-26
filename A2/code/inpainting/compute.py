## CSC320 Winter 2017 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    patch_center = [psiHatP.row(), psiHatP.col()]
    patch_radius = psiHatP.radius()
    filled_window, valid = copyutils.getWindow(filledImage, patch_center, patch_radius)
    con_window = copyutils.getWindow(confidenceImage, patch_center, patch_radius)[0]
    filled_valid_area = filled_window[np.where(valid == True)]/255
    con_valid_area = con_window[np.where(valid == True)]
    filled_con = (filled_valid_area * con_valid_area).sum()
    C = filled_con/valid.sum()
    #########################################
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#
    
def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    patch_center = [psiHatP.row(), psiHatP.col()]
    patch_radius = psiHatP.radius()
    inpaint_window, valid = copyutils.getWindow(inpaintedImage, patch_center, patch_radius)
    inpaint_window = cv.cvtColor(inpaint_window, cv.COLOR_BGR2GRAY)
    filled_window = copyutils.getWindow(filledImage, patch_center, patch_radius)[0]
    filled_area = inpaint_window * (filled_window/255)
    scharrx = cv.Scharr(filled_area, cv.CV_32F, 1, 0)
    scharry = cv.Scharr(filled_area, cv.CV_32F, 0, 1)
    gradient = np.sqrt(scharrx**2 + scharry**2)
    i, j= np.unravel_index(gradient.argmax(),gradient.shape)
    Dy = scharry[i,j]
    Dx = scharrx[i,j]
    #############################
    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    patch_center = [psiHatP.row(), psiHatP.col()]
    patch_radius = psiHatP.radius()
    filled_window, valid = copyutils.getWindow(filledImage, patch_center, patch_radius)
    filled_front_window = copyutils.getWindow(fillFront, patch_center, patch_radius)[0]
    scharrx = cv.Scharr(filled_front_window, cv.CV_32F, 1, 0)
    scharry = cv.Scharr(filled_front_window, cv.CV_32F, 0, 1)
    center_x = filled_front_window.shape[0]//2
    center_y = filled_front_window.shape[1]//2
    center = (center_x, center_y)
    if(np.sum(filled_front_window)==255):
        Ny = None
        Nx = None
    else:
        Ny = -scharrx[center]
        Nx = scharry[center]
    #########################################
    return Ny, Nx