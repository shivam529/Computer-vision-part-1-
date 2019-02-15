#!/usr/bin/env python3
# Assignment - 1
# References:
# 1. Canny Edge detection logic : http://www.aishack.in/tutorials/canny-edge-detector/
# 2. Hough Transform understanding : https://alyssaq.github.io/2014/understanding-hough-transform/

from PIL import Image
import numpy as np
import math
import sys

if __name__ == "__main__":
    
    image = sys.argv[1]
    output_image = sys.argv[2]
    output_txt = sys.argv[3]

    print("\nRecognizing form.jpg...\n")
    # Open the given Image
    im = Image.open(image)

    # Output Image to draw a box around the choosen answer
    # Converting it to color Image
    output_im = im.convert("RGB")
    
    # Converting to greyscale
    if im.mode != "L":
        grey = np.array(im).T
        R = grey[...,0]
        G = grey[...,1]
        B = grey[...,2]
        im = Image.fromarray((0.21*R + 0.72*G + 0.07*B).T)
        #im = im.convert('L')
    
    ## -----SOBEL OPERATOR-----
    # The X and Y Gradient matrices for Sobel Operation
    X_gradient = [[-1/8, 0, 1/8],
                  [-2/8, 0, 2/8],
                  [-1/8, 0, 1/8]]
    Y_gradient = [[1/8, 2/8, 1/8],
                  [0, 0, 0],
                  [-1/8, -2/8, -1/8]]

    # 2-D List to store the gradient angles from sobel operation for the image
    grad_angles = [[-1 for x in range(im.width)] for y in range(im.height)]

    # New Image for storing the output image of sobel operation
    grad_im = Image.new("L", (im.width, im.height), 1)

    # Building a binary image as well, so as to compare and make decisions
    bin_im = Image.new("1", (im.width, im.height), 0)
    
    # Nested Loop performing sobel operation over the image
    for i in range(im.width):
        for j in range(im.height):

            # Making the boundary pixels black
            if i in [0,(im.width - 1)] or j in [0,(im.height - 1)]:
                grad_im.putpixel((i,j), 1)

            # Calculating X and Y gradient
            else:
                X = im.getpixel((i-1, j-1)) *  X_gradient[0][0] + im.getpixel((i-1, j)) *  X_gradient[0][1] + im.getpixel((i-1, j+1)) *  X_gradient[0][2] +\
                    im.getpixel((i, j-1)) *  X_gradient[1][0] + im.getpixel((i, j)) *  X_gradient[1][1] + im.getpixel((i, j+1)) *  X_gradient[1][2] +\
                    im.getpixel((i+1, j-1)) *  X_gradient[2][0] + im.getpixel((i+1, j)) *  X_gradient[2][1] + im.getpixel((i+1, j+1)) *  X_gradient[2][2]
                
                Y = im.getpixel((i-1, j-1)) *  Y_gradient[0][0] + im.getpixel((i-1, j)) *  Y_gradient[0][1] + im.getpixel((i-1, j+1)) *  Y_gradient[0][2] +\
                    im.getpixel((i, j-1)) *  Y_gradient[1][0] + im.getpixel((i, j)) *  Y_gradient[1][1] + im.getpixel((i, j+1)) *  Y_gradient[1][2] +\
                    im.getpixel((i+1, j-1)) *  Y_gradient[2][0] + im.getpixel((i+1, j)) *  Y_gradient[2][1] + im.getpixel((i+1, j+1)) *  Y_gradient[2][2]
                
                # Calculating the gradient Magnitude using X and Y gradient
                Z = int(math.sqrt((X*X)+(Y*Y)))

                # Thresholding the gradient magnitude so as to remove noise if any
                if Z > 70:

                    # Building Image using the gradient magnitudes
                    grad_im.putpixel((i,j), Z)

                    # Binary Image
                    bin_im.putpixel((i,j), 1)

                    # Calculating the corresponding gradient angles (making sure not run into division by zero error)
                    # Converting negative angles to positive by taking their absolute values
                    if X!=0:
                        grad_angles[j][i] = abs(int(math.degrees(np.arctan(Y/X))))
                    else:
                        grad_angles[j][i] = abs(int(math.degrees(np.arctan(Y/1))))
                else:
                    grad_im.putpixel((i,j), 1)

    # Sharpening gausian filter used from LAB-1 Task
    # Reason for using sharpening filter : We ran canny edge detection on the gradient magnitude image, but the edges were not 
    # so great, then we tried blurring the image before canny edge detection but of no use, 
    # finally sharpening the image made drastic improvements for canny edge detection
    Gaussian = [-0.003, -0.013, -0.022, -0.013, -0.003, \
        -0.013, -0.059, -0.097, -0.059, -0.013, \
        -0.022, -0.097, (1 + 1) - 0.159, -0.097, -0.022, \
        -0.013, -0.059, -0.097, -0.059, -0.013, \
        -0.003, -0.013, -0.022, -0.013, -0.00]

    # Simple list used for cell representation while performing convolution with Sharpening filter
    C = [-2, -1, 0, 1, 2]
    
    # New image for the sharpened output image
    aft_blur = Image.new("L", (im.width, im.height), 1)
    
    # Sharpening 
    for i in range(im.width):
        for j in range(im.height):
            if i in [0,1,(im.width - 1),(im.width - 2)] or j in [0,1,(im.height - 1),(im.height - 2)]:
                aft_blur.putpixel((i,j), 1)
            else:
                f, P = 0, 0
                for x in C:
                    for y in C:
                        P += grad_im.getpixel((i+x, j+y)) * Gaussian[f]
                        f += 1
                aft_blur.putpixel((i,j), int(P))
    
    ## ------CANNY EDGE DETECTION------
    # New image for canny edge detection
    canny_im = Image.new("L", (im.width,im.height), 1)
    
    # Performing Canny Edge detection
    for i in range(im.width):
        for j in range(im.height):

            # Taking care of the boundary pixels
            if i in [0,1,(im.width - 1),(im.width - 2)] or j in [0,1,(im.height - 1),(im.height - 2)]:
                canny_im.putpixel((i,j), 1)

            # In this image we are trying to build a grid around the choice boxes so as to detect their position on the image
            # Thus we need only horizontal and vertical lines
            # Hence checking the pixels which have gradient angles 0 or 90 and ignoring the rest
            else:

                # If gradient angle is 0 degrees, then the range is set to 0-20 degrees
                if grad_angles[j][i] < 20:

                    # Non maximum suppression
                    # Compare the edge pixel with its neigbouring pixels and using it only if its maximum among them
                    if aft_blur.getpixel((i,j)) > aft_blur.getpixel((i,j-1)) and aft_blur.getpixel((i,j)) > aft_blur.getpixel((i,j+1)):
                        canny_im.putpixel((i,j), 255)
                    else:
                        canny_im.putpixel((i,j), 1)

                # If gradient angle is 90 degrees, then the range is set to 65-115 degrees
                elif grad_angles[j][i] > 65 and grad_angles[j][i] < 115:
                    if aft_blur.getpixel((i,j)) > aft_blur.getpixel((i-1,j)) and aft_blur.getpixel((i,j)) > aft_blur.getpixel((i+1,j)):
                        canny_im.putpixel((i,j), 255)
                    else:
                        canny_im.putpixel((i,j), 1)
                else:
                    canny_im.putpixel((i,j), 1)
    
    # New image (copy of original image) for drawing the grid using hough transform for line detection
    lines = im

    # ------HOUGH TRANSFORM------
    # Accumulator arrays for voting 
    # Again since we are only concerned with horizontal and vertical lines we will perform voting only for 0 and 90 degree angles
    # Accumulator for vertical lines, 
    # Lenght of this is the width of image, since the rho value in this case will varry from zero to width of the image, 
    # considering top-left corner of the image as origin
    acc0 = [0 for i in range(im.width)]

    # Accumulator for horizontal lines
    # Similarly for horizontal lines will have rho values from 0 to height of image
    acc90 = [0 for i in range(im.height)]

    # Performing Voting
    # Leaving a gap of 600 pixels from top, as we are concerned only with choice boxes 
    for i in range(im.width):
        for j in range(600,im.height):

            # Only Edge pixels get to vote
            if canny_im.getpixel((i,j)) == 255:

                # Calculate rho values
                rho_0 = i*np.cos(np.deg2rad(0)) + j*np.sin(np.deg2rad(0))
                rho_90 = i*np.cos(np.deg2rad(90)) + j*np.sin(np.deg2rad(90))

                # Increment the count of the respective rho value in the accumulator array
                acc0[abs(int(rho_0))] += 1
                acc90[abs(int(rho_90))] += 1

    # Getting the rho value with the max votes
    # for vertical lines
    max_h = 0
    for i in range(im.width):
        if acc0[i] > max_h:
            max_h = acc0[i]

    # for horizontal lines
    max_r = 0
    for i in range(im.height):
        if acc90[i] > max_r:
            max_r = acc90[i]

    # Initial threshold for detecting lines
    # That is, rho values having votes greater than these below thresholds of max values will be counted as lines
    Vertical_threshold = 0.7
    Horizontal_threshold = 0.7

    # List to store rho values of actual lines, which will be detected after applying above threshold
    H_rhos = list()
    R_rhos = list()

    # Here we found that for each edge the canny edge detection was marking two edges
    # And we know that we have 5 X 3 number of boxes along the width of image and each box has two vertical edges
    # which gives 5 X 3 X 2 X 2 = 60 
    # Thus we will need atleast 60 vertical lines to be detected
    if len(H_rhos) < 60:

        # So we are reducing the threshold and finding vertical lines again and again until we find 60 vertical lines
        # If there was just one image we could have set just one threshold 
        # but since there are multiple images with different amounts of noise, the threhold varies
        # thus we came up with this, to dynamically calculate threshold
        while len(H_rhos) < 60:

            # Thresholding rho values
            H_rhos = [i for i in range(int(im.width)) if (acc0[i] > (Vertical_threshold*max_h))]
            Vertical_threshold -= 0.05

    # Similarly for horizontal lines
    if len(R_rhos) < 116:
        while len(R_rhos) < 116:

            # Thresholding rho values
            R_rhos = [i for i in range(int(im.height)) if (acc90[i] > (Horizontal_threshold*max_r))]
            Horizontal_threshold -= 0.05
    
    # Logic to remove double lines for the same edge, as canny had detected two edges for one actual edge
    # We found boxes were at leas 30 pixels wide so removing vertical lines which are within range of 30 pixels of each other
    # If you want to see the grid of lines generated, you can show the "lines image"
    prev_r = 0  
    revised_H_rhos = list()
    for r in H_rhos:
        if prev_r != 0:
           if r - prev_r < 25:
               continue
        prev_r = r
        revised_H_rhos.append(r)
        for j in range(im.height):
            lines.putpixel((r,j), 1)

    # Similary we found the boxes to  separated from each other atleast 12 pixels away in vertical direction
    prev_h = 0
    revised_R_rhos = list()
    for h in R_rhos:
        if prev_h != 0:
            if h - prev_h < 12:
                continue
        prev_h = h
        revised_R_rhos.append(h)
        for i in range(im.width):
            lines.putpixel((i,h), 1)

    # Function to check if a given box was marked by the student
    def check_mark(img,x1,x2,y1,y2):
        count = 0
        no_of_pixels = 0
        for i in range(x1,x2):
            for j in range(y1,y2):

                # Counting the total number of pixels in the box
                no_of_pixels += 1
                if (img.getpixel((i,j)) < 120):

                    # Counting the number of marked pixels
                    count += 1

        # Return marked if the count of marked pixels is greater that 50% of total number of pixels
        if (count/no_of_pixels) > 0.5:
            return True
        else:
            return False
    
    # Output file to write answers
    output = open(output_txt, "w+")
    
    
    # In case the line above the boxes was detected like in images a-48.jpg and b-13.jpg remove it
    if revised_R_rhos[1] - revised_R_rhos[0] < 25:
        revised_R_rhos.pop(0)

    # Iterating along the questions and using the grid plotted earlier to detect the position of the choices
    V = 0
    H = 0
    for q in range(1,86):

        # Some logic which we came up after a lot of trials to make sure we dont have missing lines
        # if there are missing lines in the grid then approximate them
        if H >= len(revised_R_rhos):
            revised_R_rhos.append(revised_R_rhos[H-1]+15)
        if (H+1) >= len(revised_R_rhos):
            revised_R_rhos.append(revised_R_rhos[H]+30)
        if revised_R_rhos[H+1] - revised_R_rhos[H] < 20:
            revised_R_rhos.pop(H+1)
        if revised_R_rhos[H+1] - revised_R_rhos[H] > 39:
            revised_R_rhos.insert(H+1, revised_R_rhos[H]+30)
        if q!=29 and q!=58:
            if (H+2) >= len(revised_R_rhos):
                revised_R_rhos.append(revised_R_rhos[H+1]+15)
            if revised_R_rhos[H+2] - revised_R_rhos[H+1] > 25:
                revised_R_rhos.insert(H+2, revised_R_rhos[H+1]+15)
        if revised_H_rhos[V+1] - revised_H_rhos[V] < 25:
            revised_H_rhos.pop(V+1)

        # Call the check mark function to check if the respective choice was marked
        A = check_mark(im,revised_H_rhos[V], revised_H_rhos[V+1], revised_R_rhos[H], revised_R_rhos[H+1])
        B = check_mark(im,revised_H_rhos[V+2], revised_H_rhos[V+3], revised_R_rhos[H], revised_R_rhos[H+1])
        C = check_mark(im,revised_H_rhos[V+4], revised_H_rhos[V+5], revised_R_rhos[H], revised_R_rhos[H+1])
        D = check_mark(im,revised_H_rhos[V+6], revised_H_rhos[V+7], revised_R_rhos[H], revised_R_rhos[H+1])
        E = check_mark(im,revised_H_rhos[V+8], revised_H_rhos[V+9], revised_R_rhos[H], revised_R_rhos[H+1])
        
        # Check which one was marked and then write to the output file 
        output.write(str(q) + " ")
        if A == True:
            output.write("A")

            # Draw a green box along the marked choice
            for i in range(revised_H_rhos[V],revised_H_rhos[V+1]):
                output_im.putpixel((i,revised_R_rhos[H]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+3), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+3), (0,128,0))
            for j in range(revised_R_rhos[H],revised_R_rhos[H+1]):
                output_im.putpixel((revised_H_rhos[V],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V]+3,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+1],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+1]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+1]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+1]+3,j), (0,128,0))
        if B == True:
            output.write("B")
            for i in range(revised_H_rhos[V+2],revised_H_rhos[V+3]):
                output_im.putpixel((i,revised_R_rhos[H]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+3), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+3), (0,128,0))
            for j in range(revised_R_rhos[H],revised_R_rhos[H+1]):
                output_im.putpixel((revised_H_rhos[V+2],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+2]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+2]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+2]+3,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+3],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+3]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+3]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+3]+3,j), (0,128,0))
        if C == True:
            output.write("C")
            for i in range(revised_H_rhos[V+4],revised_H_rhos[V+5]):
                output_im.putpixel((i,revised_R_rhos[H]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+3), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+3), (0,128,0))
            for j in range(revised_R_rhos[H],revised_R_rhos[H+1]):
                output_im.putpixel((revised_H_rhos[V+4],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+4]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+4]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+4]+3,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+5],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+5]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+5]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+5]+3,j), (0,128,0))
        if D == True:
            output.write("D")
            for i in range(revised_H_rhos[V+6],revised_H_rhos[V+7]):
                output_im.putpixel((i,revised_R_rhos[H]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+3), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+3), (0,128,0))
            for j in range(revised_R_rhos[H],revised_R_rhos[H+1]):
                output_im.putpixel((revised_H_rhos[V+6],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+6]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+6]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+6]+3,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+7],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+7]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+7]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+7]+3,j), (0,128,0))
        if E == True:
            output.write("E")
            for i in range(revised_H_rhos[V+8],revised_H_rhos[V+9]):
                output_im.putpixel((i,revised_R_rhos[H]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H]+3), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+1), (0,128,0))
                output_im.putpixel((i,revised_R_rhos[H+1]+2), (0,128,0)), output_im.putpixel((i,revised_R_rhos[H+1]+3), (0,128,0))
            for j in range(revised_R_rhos[H],revised_R_rhos[H+1]):
                output_im.putpixel((revised_H_rhos[V+8],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+8]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+8]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+8]+3,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+9],j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+9]+1,j), (0,128,0))
                output_im.putpixel((revised_H_rhos[V+9]+2,j), (0,128,0)), output_im.putpixel((revised_H_rhos[V+9]+3,j), (0,128,0))
        output.write("\n")

        H += 2
        if q == 29:
            H = 0
            V = 10
        if q == 58:
            H = 0
            V = 20

    output_im.save(output_image)
            

