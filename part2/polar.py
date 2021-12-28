#!/usr/local/bin/python3
#
# Authors: Dhruti Patel - dsp3
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

import math
from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import math

# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = image.copy()
    new_image = draw_boundary(new_image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)

def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

emission_pixel_probs = []
airice_boundary_simple = []
icerock_boundary_simple = []
airice_boundary_hmm = []
icerock_boundary_hmm = []
airice_boundary_feedback = []
emission_with_edge = []
transition_probs = []
transition_probs_hf_ai = []
transition_probs_hf_ir = []
epsilon = 0.9

def generate_emission_pixel(image_array):

    transpose_image = image_array.T

    for i in range(len(transpose_image)):
        emission_pixel_probs.append([None]*len(transpose_image[0]))
        for j in range(len(transpose_image[0])):
            emission_pixel_probs[i][j] = math.log(1/phi(transpose_image[i][j]/21))
            # emission_pixel_probs[i][j] = transpose_image[i][j] * epsilon
            # emission_pixel_probs[i][j] = abs(224-transpose_image[i][j])

    for i in range(len(emission_pixel_probs)):
        sum_row = sum(emission_pixel_probs[i])
        for j in range(len(emission_pixel_probs[0])):
            emission_pixel_probs[i][j] = emission_pixel_probs[i][j]/sum_row

def add_edge_strength(edge_strength_array):
    transpose_edge_strength = edge_strength_array.T
    for i in range(len(emission_pixel_probs)):
        emission_with_edge.append([None]*len(emission_pixel_probs[0]))
        for j in range(len(emission_pixel_probs[0])):
            emission_with_edge[i][j] = transpose_edge_strength[i][j]
    
    for i in range(len(emission_with_edge)):
        sum_row = sum(emission_with_edge[i])
        for j in range(len(emission_with_edge[0])):
            emission_with_edge[i][j] = emission_with_edge[i][j]/sum_row

def generate_transition_probs(image_array):
    for i in range(len(image_array)):
        transition_probs.append([None]*len(image_array))
        for j in range(len(image_array)):
            if abs(i-j) == 0:
                dist = abs(i-j)+1
                transition_probs[i][j] = 1/dist
            elif abs(i-j) > 5:
                transition_probs[i][j] = 0.0000001
            else:
                dist = abs(i-j)
                transition_probs[i][j] = 1/dist
    
    for i in range(len(transition_probs)):
        sum_row = sum(transition_probs[i])
        for j in range(len(transition_probs[0])):
            transition_probs[i][j] = transition_probs[i][j]/sum_row

def generate_transition_probs_hf_ai(gtairice, image_array):
    for i in range(len(image_array)):
        transition_probs_hf_ai.append([None]*len(image_array))
        for j in range(len(image_array)):
            # if abs(gtairice[1]-j) == 0:
            #     dist = abs(gtairice[1]-j)+1
            # else:
            #     dist = abs(gtairice[1]-j)
            # transition_probs_hf_ai[i][j] = 1/dist
            if abs(i-gtairice[0]) <= 10:
                if abs(gtairice[1]-j) == 0:
                    dist = abs(gtairice[1]-j)+1
                else:
                    dist = abs(gtairice[1]-j)
                transition_probs_hf_ai[i][j] = 1/dist
            else:
                if abs(i-j) == 0:
                    dist = abs(i-j)+1
                    transition_probs_hf_ai[i][j] = 1/dist
                elif abs(i-j)>5:
                    transition_probs_hf_ai[i][j] = 0.0000001
                else:
                    dist = abs(i-j)
                    transition_probs_hf_ai[i][j] = 1/dist
    
    for i in range(len(transition_probs_hf_ai)):
        sum_row = sum(transition_probs_hf_ai[i])
        for j in range(len(transition_probs_hf_ai[0])):
            transition_probs_hf_ai[i][j] = transition_probs_hf_ai[i][j]/sum_row

def generate_transition_probs_hf_ir(gticerock, image_array):
    for i in range(len(image_array)):
        transition_probs_hf_ir.append([None]*len(image_array))
        for j in range(len(image_array)):
            # if abs(gticerock[1]-j) == 0:
            #     dist = abs(gticerock[1]-j)+1
            # else:
            #     dist = abs(gticerock[1]-j)
            # transition_probs_hf_ir[i][j] = 1/dist
            if abs(i-gticerock[0]) <=10:
                if abs(gticerock[1]-j) == 0:
                    dist = abs(gticerock[1]-j)+1
                else:
                    dist = abs(gticerock[1]-j)
                transition_probs_hf_ir[i][j] = 1/dist
            else:
                if abs(i-j) == 0:
                    dist = abs(i-j)+1
                    transition_probs_hf_ir[i][j] = 1/dist
                elif abs(i-j) > 5:
                    transition_probs_hf_ir[i][j] = 0.0000001
                else:
                    dist = abs(i-j)
                    transition_probs_hf_ir[i][j] = 1/dist
    
    for i in range(len(transition_probs_hf_ir)):
        sum_row = sum(transition_probs_hf_ir[i])
        for j in range(len(transition_probs_hf_ir[0])):
            transition_probs_hf_ir[i][j] = transition_probs_hf_ir[i][j]/sum_row

def get_airice_simple():
    for i in range(len(emission_with_edge)):
        max_probability = max(emission_with_edge[i])
        airice_boundary_simple.append(emission_with_edge[i].index(max_probability))
    #print("airice_boundary_simple",airice_boundary_simple)
    return airice_boundary_simple

def get_icerock_simple():
    for i in range(len(emission_with_edge)):
        airice_boundary = airice_boundary_simple[i]
        curr_col = emission_with_edge[i][airice_boundary+10:]
        max_probability = max(curr_col)
        icerock_boundary_simple.append(curr_col.index(max_probability)+10+airice_boundary)
    #print("icerock_boundary_simple",icerock_boundary_simple)
    return icerock_boundary_simple

def get_airice_hmm():
    viterbi_table = []
    path_track = []
    path = []
    initial_prob = []
    for i in range(len(image_array)):
        initial_prob.append(0.5)
    for i in range(len(initial_prob)):
        initial_prob[i] = initial_prob[i]/sum(initial_prob)

    for i in range(len(emission_with_edge)):
        viterbi_table.append([None]* len(emission_with_edge[0]))
        if i != 0:
            path_track.append([None]* len(emission_with_edge[0]))
        for j in range(len(emission_with_edge[0])):
            if i == 0:
                if emission_with_edge[i][j] == 0:
                    viterbi_table[i][j] = math.log(initial_prob[j]) + math.log(0.0000001)
                else:
                    viterbi_table[i][j] = math.log(initial_prob[j]) + math.log(emission_with_edge[i][j])
            else:
                max_prob = -sys.maxsize - 1
                max_prev_row = None
                for k in range(len(emission_with_edge[0])):
                    if emission_with_edge[i][j] == 0:
                        viterbi_table[i][j] = viterbi_table[i-1][k] + math.log(transition_probs[k][j]) + math.log(0.0000001)
                    else:
                        temp = viterbi_table[i-1][k] + math.log(transition_probs[k][j]) + math.log(emission_with_edge[i][j])
                    if temp > max_prob:
                        max_prob = temp
                        max_prev_row = k
                viterbi_table[i][j] = max_prob 
                path_track[i-1][j] = max_prev_row
    max_last_level = viterbi_table[len(viterbi_table)-1].index(max(viterbi_table[len(viterbi_table)-1]))
    path.append(max_last_level)
    for i in range(len(path_track)-1,-1,-1):
        max_last_level = path_track[i][max_last_level]
        path.append(max_last_level)
    airice_boundary_hmm = path[::-1]
    #print("airice_boundary_hmm",airice_boundary_hmm)
    return airice_boundary_hmm

def get_icerock_hmm(airice_hmm, image_array):
    viterbi_table = []
    path_track = []
    path = []
    initial_prob = []
    for i in range(len(image_array)):
        if i <= airice_hmm[0]+10:
            initial_prob.append(0.0000001)
        else:
            initial_prob.append(0.5)
    for i in range(len(initial_prob)):
        initial_prob[i] = initial_prob[i]/sum(initial_prob)
    for i in range(len(emission_with_edge)):
        viterbi_table.append([None]* len(emission_with_edge[0]))
        if i != 0:
            path_track.append([None]* len(emission_with_edge[0]))
        for j in range(len(emission_with_edge[0])):
            if i == 0:
                if emission_with_edge[i][j] == 0:
                    viterbi_table[i][j] = math.log(initial_prob[j]) + math.log(0.0000001)
                else:
                    viterbi_table[i][j] = math.log(initial_prob[j]) + math.log(emission_with_edge[i][j])
            else:
                max_prob = -sys.maxsize - 1
                max_prev_row = None

                for k in range(len(emission_with_edge[0])):
                    if k <= airice_hmm[i]+10:
                        continue
                    if emission_with_edge[i][j] == 0:
                        temp = viterbi_table[i-1][k] + math.log(transition_probs[k][j]) + math.log(0.0000001)
                    else:
                        temp = viterbi_table[i-1][k] + math.log(transition_probs[k][j]) + math.log(emission_with_edge[i][j])
                    if temp > max_prob:
                        max_prob = temp
                        max_prev_row = k
                viterbi_table[i][j] = max_prob
                path_track[i-1][j] = max_prev_row

    last_airice = airice_hmm[-1]
    max_last_level = None
    max_last_prob = -sys.maxsize - 1
    for i in range(len(viterbi_table[0])):
        if viterbi_table[len(viterbi_table)-1][i] > max_last_prob:
            max_last_prob = viterbi_table[len(viterbi_table)-1][i]
            max_last_level = i
    
    path.append(max_last_level)

    for i in range(len(path_track)-1,-1,-1):
        max_last_level = path_track[i][max_last_level]
        path.append(max_last_level)
    icerock_boundary_hmm = path[::-1]
    #print("icerock_boundary_hmm",icerock_boundary_hmm)
    return icerock_boundary_hmm

def get_airice_feedback(gt_airice):
    viterbi_table = []
    path_track = []
    path = []
    initial_prob = []
    for i in range(len(image_array)):
        initial_prob.append(0.5)
    for i in range(len(initial_prob)):
        initial_prob[i] = initial_prob[i]/sum(initial_prob)

    for i in range(len(emission_with_edge)):
        viterbi_table.append([None]* len(emission_with_edge[0]))
        if i != 0:
            path_track.append([None]* len(emission_with_edge[0]))
        for j in range(len(emission_with_edge[0])):
            if i == 0:
                if emission_with_edge[i][j] == 0:
                    viterbi_table[i][j] = math.log(initial_prob[j]) + math.log(0.0000001)
                else:
                    viterbi_table[i][j] = math.log(initial_prob[j]) + math.log(emission_with_edge[i][j])
            else:
                max_prob = -sys.maxsize - 1
                max_prev_row = None
                for k in range(len(emission_with_edge[0])):
                    if i == gt_airice[0] and k == gt_airice[1]:
                        max_prob = math.log(1)
                        max_prev_row = k
                        continue
                    if emission_with_edge[i][j] == 0:
                        viterbi_table[i][j] = viterbi_table[i-1][k] + math.log(transition_probs_hf_ai[k][j]) + math.log(0.0000001)
                    else:
                        temp = viterbi_table[i-1][k] + math.log(transition_probs_hf_ai[k][j]) + math.log(emission_with_edge[i][j])
                    if temp > max_prob:
                        max_prob = temp
                        max_prev_row = k
                viterbi_table[i][j] = max_prob 
                path_track[i-1][j] = max_prev_row
    max_last_level = viterbi_table[len(viterbi_table)-1].index(max(viterbi_table[len(viterbi_table)-1]))
    path.append(max_last_level)
    for i in range(len(path_track)-1,-1,-1):
        max_last_level = path_track[i][max_last_level]
        path.append(max_last_level)
    airice_boundary_feedback = path[::-1]
    #print("airice_boundary_feedback",airice_boundary_feedback)
    return airice_boundary_feedback

def get_icerock_feedback(gt_icerock, airice_feedback):
    viterbi_table = []
    path_track = []
    path = []
    initial_prob = []
    for i in range(len(image_array)):
        if i <= airice_hmm[0]+10:
            initial_prob.append(0.0000001)
        else:
            initial_prob.append(0.5)
    for i in range(len(initial_prob)):
        initial_prob[i] = initial_prob[i]/sum(initial_prob)
    for i in range(len(emission_with_edge)):
        viterbi_table.append([None]* len(emission_with_edge[0]))
        if i != 0:
            path_track.append([None]* len(emission_with_edge[0]))
        for j in range(len(emission_with_edge[0])):
            if i == 0:
                if emission_with_edge[i][j] == 0:
                    viterbi_table[i][j] = math.log(initial_prob[j]) + math.log(0.0000001)
                else:
                    viterbi_table[i][j] = math.log(initial_prob[j]) + math.log(emission_with_edge[i][j])
            else:
                max_prob = -sys.maxsize - 1
                max_prev_row = None

                for k in range(len(emission_with_edge[0])):
                    if k <= airice_hmm[i]+10:
                        continue
                    if i == gt_icerock[0] and k == gt_icerock[1]:
                        max_prob = math.log(1)
                        max_prev_row = k
                        continue
                    if emission_with_edge[i][j] == 0:
                        temp = viterbi_table[i-1][k] + math.log(transition_probs_hf_ir[k][j]) + math.log(0.0000001)
                    else:
                        temp = viterbi_table[i-1][k] + math.log(transition_probs_hf_ir[k][j]) + math.log(emission_with_edge[i][j])
                    if temp > max_prob:
                        max_prob = temp
                        max_prev_row = k
                viterbi_table[i][j] = max_prob
                path_track[i-1][j] = max_prev_row

    last_airice = airice_hmm[-1]
    max_last_level = None
    max_last_prob = -sys.maxsize - 1
    for i in range(len(viterbi_table[0])):
        if viterbi_table[len(viterbi_table)-1][i] > max_last_prob:
            max_last_prob = viterbi_table[len(viterbi_table)-1][i]
            max_last_level = i
    
    path.append(max_last_level)

    for i in range(len(path_track)-1,-1,-1):
        max_last_level = path_track[i][max_last_level]
        path.append(max_last_level)
    icerock_boundary_feedback = path[::-1]
    #print("icerock_boundary_feedback",icerock_boundary_feedback)
    return icerock_boundary_feedback

# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength_array = edge_strength(input_image)
    imageio.imwrite('edges.png', uint8(255 * edge_strength_array / (amax(edge_strength_array))))

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.

    # Train the model
    generate_emission_pixel(image_array)   
    add_edge_strength(edge_strength_array)
    generate_transition_probs(image_array)
    generate_transition_probs_hf_ai(gt_airice, image_array)
    generate_transition_probs_hf_ir(gt_icerock, image_array)
    
    # Calculate the boundaries for test images
    airice_simple = get_airice_simple()
    airice_hmm = get_airice_hmm()
    airice_feedback= get_airice_feedback(gt_airice)

    icerock_simple = get_icerock_simple()
    icerock_hmm = get_icerock_hmm(airice_hmm, image_array)
    icerock_feedback= get_icerock_feedback(gt_icerock,airice_feedback)

    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
