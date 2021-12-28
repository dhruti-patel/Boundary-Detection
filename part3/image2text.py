#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Anita Badrinarayanan (anbadrin)
#          Nikhil Kamble (nkamble)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

char_count = {}
train_text = []
init_prob = {}
prior_prob = {}
initial_char_count = {}
transition_count = {}
transition_probs = {}
character_count = {}
emission_count = {}
emission_probs = {}
correlation_matrix = {}

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # to fit in the given character dimension
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####

# main program
def read_data(fname):
    train_text = []
    file = open(fname, 'r')
    for line in file:
        temp = []
        for char in line:
            temp.append(char)
        temp.pop(-1)
        train_text.append(temp)
    return train_text

def total_count_of_given_character(train_text):
    total_characters = 0
    for i in range(len(train_text)):
        total_characters += len(train_text[i])
        for j in range(len(train_text[i])):
            if(j == 0):
                if(train_text[i][j] in initial_char_count.keys()):
                    initial_char_count[train_text[i][j]] = initial_char_count[train_text[i][j]] + 1
                else:
                    initial_char_count[train_text[i][j]] = 1
            if train_text[i][j] in char_count.keys():
                char_count[train_text[i][j]] = char_count[train_text[i][j]]+1
            else:
                char_count[train_text[i][j]] = 1
    return total_characters

def build_transition_count(char1, char2):
    if char1 not in transition_count:
        transition_count[char1] = {char2: 1}
    elif char2 not in transition_count[char1]:
        transition_count[char1][char2] = 1
    else:
        transition_count[char1][char2] += 1

def build_transition_probs():
    for char1 in transition_count:
        for char2 in transition_count[char1]:
            if char1 not in transition_probs:
                transition_probs[char1] = {char2: transition_count[char1][char2] / character_count[char1]}
            else:
                transition_probs[char1][char2] = transition_count[char1][char2] / character_count[char1]

def calculate_init_probabilities(train_text):
    for char in initial_char_count.keys():
        init_prob[char] = initial_char_count[char]/len(train_text)

def calculate_prior_probabilities(total_characters):
    for char in char_count.keys():
        prior_prob[char] = char_count[char]/total_characters
    prior_prob["\""] = 0.0000001

def compare_images(train_letters,test_letters):
    for train_char in train_letters.keys():
        curr_train_char = train_letters[train_char]
        for test_char_pos in range(len(test_letters)):
            char_match = 0
            space_match = 0
            unmatch = 0
            curr_test_char = test_letters[test_char_pos]
            for i,j in zip(range(len(curr_train_char)),range(len(curr_test_char))):
                for k,l in zip(range(len(curr_train_char[0])),range(len(curr_test_char[0]))):
                    if curr_train_char[i][k] == curr_test_char[j][l] and curr_test_char[j][l] == '*':
                        char_match = char_match+1
                    elif curr_train_char[i][k] == curr_test_char[j][l] and curr_test_char[j][l] == ' ':
                        space_match = space_match+1
                    elif curr_train_char[i][k] != curr_test_char[j][l]:
                        unmatch = unmatch+1
                if train_char in correlation_matrix.keys():
                    if test_char_pos in correlation_matrix[train_char].keys():
                        correlation_matrix[train_char][test_char_pos] = (math.pow(0.7,char_match))*(math.pow(0.2,space_match))*(math.pow(0.1,unmatch))
                    else:
                        correlation_matrix[train_char][test_char_pos] = (math.pow(0.7,char_match))*(math.pow(0.2,space_match))*(math.pow(0.1,unmatch))
                else:
                    correlation_matrix[train_char] = {test_char_pos:(math.pow(0.7,char_match))*(math.pow(0.2,space_match))*(math.pow(0.1,unmatch))}
    return correlation_matrix

def calculate_emission_probabilities(train_letters,test_letters):
    correlation_matrix = compare_images(train_letters,test_letters)

    for test_char_pos in range(len(test_letters)):
        max = float('-inf')
        for train_char in train_letters.keys():
            if max < math.log(correlation_matrix[train_char][test_char_pos],10):
                emission_probs[test_char_pos] = (train_char,math.log(correlation_matrix[train_char][test_char_pos],10))
                max = math.log(correlation_matrix[train_char][test_char_pos],10)

def train(train_txt_fname,train_letters,test_letters):
    # data cleaning and putting it in the list (per sentence)
    train_text = read_data(train_txt_fname)

    # to get number of given characters in the given file in a dictionary and total characters in complete file
    # it also returns initial_char_count ie. occurence of given character in the first letter of the sentence
    total_characters = total_count_of_given_character(train_text)

    # calculating prior probability
    calculate_prior_probabilities(total_characters)

    # calculating init probability
    calculate_init_probabilities(train_text)
    for i in range(len(train_text)):
        for j in range(len(train_text[i])):
            if train_text[i][j] in character_count.keys():
                character_count[train_text[i][j]] += 1
            else:
                character_count[train_text[i][j]] = 1
            if i > 0:
                build_transition_count(train_text[i][j - 1], train_text[i][j])
    build_transition_probs()
    calculate_emission_probabilities(train_letters,test_letters)

def simplified(test_letters):
    words = []
    for i in range(len(test_letters)):
        words.append(emission_probs[i][0])
    simplified_sentence = "".join(words)
    return simplified_sentence

def hmm_viterbi(train_letters,test_letters):
    # viterbi_table[0]['noun'] - here 0 is the 1st letter in the sentence
    viterbi_table = [{}]
    em_weight = 0.05
    path_track = []
    path = []
    for train_char in train_letters.keys():
        for test_char_pos in range(len(test_letters)):
            if train_char in init_prob.keys():
                if test_char_pos in correlation_matrix[train_char].keys():
                    viterbi_table[0][train_char] = math.log(init_prob[train_char], 10) + math.log(correlation_matrix[train_char][test_char_pos], 10)*em_weight
                else:
                    viterbi_table[0][train_char] = math.log(init_prob[train_char], 10) + math.log(0.0000001,10)*em_weight
            else:
                if test_char_pos in correlation_matrix[train_char].keys():
                    viterbi_table[0][train_char] = math.log(0.0000001,10) + math.log(correlation_matrix[train_char][test_char_pos], 10)*em_weight
                else:
                    viterbi_table[0][train_char] = math.log(0.0000001,10)*em_weight
    for test_char_pos in range(1,len(test_letters)):
        viterbi_table.append({})
        path_track.append({})
        for train_char in train_letters.keys():
            max_value = float('-inf')
            for pre_char in train_letters.keys():
                if pre_char in transition_probs.keys():
                    if train_char in transition_probs[pre_char].keys():
                        value = viterbi_table[test_char_pos - 1][pre_char] + math.log(transition_probs[pre_char][train_char], 10)
                    else:
                        value = viterbi_table[test_char_pos - 1][pre_char] + math.log(0.0000001,10)
                else:
                    value = viterbi_table[test_char_pos - 1][pre_char] + math.log(0.0000001,10)
                if max_value < value:
                    max_value = value
                    max_char = pre_char
            viterbi_table[test_char_pos][train_char] = max_value + math.log(correlation_matrix[train_char][test_char_pos], 10)
            path_track[test_char_pos - 1][train_char] = max_char

    max_last_level = float('-inf')
    for char in train_letters.keys():
        if viterbi_table[len(test_letters) - 1][char] > max_last_level:
            max_last_level_char = char
            max_last_level = viterbi_table[len(test_letters) - 1][char]

    path.append(max_last_level_char)

    for i in range(len(path_track) - 1, -1, -1):
        max_last_level_char = path_track[i][max_last_level_char]
        path.append(max_last_level_char)

    hmm_viterbi_sentence = "".join(path[::-1])
    return hmm_viterbi_sentence

if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
train(train_txt_fname, train_letters, test_letters)
simple = simplified(test_letters)
viterbi = hmm_viterbi(train_letters,test_letters)

print("Simple: ", simple)
print("HMM: ", viterbi)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!
# print("TRAIN LETTERS", train_letters)

# print("TEST LETTERS", test_letters)

# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print("\n".join([ r for r in test_letters[2] ]))



# The final two lines of your output should look something like this:
#print("Simple: " + "Sample s1mple resu1t")
#print("   HMM: " + "Sample simple result") 


