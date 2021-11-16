# The tagger.py starter code for CSC384 A4.

import collections
import os
import sys

import numpy as np
from collections import Counter

UNIVERSAL_TAGS = [
    "VERB",
    "NOUN",
    "PRON",
    "ADJ",
    "ADV",
    "ADP",
    "CONJ",
    "DET",
    "NUM",
    "PRT",
    "X",
    ".",
]

N_tags = len(UNIVERSAL_TAGS)

def read_data_train(path):
    return [tuple(line.split(' : ')) for line in open(path, 'r').read().split('\n')[:-1]]

def read_data_test(path):
    return open(path, 'r').read().split('\n')[:-1]

def read_data_ind(path):
    return [int(line) for line in open(path, 'r').read().split('\n')[:-1]]

def write_results(path, results):
    with open(path, 'w') as f:
        f.write('\n'.join(results))

def train_HMM(train_file_name):
    """
    Estimate HMM parameters from the provided training data.

    Input: Name of the training files. Two files are provided to you:
            - file_name.txt: Each line contains a pair of word and its Part-of-Speech (POS) tag
            - fila_name.ind: The i'th line contains an integer denoting the starting index of the i'th sentence in the text-POS data above

    Output: Three pieces of HMM parameters stored in LOG PROBABILITIES :
 
            - prior:        - An array of size N_tags
                            - Each entry corresponds to the prior log probability of seeing the i'th tag in UNIVERSAL_TAGS at the beginning of a sequence
                            - i.e. prior[i] = log P(tag_i)

            - transition:   - A 2D-array of size (N_tags, N_tags)
                            - The (i,j)'th entry stores the log probablity of seeing the j'th tag given it is a transition coming from the i'th tag in UNIVERSAL_TAGS
                            - i.e. transition[i, j] = log P(tag_j|tag_i)

            - emission:     - A dictionary type containing tuples of (str, str) as keys
                            - Each key in the dictionary refers to a (TAG, WORD) pair
                            - The TAG must be an element of UNIVERSAL_TAGS, however the WORD can be anything that appears in the training data
                            - The value corresponding to the (TAG, WORD) key pair is the log probability of observing WORD given a TAG
                            - i.e. emission[(tag, word)] = log P(word|tag)
                            - If a particular (TAG, WORD) pair has never appeared in the training data, then the key (TAG, WORD) should not exist.

    Hints: 1. Think about what should be done when you encounter those unseen emission entries during deccoding.
           2. You may find Python's builtin Counter object to be particularly useful 
    """

    pos_data = read_data_train(train_file_name+'.txt')
    sent_inds = read_data_ind(train_file_name+'.ind')

    ####################
    # STUDENT CODE HERE
    ####################

    # init
    index = iter(sent_inds)
    sen_index = next(index, len(sent_inds)-1)
    pos_count = dict.fromkeys(UNIVERSAL_TAGS,0)
    prior_track = dict.fromkeys(UNIVERSAL_TAGS,0)
    trans_track = [[0]*N_tags for i in range(N_tags)]
    emission_track = dict()


    # main loop
    for i in range(len(pos_data)):
        word = pos_data[i][0]
        pos = pos_data[i][1]
        pos_count[pos] = pos_count[pos] + 1

        if i == sen_index:
            prior_track[pos] = prior_track[pos] + 1
            sen_index = next(index, len(sent_inds)-1)
        else:
            prev_pos = pos_data[i-1][1]
            row = UNIVERSAL_TAGS.index(prev_pos)
            col = UNIVERSAL_TAGS.index(pos)
            trans_track[row][col] = trans_track[row][col]+1

        key = (pos, word)
        if key not in emission_track:
            emission_track[key] = 1
        else:
            emission_track[key] = emission_track[key] + 1


    prior = calc_prior(prior_track)
    transition = calc_trans(trans_track)
    emission = calc_emiss(emission_track, pos_count)
    return prior, transition, emission


# takes in a dict of tag:count and returns the log prob in np array object
    """ 
        similar to Language model unigram estimation
        # P('tag') ~ Count('tag')/Count(first token in each sentence)
        P(“Bob”) ~ Count(“Bob”) / Count(All words)
        P(w1) ~ Count(w1) / [Count(w1) + … + Count(w|V|)], V is size of vocab (aka number of tags)
        An array of size N_tags
        - Each entry corresponds to the prior log probability of seeing the i'th tag in UNIVERSAL_TAGS at the beginning of a sequence
        - i.e. prior[i] = log P(tag_i)
    """
def calc_prior(prior_track):
    prior_counter = collections.Counter(prior_track)
    prior_total_count = sum(prior_counter.values())
    prior = [0] * N_tags
    for i in range(N_tags):
        tag = UNIVERSAL_TAGS[i]
        prior[i] = np.log(prior_counter[tag]/prior_total_count)  
    return np.array(prior)   


# takes a 2-d array of size N_tag * N_tag that represent the i->j transition count 
    """
        # similar to Language model bigram estimation
        initial: P(“Bob”) ~ Count(w1=“Bob”) / Count(All sentences)
        P(“walks” | ”Bob”) ~ Count(“Bob walks”) / Count(All bigrams in which the first word is “Bob”)
        sentence P(“Bob walks in the park”) ~ P(“Bob”) * P(“walks” | ”Bob”) * P(“in” | “walks”) * P(“the” | “in) * P(“park” | “the”)
        A 2D-array of size (N_tags, N_tags)
        - The (i,j)'th entry stores the log probablity of seeing the j'th tag given it is a transition coming from the i'th tag in UNIVERSAL_TAGS
        - i.e. transition[i, j] = log P(tag_j|tag_i)
    """
def calc_trans(trans_track):
    transition = [[0]*N_tags for i in range(N_tags)]
    for row in range(len(trans_track)):
        row_sum = sum(trans_track[row])
        for col in range(len(trans_track[row])):
            transition[row][col] = np.log(trans_track[row][col]/row_sum)
    # print(transition)
    return np.array(transition)


def calc_emiss(emission_track, pos_count):
    emission = dict()
    for key in emission_track:
        tag = key[0]
        prob = emission_track[key]/pos_count[tag]
        emission[key] = np.log(prob)
    return emission


def tag(train_file_name, test_file_name):
    """
    Train your HMM model, run it on the test data, and finally output the tags.

    Your code should write the output tags to (test_file_name).pred, where each line contains a POS tag as in UNIVERSAL_TAGS
    """

    prior, transition, emission = train_HMM(train_file_name)

    pos_data = read_data_test(test_file_name+'.txt')
    sent_inds = read_data_ind(test_file_name+'.ind')

    ####################
    # STUDENT CODE HERE
    ####################

    """
    req:
        state (s) -> tags = pos
        observation (o) -> sentence

        initial (aka prior)
        transition
        emission
        (all three given by the training)

        prob trellis and path trellis = matrix (len(s), len(o))


    """
    best_path = []
    results = []
    sen = pos_data
    prob_trellis = np.array([[0]*N_tags for i in range(len(sen))])
    path_trellis = init_matrix(len(sen),N_tags)
    for i in range(len(sen)):
        for j in range(N_tags):
            tag = UNIVERSAL_TAGS[j]
            word = sen[i]
            if (tag,word) not in emission: 
                prob_trellis[i][j] = np.log(0.0000001)
            elif i == 0:
                prob_trellis[i][j] = emission[(tag, word)] + prior[j]
                path_trellis[i][j] = [tag]
            else:
                x = calc_argmax(prob_trellis, transition, emission, tag, word, i, j)
                prob_trellis[i][j] = prob_trellis[i-1][x] + transition[j][x] + emission[(tag,word)]
                path_trellis[i][j] = path_trellis[i-1][x][:]
                path_trellis[i][j].append(tag)
        best_path.append(np.argmin(prob_trellis[i]))

    for x in best_path:
        results.append(UNIVERSAL_TAGS[x])

    write_results(test_file_name+'.pred', results)


def calc_argmax(prob_trellis, transition, emission, tag, word, i, j):
    col = []
    for x in range(N_tags):
        col.append(prob_trellis[i-1][x] + transition[j][x] + emission[(tag,word)])
    return np.argmax(col)

def init_matrix(height, width):
    return [[[]]*width for i in range(height)]
            

if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training file> -t <test file>"
    # E.g. python3 tagger.py -d data/train-public -t data/test-public-small
    parameters = sys.argv
    train_file_name = parameters[parameters.index("-d")+1]
    test_file_name = parameters[parameters.index("-t")+1]

    # Start the training and tagging operation.
    tag (train_file_name, test_file_name)