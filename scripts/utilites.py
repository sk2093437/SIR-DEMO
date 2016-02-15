__author__ = 'TonySun'
"""
some common functions shared by all scripts
"""

import os
import setup
import cPickle as pickle
from nltk.corpus import stopwords
import re


# get absolute path of a file. Compatible in all OSs
def getAbsPath(path):
    """
    find absolute path given a relative file path
    :param path: relative path
    :return: absolute path
    """
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    abs_file_path = os.path.join(script_dir, path)
    return abs_file_path


def tupleToPhrase(tuple_list):
    """
    get a list of tuples and convert them to phrases
    e.g. (man, ride) to man_ride
    :return: a list contains all phrases
    """
    phrase_list = []
    for t in tuple_list:
        p_temp = t[0] + '_' + t[1]  # concatenate two terms
        phrase_list.append(p_temp)

    return phrase_list


def get_raw_text():
    """
    read text from a corpus file and do some simple pre-processing
    :return: processed raw text
    """
    raw_text_file = open(getAbsPath(setup.corpus_file_path))
    raw_text = raw_text_file.readlines()
    print("Corpus file " + raw_text_file.name + " was loaded.")
    # use re to split the raw text string and replace the original text
    # After this all the sentence are split into such format:
    # [0]filename, [1]order of annotation, [2]annotation text
    raw_text = [re.split('\t|#', singleLine.replace('\n', '')) for singleLine in raw_text]
    return raw_text


def get_file_list(file_path):
    raw_text_file = open(getAbsPath(file_path))
    raw_text = raw_text_file.readlines()
    raw_text = [("static/Corel5K/" + singleLine + ".jpeg").replace('\n', '') for singleLine in raw_text]
    return raw_text


# time to do some NLP stuff. We need a function to do this
def annotation_to_wordlists(annotations, remove_stop_words = False):
    """
    Function to convert a whole sentence to a word list
    Common NLP techniques are applied, such as to lower case, tokenization
    Returns a words lists, each list contains a tokenized sentence
    :param annotations: raw sentences extracted from corpus file
    :param remove_stop_words: whether stop words should be removed
    :return: list of sentences, each sentence is a list contains split words
    """
    sentences = [sentence.replace('.', '').strip().decode('utf-8').lower().split(" ") for
                 sentence in annotations]
    return sentences


def removeStopWords(words):
    """
    remove stops word from a given text corpus
    :param words: could be text of any length
    :return: text without stop words
    """
    # use english stopword list from nltk
    stop = set(stopwords.words('english'))
    words_copy = words.copy()
    for word in words:
        if word in stop:
            words_copy.remove(word)

    return words_copy


def saveVariableToFile(variable, path = setup.variable_obj_path):
    """
    save a variable to a file for long time usage
    :param variable: the variable to be saved, could be any data format
    """
    save_path = getAbsPath(path)
    f = open(save_path, 'wb')
    pickle.dump(variable, f, 0) # save variable as string stream
    f.close()
    print ("Successfully saved variable to " + save_path + '.')


def loadVariableFromFile(path = setup.variable_obj_path):
    """
    load a variable from a file
    :return: the load variable
    """
    load_path = getAbsPath(path)
    f = open(load_path, 'rb')
    var = pickle.load(f) # use pickle to load the variable in file f
    f.close()
    print ("Successfully loaded variable from " + load_path + '.')
    return var

