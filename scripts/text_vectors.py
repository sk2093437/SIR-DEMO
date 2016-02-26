__author__ = 'TonySun'
"""
This file is used to test the text vectors using Google's
# Word2Vec and Doc2Vec tools in Gensim toolkit
# till now the text file used is from annotation of Flickr 8k annotation dataset
# Each image have 5 annotations
# All the annotations are manually done and each one appears as a single sentence
"""
import gensim
import utilites
import setup
from sklearn.metrics import pairwise


# return text vectors calculated using Word2Vec by gensim
"""
open the annotation text file and read content
build word vectors using Word2Vec and then extract
the term/vector pairs into a dictionary
"""
# get all filtered term(tag) names
terms_corel5k_filtered = utilites.loadVariableFromFile("Corel5k/terms_corel5k_filtered.pkl")
# get training image annotations: lists of separate terms
train_anno_filtered = utilites.loadVariableFromFile("Corel5k/train_anno_filtered.pkl")

# initialize a model using parameters above
word_model = gensim.models.Word2Vec.load_word2vec_format(utilites.getAbsPath(setup.lmodel_file_path), binary=True)

"""
Calculate similarity matrix using given vectors
We use pairwise distances to build the matrix
"""
print ("Extracting word vectors...")
vecs = []
# Index2word is a list that contains the names of the words in
for word in terms_corel5k_filtered:
    vecs.append(word_model[word])  # now we extract all word vectors from the model

print ("Term vectors haven been created.")
d_pairwise_vecs = 1 - pairwise.pairwise_distances(vecs, metric='cosine')
print("Similarity matrix has been built.")

utilites.saveVariableToFile(d_pairwise_vecs, "Corel5k/tag_textual_similarity_matrix.pkl")
