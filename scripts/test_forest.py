from __future__ import division
from __future__ import unicode_literals
import utilites
from scipy.io import loadmat
import numpy as np
from build_supervised_tree import TreeNode
from build_supervised_tree import LabelCount
import time
from collections import Counter
from operator import itemgetter
from scipy.spatial import distance

__author__ = 'TonySun'

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

# we need to parse each test sample here
# get all terms from txt file
"""
test_file = open(utilites.getAbsPath('static/Corel5K/corel5k_test_list.txt'))
test_file_list = test_file.readlines()
test_file_list = [term.strip().decode('utf-8').replace('\n', '') + '.jpeg' for term in test_file_list]
utilites.saveVariableToFile(test_file_list, utilites.getAbsPath('static/Corel5K/corel5k_test_list.pkl'))
"""
test_file_list = utilites.loadVariableFromFile('static/Corel5K/corel5k_test_list.pkl')
test_anno = utilites.loadVariableFromFile('static/Corel5K/test_anno_filtered.pkl')
train_anno = utilites.loadVariableFromFile('static/Corel5K/train_anno_filtered.pkl')
train_vectors = loadmat(utilites.getAbsPath('static/Corel5K/train_vectors_original.mat'))
train_vectors = train_vectors['train_vectors']
test_vectors = loadmat(utilites.getAbsPath('static/Corel5K/test_vectors_original.mat'))
test_vectors = test_vectors['test_vectors']

train_vectors_classic = loadmat(utilites.getAbsPath('static/Corel5K/baseline_features/corel5k_train_feats_classic.mat'))
train_vectors_classic = train_vectors_classic['corel5k_train_feats_classic']

test_vectors_classic = loadmat(utilites.getAbsPath('static/Corel5K/baseline_features/corel5k_test_feats_classic.mat'))
test_vectors_classic = test_vectors_classic['corel5k_test_feats_classic']



print("Loading forest model...")
tic()
forest = utilites.loadVariableFromFile("static/Corel5K/forest/forest_128.pkl")
print("Done.")
toc()


def parse_single_tree(sample, node):
    """
    parse an new instance using given tree
    :param sample: test instance
    :param node: single tree, root node
    :return: training sample names related to this instance and their label count
    """
    if node.is_leaf:
        # return sample names and corresponding label count of this node
        return node.label_count, node.orig_sample_indexes
    else:
        if sample[node.feat_split_index] >= node.feat_split_value:
            return parse_single_tree(sample, node.upper_child)
        else:
            return parse_single_tree(sample, node.lower_child)


def parse_forest(sample, forest):
    """
    Feed test sample to all trees in the given forest then get
    statistical result of all trees
    :param sample: test sample
    :param forest: forest
    :return: summary of results of all trees
    """
    a_rc = []
    a_rs = []

    for tree in forest:
        rc, rs = parse_single_tree(sample, tree)
        a_rc.append(rc.label_count)
        a_rs = a_rs + list(rs)

    a_rc = np.asarray(a_rc)
    sum_a_rc = np.sum(a_rc, axis=0)  # get count in all trees for each concept
    sum_a_rs = Counter(a_rs)

    return sum_a_rc, sum_a_rs


def parse_test_sample(test_sample, forest, k=10):
    """
    Pass the feature of test sample to the forest and retrieve the result
    :param test_sample: feature of test sample
    :param forest: loaded forest
    :return: src_top5: top5 concept label count, srs_top5: top5 sample count
    """
    # src, srs = self.parse_forest(test_sample)
    src, srs = parse_forest(test_sample, forest)
    label_name = range(100)  # build a label sequence
    src_dict = dict(zip(label_name, src))  # build dict with label sequence and count

    src_sorted = sorted(src_dict.items(), key=itemgetter(1))[::-1]  # sort the dict according to count
    srs_sorted = sorted(srs.items(), key=itemgetter(1))[::-1]  # sort the sample according to count

    src_topk = src_sorted[0:k]  # get top5 label count
    srs_topk = srs_sorted[0:k]  # get top5 sample count
    # replace sample index with real image path

    print src_topk
    print srs_topk

    return src_topk, srs_topk


def find_near_neighbor(test_sample, k=5):
    """
    find nearest neighbor by computed cosine distance between test sample and all training samples
    :param test_sample:
    :param k:
    :return:
    """
    test_sample = np.array([test_sample])
    dist = 1 - distance.cdist(test_sample, train_vectors_classic, metric='cosine')  # compute distance to fine k nearest neighbor
    dist_sorted = dist.argsort()[0][::-1] # sort the distance with descending order and return sample index
    dist_topk = list(dist_sorted[0:k])
    # sample_topk_path = [self.train_file_path[s_index] for s_index in dist_topk]
    print dist_topk
    return dist_topk


def cal_semantic_knn_metric(query_vector_index, query_anno, forest, k=10):
    """
    calculate semantic k nearest neighbour metric
    :param test_vector: vector of query image
    :param test_anno: tags of query image
    :param forest: random forest model
    :param k: top k retrieved result
    :return:
    """
    src_topk, srs_topk= parse_test_sample(test_vectors[query_vector_index], forest, k)  # parse a test sample
    index_knn = find_near_neighbor(test_vectors_classic[query_vector_index], k)  # parse test sample with KNN method
    query_anno_index = set(np.where(query_anno == 1)[0])
    CBRM_sample = [s[0] for s in srs_topk]  # get sample index
    SKNNM_RF = 0  # count SKNNM for CGRM method
    SKNNM_KNN = 0  # count for knn baseline method

    # get tag co-occurrence count where tags in query image also exist in training images
    for s_index in CBRM_sample:
        temp_anno = train_anno[s_index]
        temp_anno_index = set(np.where(temp_anno == 1)[0])
        res = list(query_anno_index.intersection(temp_anno_index))
        SKNNM_RF += len(res)

    for k_index in index_knn:
        temp_anno = train_anno[k_index]
        temp_anno_index = set(np.where(temp_anno == 1)[0])
        res = list(query_anno_index.intersection(temp_anno_index))
        SKNNM_KNN += len(res)

    print("Count of RF: " + str(SKNNM_RF))
    print("Count of KNN: " + str(SKNNM_KNN))
    return SKNNM_RF, SKNNM_KNN


def cal_all_sknnm(topk=10):
    sum_rf = 0
    sum_knn = 0
    # test code only
    for i in range(len(test_vectors)):
        rf, knn = cal_semantic_knn_metric(i, test_anno[i], forest, topk)
        sum_rf += rf
        sum_knn += knn
        print ("This is query image " + str(i))

    print("")
    print("Number of nearest neighbour: " + str(topk))
    print("Total result of RF: " + str(sum_rf))
    print("Total result of KNN: " + str(sum_knn))
    return sum_rf, sum_knn



# example: 20000/20004.jpeg
# 108000/108049.jpeg