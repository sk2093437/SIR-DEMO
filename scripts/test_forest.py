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
import matplotlib.pyplot as plt
import skimage.io as io

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
train_file_path = utilites.loadVariableFromFile("static/Corel5K/train_file_path.pkl")
test_anno = utilites.loadVariableFromFile('static/Corel5K/test_anno_filtered.pkl')
train_anno = utilites.loadVariableFromFile('static/Corel5K/train_anno_filtered.pkl')
train_anno_concept = utilites.loadVariableFromFile("static/Corel5K/train_anno_concept.pkl")
test_anno_concept = utilites.loadVariableFromFile("static/Corel5K/test_anno_concept.pkl")
all_prob = utilites.loadVariableFromFile("static/Corel5K/all_probs.pkl")
concepts = utilites.loadVariableFromFile("static/Corel5K/cluster_contents.pkl")
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


def parse_test_sample_findall(test_sample, forest):
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

    print src_sorted
    print srs_sorted

    return src_sorted, srs_sorted


def cal_concept_prob_single(test_sample_index):
    """
    calculate the probability that a given test image contains each tag
    :param test_sample_index: index of test sample in all test samples
    :return: a list containing probabilities for the image containing each tag
    """
    # first obtain all nearest neighbour
    src, srs = parse_test_sample_findall(test_vectors[test_sample_index], forest)
    prob = np.zeros((1, len(src)))  # initialize an empty zero to store concept probability

    all_count = 0.0  # sum counts for all retrieved semantic nearest neighbour
    for rs in srs:
        all_count += rs[1]

    for i in range(0, len(src)):  # for each concept
        for pair in srs:  # for every retrieved semantic nearest neighbour
            if train_anno_concept[pair[0]][i] == 1:  # if this SNN contains concept i
                prob[0][i] += pair[1]  # add semantic measure to probability

        prob[0][i] = prob[0][i] / all_count

    return prob


def cal_concept_prob_all():
    """
    calculate probability of containing each concept for all test sample
    :return: probability matrix, row denotes each test sample, column denotes probability of each concept
    """
    tic()
    all_prob = []
    for i in range(0, len(test_file_list)):
        all_prob.append(np.squeeze(cal_concept_prob_single(i)))

    all_prob = np.asarray(all_prob)
    toc()
    return all_prob


def concept_based_retrieval(concept_index, all_prob, topk=5, get_path=False):
    prob_concept = all_prob[:, concept_index]
    test_index = list(prob_concept.argsort()[::-1])
    test_topk = test_index[0:topk]

    if get_path:
        test_topk = ['static/Corel5K/' + test_file_list[i] for i in test_topk]
        print test_topk

    return test_topk


def cal_avg_precision(concept_index, all_prob, k=10):
    """
    calculate average precision for a single concept retrieval
    :param concept_index: index of query concept
    :param all_prob: probability matrix
    :param k: top k result used for computing
    :return: average precision of this concept
    """
    result = concept_based_retrieval(concept_index, all_prob,k)  # first get the retrieval result
    preci_indicator = np.zeros(len(result))  # initialize empty array to store precision at each rank
    for i in range(0, len(result)):  # given top k result
        if test_anno_concept[result[i]][concept_index] == 1:
            preci_indicator[i] = 1.0
    # print preci_indicator

    avg_precision = 0.0
    for j in range(0, len(preci_indicator)):
        # print np.sum(preci_indicator[0:j+1])
        temp = np.sum(preci_indicator[0:j+1]) / (j+1)
        # print temp
        avg_precision += temp

    avg_precision = avg_precision / len(preci_indicator)
    print("Average precision of this query: " + str(avg_precision))
    return avg_precision


def cal_mean_avg_precision(all_prob):
    """
    calculate mean average precision for multiple queries
    :param all_prob: probability matrix
    :return: mean average precision
    """
    mean_avg_precision = 0.0
    for i in range(0, len(concepts)):
        result = cal_avg_precision(i, all_prob, k=10)
        mean_avg_precision += result
        # print mean_avg_precision

    mean_avg_precision = mean_avg_precision / len(concepts)
    print("Mean average precision for " + str(len(concepts)) + " concept queries is: " + str(mean_avg_precision))
    return mean_avg_precision


def show_images(image_list,titles=None):
    """Display a list of images"""
    images = []
    for j in range(0, len(image_list)):
        images.append(io.imread(utilites.getAbsPath(image_list[j])))

    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if image.ndim == 2: # Is image grayscale?
            plt.gray() # Only place in this blog you can't replace 'gray' with 'grey'
        plt.imshow(image)
        a.set_title(title)
        n += 1
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def concept_to_image(concept_index, all_prob, topk=5):
    print ("Content of current query concept " + str(concept_index) + ": ")
    print concepts[concept_index]
    result = concept_based_retrieval(concept_index, all_prob, topk, get_path=True)
    precision = cal_avg_precision(concept_index, all_prob, topk)
    show_images(result)

