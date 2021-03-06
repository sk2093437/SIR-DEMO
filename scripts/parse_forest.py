from __future__ import division
from __future__ import unicode_literals

__author__ = 'TonySun'

from collections import Counter
import numpy as np
from build_supervised_tree import TreeNode
from build_supervised_tree import LabelCount
from operator import itemgetter
import utilites
import time
import matlab_wrapper
from random import shuffle
from scipy.io import loadmat
from scipy.spatial import distance


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


class ParseForest():

    def __init__(self):

        print("Loading forest model...")
        tic()
        self.forest = utilites.loadVariableFromFile("static/Corel5K/forest/forest_128.pkl")
        print("Done.")
        toc()
        self.train_vectors = loadmat(utilites.getAbsPath('static/Corel5K/train_vectors_original.mat'))
        self.train_vectors = self.train_vectors['train_vectors']
        self.train_file_path = utilites.loadVariableFromFile("static/Corel5K/train_file_path.pkl")
        # load contents of concepts
        self.concepts = utilites.loadVariableFromFile("static/Corel5K/cluster_contents.pkl")
        self.tag_scores = utilites.loadVariableFromFile("static/Corel5K/all_tags_scores.pkl")

        self.train_vectors_classic = loadmat(utilites.getAbsPath('static/Corel5K/baseline_features/corel5k_train_feats_classic.mat'))
        self.train_vectors_classic = self.train_vectors_classic['corel5k_train_feats_classic']

        self.test_vectors_classic = loadmat(utilites.getAbsPath('static/Corel5K/baseline_features/corel5k_test_feats_classic.mat'))
        self.test_vectors_classic = self.test_vectors_classic['corel5k_test_feats_classic']
        self.test_file_name = utilites.loadVariableFromFile('static/Corel5K/corel5k_test_file_name.pkl')
        self.feat_dict_classic = dict(zip(self.test_file_name, self.test_vectors_classic))

        # start a matlab session for feature extraction
        self.matlab = matlab_wrapper.MatlabSession(matlab_root="/Applications/MATLAB_R2015b.app")  # start matlab
        self.matlab.eval('run MatConvNet/matlab/vl_setupnn')  # basic config
        self.matlab.eval('run vlfeat/toolbox/vl_setup')  ## basic config
        self.matlab.eval("feature('DefaultCharacterSet', 'UTF8')")
        print("Loading cnn model...")
        tic()
        self.matlab.eval("net = load('/Users/TONYSUN/Desktop/SIR_Corel5K_demo/static/cnnmodel/imagenet-matconvnet-vgg-verydeep-16.mat')")
        toc()
        print("Matlab session started.")
        print("Ready for work ^_^.")


    def disp_tree_info(self, node):
        """
        Display all nodes contained in the given tree
        :param node: root node of a tree
        :return: nothing
        """
        print("")
        if node.parent is None:
            print("This is root node.")
        node.__str__()

        if not node.is_leaf:
            print("Upper child: ")
            self.disp_tree_info(node.upper_child)
            print("Lower child: ")
            self.disp_tree_info(node.lower_child)

        return ''


    def parse_single_tree(self, sample, node):
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
                return self.parse_single_tree(sample, node.upper_child)
            else:
                return self.parse_single_tree(sample, node.lower_child)


    def parse_forest(self, sample):
        """
        Feed test sample to all trees in the given forest then get
        statistical result of all trees
        :param sample: test sample
        :param forest: forest
        :return: summary of results of all trees
        """
        a_rc = []
        a_rs = []

        for tree in self.forest:
            rc, rs = self.parse_single_tree(sample, tree)
            a_rc.append(rc.label_count)
            a_rs = a_rs + list(rs)

        a_rc = np.asarray(a_rc)
        sum_a_rc = np.sum(a_rc, axis=0)  # get count in all trees for each concept
        sum_a_rs = Counter(a_rs)

        return sum_a_rc, sum_a_rs


    def find_near_neighbor(self, test_file_name, k=5):
        """
        find nearest neighbor by computed cosine distance between test samle and all training samples
        :param test_sample:
        :param k:
        :return:
        """
        test_sample = np.array([self.feat_dict_classic[test_file_name]])
        dist = 1 - distance.cdist(test_sample, self.train_vectors_classic, metric='cosine')  # compute distance to fine k nearest neighbor
        dist_sorted = dist.argsort()[0][::-1] # sort the distance with descending order and return sample index
        dist_topk = list(dist_sorted[0:k])
        sample_topk_path = [self.train_file_path[s_index] for s_index in dist_topk]
        print sample_topk_path
        return sample_topk_path


    def parse_test_sample(self, test_sample, test_file_name, k=5):
        """
        Pass the feature of test sample to the forest and retrieve the result
        :param test_sample: feature of test sample
        :param forest: loaded forest
        :return: src_top5: top5 concept label count, srs_top5: top5 sample count
        """
        # src, srs = self.parse_forest(test_sample)
        rf_start = time.time()
        src, srs = self.parse_forest(test_sample)
        label_name = range(100)  # build a label sequence
        src_dict = dict(zip(label_name, src))  # build dict with label sequence and count

        src_sorted = sorted(src_dict.items(), key=itemgetter(1))[::-1]  # sort the dict according to count
        srs_sorted = sorted(srs.items(), key=itemgetter(1))[::-1]  # sort the sample according to count

        src_topk = src_sorted[0:k]  # get top5 label count
        srs_topk = srs_sorted[0:k]  # get top5 sample count
        # replace sample index with real image path
        srs_topk = [(self.train_file_path[pair[0]], pair[1]) for pair in srs_topk]
        rf_end = time.time() - rf_start
        rf_end = str(float("{0:.2f}".format(rf_end))) + ' seconds'

        print src_topk
        print srs_topk

        bl_start = time.time()
        im_near_path = self.find_near_neighbor(test_file_name)

        bl_end = time.time() - bl_start
        bl_end = str(float("{0:.2f}".format(bl_end))) + ' seconds'

        return src_topk, srs_topk, im_near_path, (rf_end, bl_end)


    def extract_feature(self, test_input):
        """
        extract cnn feature from given image (ndarray)
        :param inputs:
        :param layername:
        :return:
        """
        self.matlab.put('test_input', test_input)
        self.matlab.eval("tr_im_temp = imread(test_input);")
        self.matlab.eval("averaged_im = single(tr_im_temp) ;")
        self.matlab.eval("averaged_im = imresize(averaged_im, net.meta.normalization.imageSize(1:2)) ;")
        self.matlab.eval("av_im = net.meta.normalization.averageImage ;")
        self.matlab.eval("temp = zeros(size(averaged_im)) ;")
        self.matlab.eval("temp(:,:,1) = av_im(1);")
        self.matlab.eval("averaged_im(:,:,1) = averaged_im(:,:,1) - temp(:,:,1) ;")
        self.matlab.eval("temp(:,:,2) = av_im(2);")
        self.matlab.eval("averaged_im(:,:,2) = averaged_im(:,:,2) - temp(:,:,2) ;")
        self.matlab.eval("temp(:,:,3) = av_im(3);")
        self.matlab.eval("averaged_im(:,:,3) = averaged_im(:,:,3) - temp(:,:,3) ;")

        self.matlab.eval("result = vl_simplenn(net, averaged_im) ;")
        self.matlab.eval("feat_vec = transpose(squeeze(result(35).x)) ;")

        return self.matlab.workspace.feat_vec


    def im2res(self, im_path, test_file_name):
        """
        read an image file and give the final parsed result
        :param im_path: image file path
        :return: top5 concept label count and top5 sample count
        """
        im_vec = self.extract_feature(utilites.getAbsPath(im_path))
        src, srs, im_near_path, rf_bl_time = self.parse_test_sample(im_vec, test_file_name)

        return src, srs, im_near_path, rf_bl_time


    def gen_random_labels(self, src):
        """
        generate labels according to given concepts
        :param src: top 5 concepts counts
        :return: generated labels for test image
        """
        src1 = src[0]  # keep top 2 result
        src2 = src[1]
        con1 = self.concepts[src1[0]]  # get labels for each concepts
        con2 = self.concepts[src2[0]]
        shuffle(con1)
        shuffle(con2)

        if len(con1) > 3:
            con1 = con1[0:3]  # keep at most three labels for each concepts

        if len(con2) > 3:
            con2 = con2[0:3]

        return con1 + con2  # return all labels


    def gen_weighted_labels(self, src):
        """
        return weighted labels from given concepts
        :param src: top 5 concepts counts
        :return: weighted labels for test image
        """
        src1 = src[0]  # keep top 2 result
        src2 = src[1]
        con1 = self.concepts[src1[0]]  # get labels for each concepts
        con2 = self.concepts[src2[0]]

        score1 = []  # create empty list to store score
        score2 = []

        for lb1 in con1:
            score1.append(self.tag_scores[lb1])  # find score for given tag in tag dict

        for lb2 in con2:
            score2.append(self.tag_scores[lb2])

        d_c1 = dict(zip(con1, score1))  # build dict to store tag name and score
        d_c2 = dict(zip(con2, score2))

        d_c1_sorted = sorted(d_c1.items(), key=itemgetter(1))[::-1]  ## sort dict according to tag score
        d_c2_sorted = sorted(d_c2.items(), key=itemgetter(1))[::-1]

        if len(d_c1_sorted) > 3:
            d_c1_sorted = d_c1_sorted[0:3]
        if len(d_c2_sorted) > 3:
            d_c2_sorted = d_c2_sorted[0:3]

        d_c1_sorted = [pair[0] for pair in d_c1_sorted]
        d_c2_sorted = [pair[0] for pair in d_c2_sorted]
        print("Result of weighted labelling:")
        print d_c1_sorted + d_c2_sorted

        return d_c1_sorted + d_c2_sorted


# test code only
# train_file_path = utilites.loadVariableFromFile("static/Corel5K/train_file_path.pkl")
# train_vectors = loadmat(utilites.getAbsPath('static/Corel5K/train_vectors_original.mat'))
# train_vectors = train_vectors['train_vectors']
# test_vectors = loadmat(utilites.getAbsPath('static/Corel5K/test_vectors_original.mat'))
# test_vectors = test_vectors['test_vectors']
#
# def find_near_neighbor(test_sample, k=5):
#     test_sample = np.array([test_sample])
#     dist = 1 - distance.cdist(test_sample, train_vectors, metric='cosine')  # compute distance to fine k nearest neighbor
#     dist_sorted = dist.argsort()[0][::-1] # sort the distance with descending order and return sample index
#     dist_topk = list(dist_sorted[0:k])
#     print dist_topk
#     sample_topk_path = [train_file_path[s_index] for s_index in dist_topk]
#     return sample_topk_path
#
# result = find_near_neighbor(test_vectors[0])