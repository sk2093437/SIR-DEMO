__author__ = 'TonySun'

from collections import Counter
import numpy as np
from build_supervised_tree import TreeNode
from build_supervised_tree import LabelCount
from operator import itemgetter
import utilites
import time
import matlab_wrapper


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
        self.forest = utilites.loadVariableFromFile("static/Corel5K/forest_400_trees_64_feats/forest.pkl")
        print("Done.")
        toc()
        self.train_file_path = utilites.loadVariableFromFile("static/Corel5K/train_file_path.pkl")
        # start a matlab session for feature extraction
        self.matlab = matlab_wrapper.MatlabSession(matlab_root="/Applications/MATLAB_R2015b.app")  # start matlab
        self.matlab.eval('run MatConvNet/matlab/vl_setupnn')  # basic config
        self.matlab.eval('run vlfeat/toolbox/vl_setup')  ## basic config
        print("Loading cnn model...")
        tic()
        self.matlab.eval("net = load('/Users/TONYSUN/Desktop/sir_demo/static/cnnmodel/imagenet-matconvnet-vgg-verydeep-16.mat')")
        toc()
        print("Matlab session started.")
        print("Ready for working ^_^.")


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
        print(sample)
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


    def parse_test_sample(self, test_sample):
        """
        Pass the feature of test sample to the forest and retrieve the result
        :param test_sample: feature of test sample
        :param forest: loaded forest
        :return: src_top5: top5 concept label count, srs_top5: top5 sample count
        """
        # src, srs = self.parse_forest(test_sample)
        src, srs = self.parse_forest(test_sample)
        label_name = range(100)  # build a label sequence
        src_dict = dict(zip(label_name, src))  # build dict with label sequence and count

        src_sorted = sorted(src_dict.items(), key=itemgetter(1))[::-1]  # sort the dict according to count
        srs_sorted = sorted(srs.items(), key=itemgetter(1))[::-1]  # sort the sample according to count

        src_top5 = src_sorted[0:5]  # get top5 label count
        srs_top5 = srs_sorted[0:5]  # get top5 sample count
        # replace sample index with real image path
        srs_top5 = [(self.train_file_path[pair[0]], pair[1]) for pair in srs_top5]

        return src_top5, srs_top5


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


    def im2res(self, im_path):
        """
        read an image file and give the final parsed result
        :param im_path: image file path
        :return: top5 concept label count and top5 sample count
        """
        im_vec = self.extract_feature(utilites.getAbsPath(im_path))
        src, srs = self.parse_test_sample(im_vec)

        return src, srs





