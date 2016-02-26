
"""
After we extracted images features of Corel5K using MatConvNet (Matlab Toolbox)
Since we choose fc7 layer which consists 4096 dimension of features
We need to reduce it to a much lower dimension in order to compute pairwise distance
or building random forest
"""

from __future__ import division
import utilites
from scipy.io import loadmat
from sklearn.metrics import pairwise
import numpy as np
import multiprocessing

__author__ = 'TONYSUN'

"""
Define a tic toc function similar as Matlab
"""
import time

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

"""
Prepare data
"""
# START TO RUN HERE
# load datasets
train_original = loadmat(utilites.getAbsPath('Corel5K/train_vectors_original.mat'))
test_original = loadmat(utilites.getAbsPath('Corel5K/test_vectors_original.mat'))
train_original = train_original['train_vectors']
test_original = test_original['test_vectors']

# l2 normalize the feature vectors
train_original_list = [tr.astype('float32') for tr in train_original]

# construct a pairwise distance matrix to lookup (affinity matrix)
d_cos_train_vecs = 1 - pairwise.pairwise_distances(train_original_list, metric='cosine')
# d_eu_train_vecs = pairwise.pairwise_distances(train_pca_300_list, metric='euclidean')

# get all terms from txt file
terms_file = open(utilites.getAbsPath('Corel5K/corel5k_words.txt'))
terms_corel5k = terms_file.readlines()
print("Term file " + terms_file.name + " was loaded.")
terms_corel5k = [term.strip().decode('utf-8').replace('\n', '') for term in terms_corel5k]

# load annotations
# the annotations are in one-hot representation: 1 means the tag corresponds to a image, 0 otherwise
train_anno = loadmat(utilites.getAbsPath('Corel5K/corel5k_train_annot.mat'))
test_anno = loadmat(utilites.getAbsPath('Corel5K/corel5k_test_annot.mat'))

train_anno = train_anno['corel5k_train_annot']
test_anno = test_anno['corel5k_test_annot']

# we need to know what images share a same tag
term_assoi_image = []
for t in range(len(terms_corel5k)):
    # for each term in the term list, for each column in train_anno
    # if it is equal to 1, put the corresponding index into the list
    term_assoi_image.append(np.where(train_anno[:, t] == 1)[0])

"""
filter tags according to their visual discriminating power and frequency
"""

def cal_visual_scores(tag, k=300):
    """
    Calculate visual score of a given tag, the score indicates its visual discriminating power
    we first find all associated images of this tag
    and for each image, we find k visual nearest neighbour and
    find if given tag appears in the image annotation
    We use weight score related with the position in the k nearest neighbour
    :param tag: given tag that is used to calculate score
    :param k: number of nearest neighbour obtained
    :return: the visual discriminating score of this tag, type:double
    """
    score_curr_tag = []
    # for tag in range(10):
    # get the first image set which contains tag 1:
    t_images = term_assoi_image[tag]
    print('')
    print('For tag ' + terms_corel5k[tag] + ', ' + str(len(t_images)) + ' images are associated.')
    scores_t_images = []
    # for each image associated with this tag
    for t_im in t_images:
        # lookup corresponding row in affinity matrix
        # here t_im is the index of images associated with given tag
        aff_t_im = d_cos_train_vecs[t_im]
        # sort the aff_t_im by descending order and return the original index
        aff_t_im_sort_index = aff_t_im.argsort()[::-1]
        # take top K affine images
        aff_t_im_topK = aff_t_im_sort_index[0:k]
        temp_sum = 0.0

        for im_topK in range(len(aff_t_im_topK)):
            # get the annotation of this image, it's a row in annotation matrix
            temp_anno = train_anno[aff_t_im_topK[im_topK], :]
            # calculate weight
            temp_weight = 1 - im_topK / k
            # find if given tag appears in the im_topK-th image
            score = temp_anno[tag] * temp_weight
            # add score for all images
            temp_sum = temp_sum + score

        scores_t_images.append(temp_sum)
        #print ('The score of tag ' + terms_corel5k[tag] + ' for image ' + str(t_im)
            #   + ' is ' + str(temp_sum) + '.')

    scores_t_images = np.array(scores_t_images)
    # calculate the median for all scores of given tag
    median = np.median(scores_t_images)
    print('The median score of tag ' + terms_corel5k[tag] + ' is ' + str(median) + '.')

    score_curr_tag.append(median)

    return score_curr_tag


# use parallel computing to calculate scores for all tags
tags = range(len(terms_corel5k))
pool = multiprocessing.Pool(multiprocessing.cpu_count())
tic()
scores_tags = pool.map(cal_visual_scores, tags)
toc()
pool.close()
pool.join()

tag_score_dict = dict(zip(terms_corel5k, scores_tags))

utilites.saveVariableToFile(tag_score_dict, "Corel5K/all_tags_scores.pkl")

# retain tags that appears more than 5 times
tag_frequency = []
for te in range(len(terms_corel5k)):
    tag_frequency.append(sum(train_anno[:, te]))
tag_frequency = np.asarray(tag_frequency)

# get the index of infrequent tags in the tag list
r_infreq_tags = np.where(tag_frequency < 5)[0]
# thresholding
scores_tags_th = np.where(np.asarray(scores_tags) <= 1)[0]
# integrate low frequency and low scored
filtered_index = np.union1d(scores_tags_th, r_infreq_tags)
lwords = utilites.loadVariableFromFile("Corel5k/words_of_lmodel.pkl")
# find tags which are not available in language model
no_avail_tags = list(set(terms_corel5k) - set(terms_corel5k).intersection(set(lwords)))

# get index of not available tags
no_avail_index = []
for i in range(len(no_avail_tags)):
    if no_avail_tags[i] in terms_corel5k:
        no_avail_index.append(terms_corel5k.index(no_avail_tags[i]))

filtered_index = np.union1d(filtered_index, no_avail_index)

# get abandaned tags
filtered_tags = np.asarray(terms_corel5k)[filtered_index]
# filter tags
terms_corel5k_filtered = np.delete(terms_corel5k, filtered_index)
# remove columns from corresponding annotation matrix 0: row, 1: column
train_anno_filtered = np.delete(train_anno, list(filtered_index), 1)


"""
Calculate tag similarity in terms of visual appearance
"""

def cal_tag_similarity(base_tag, k=300):
    """
    Calculate similarity of base tag and another given tag
    for base tag, we still obtain all its associated images
    and for each image, we find visual k nearest neighbours,
    and then find if another tag appears in these k neighbours
    again, weight scores are calculated and median score from all associated images is taken
    note that simi(tag1, tag2) is different with simi(tag2, tag1) since
    their associated images' nearest neighbours differ
    :param base_tag: the base tag used to calculate similarity score with other tags
    :param k: number of nearest neighbour obtained
    :return: the similarity scores between base tag and other tags, type: double 1-d array
    """
    score_curr_tag = []
    # for tag in range(10):
    # get the first image set which contains tag 1:
    t_images = term_assoi_image[base_tag]
    print('')
    print('For tag ' + terms_corel5k_filtered[base_tag] + ', ' + str(len(t_images)) + ' images are associated.')

    # for each other tag in all tags
    for other_tag in range(len(terms_corel5k_filtered)):
        if other_tag == base_tag:
            # do not compare similarity with itself
            score_curr_tag.append(1.0)
            continue

        scores_t_images = []
        # for each image associated with base tag
        for t_im in t_images:
            # lookup corresponding row in affinity matrix
            # here t_im is the index of images associated with given tag
            aff_t_im = d_cos_train_vecs[t_im]
            # sort the aff_t_im by descending order and return the original index
            aff_t_im_sort_index = aff_t_im.argsort()[::-1]
            # take top K affine images
            aff_t_im_topK = aff_t_im_sort_index[0:k]
            temp_sum = 0.0

            for im_topK in range(len(aff_t_im_topK)):
                # get the annotation of this image, it's a row in annotation matrix
                temp_anno = train_anno_filtered[aff_t_im_topK[im_topK], :]
                # calculate weight
                temp_weight = 1 - im_topK / k
                # find if the other given tag appears in the im_topK-th image
                # 0 if no occurrence, or a scalar if occurred
                score = temp_anno[other_tag] * temp_weight
                # add score for all images
                temp_sum = temp_sum + score

            # add the similarity between base tag and other tag on t_im-th image
            scores_t_images.append(temp_sum)
            # print('The similarity score of tag ' + terms_corel5k_filtered[base_tag] + ' and ' +
                  # terms_corel5k_filtered[other_tag] + ' for image ' + str(t_im) + ' is ' + str(temp_sum) + '.')
        # when all images associated with base tag are visited
        scores_t_images = np.array(scores_t_images)
        # calculate the median for all scores of given tag
        median = np.median(scores_t_images)
        print('The similarity score of tag ' + terms_corel5k_filtered[base_tag] +
              ' and ' + terms_corel5k_filtered[other_tag] + ' is ' + str(median) + '.')

        # when similarity between base tag and all other tags are calculated
        score_curr_tag.append(median)

    return score_curr_tag


# use parallel computing to calculate scores for all tags
base_tags = range(len(terms_corel5k_filtered))
pool = multiprocessing.Pool(multiprocessing.cpu_count())
tic()
scores_t_similarity = pool.map(cal_tag_similarity, base_tags)
toc()
pool.close()
pool.join()

utilites.saveVariableToFile(scores_t_similarity, "Corel5k/tag_affinity_matrix.pkl")
utilites.saveVariableToFile(terms_corel5k_filtered, "Corel5k/terms_corel5k_filtered.pkl")
utilites.saveVariableToFile(train_anno_filtered, "Corel5k/train_anno_filtered.pkl")


"""
Since similarity between tag1 & tag2 is not the same with tag2 & tag1
This is normal due to the different visual K nearest neighbour
Now we need to retain only value between the similarity pairs
Considering using mean or min, mean by default
That is to say, similarity = average(sim(tag1, tag2), sim(tag2, tag1))
"""
def adjust_and_norm_affinity(affinity_matrix, method='average'):
    """
    adjust values of the affinity matrix
    for pairwise values, we use average or min of them
    for values of the same tag, we simply set it to 1
    :param affinity_matrix: the affinity matrix need to be processed
    :param method: method to tackle with pairwise values, average by default
    :return: processed affinity matrix
    """
# adjust the values of affinity matrix in order to make it a diagonal one
    for i in range(len(affinity_matrix)):
        for j in range(i + 1, len(terms_corel5k_filtered)):

            temp_tag12 = affinity_matrix[i][j]
            temp_tag21 = affinity_matrix[j][i]
            if method == 'average':
                av_value = (temp_tag12 + temp_tag21) / 2
                affinity_matrix[i][j] = av_value
                affinity_matrix[j][i] = av_value
            else:
                m = min(temp_tag12, temp_tag21)
                affinity_matrix[i][j] = m
                affinity_matrix[j][i] = m

    for k in range(len(affinity_matrix)):
        affinity_matrix[k][k] = 1.0

    return affinity_matrix

# adjust the values of matrix
tag_affinity_matrix = np.asarray(utilites.loadVariableFromFile("Corel5k/tag_affinity_matrix.pkl"))
# we do not want to normalize similarity between the same tag since it make no sense
# so we set the similarity score as inf for same tag: the diagonal of the affinity matrix


def MinMaxScale(data):
    """
    Re-scale given data using min max method
    :param data: data to be scaled
    :return: scaled data
    """
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return scaled_data

# re-scale the data
tag_affinity_matrix_scaled = MinMaxScale(tag_affinity_matrix)
# adjust values of the affinity matrix
tag_affinity_matrix_scaled = adjust_and_norm_affinity(tag_affinity_matrix_scaled)
utilites.saveVariableToFile(tag_affinity_matrix_scaled, "Corel5k/tag_affinity_matrix_scaled.pkl")

"""
Now we got visual similarity of between all tags
Next we will focus on textual similarity between them
"""



