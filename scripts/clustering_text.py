__author__ = 'TonySun'
"""
After we construct visual similarity and textual similarity matrics,
we can perform spectral clustering to group similar terms together
"""

import utilites
from sklearn.cluster import spectral_clustering
import numpy as np
from scipy.special import expit


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
        for j in range(i + 1, len(affinity_matrix)):

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

    scaled_data = expit(affinity_matrix)

    for k in range(len(scaled_data)):
        scaled_data[k][k] = 1.0

    return scaled_data


def perform_clustering(alpha=0.0, num_clusters=100):
    """
    clustering the tag/terms and return the cluster ids for each tag
    :param alpha: parameter to combine visual and textual similarity matrix
    :param num_clusters: number of clusters/concepts obtained
    :return: cluster ids for each tag
    """
    vis_sim_mat = utilites.loadVariableFromFile("Corel5k/tag_affinity_matrix_scaled.pkl")
    tex_sim_mat = utilites.loadVariableFromFile("Corel5k/tag_textual_similarity_matrix.pkl")

    tex_sim_mat = adjust_and_norm_affinity(tex_sim_mat)
    vis_sim_mat = expit(vis_sim_mat)

    # introduce a parameter alpha to merge the two matrics
    joint_mat = alpha * vis_sim_mat + (1 - alpha) * tex_sim_mat

    # let's start spectrum clustering
    # obtain cluster IDs for each word
    # eigen_solver: None, arpack, lobpcg, or amg
    cluster_ids = spectral_clustering(joint_mat, n_clusters=num_clusters, eigen_solver='arpack')
    print("Done...")
    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    words = utilites.loadVariableFromFile("Corel5k/terms_corel5k_filtered.pkl")
    word_centroid_map = dict(zip(words, cluster_ids))
    utilites.saveVariableToFile(cluster_ids, "Corel5k/concepts_ids.pkl")

    cluster_contents = []
    # For the first 10 clusters
    for cluster in range(0, num_clusters):
        # print the cluster number
        print("\nCluster %d" % cluster)
        # Find all of the words for that cluster number, and print them out
        r_words = []
        for i in range(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                r_words.append(word_centroid_map.keys()[i])

        print (r_words)
        cluster_contents.append(r_words)

    utilites.saveVariableToFile(cluster_contents, "Corel5k/cluster_contents.pkl")

    return cluster_ids


def get_concept_anno():
    """
    build new concept annotation matrix upon old tag based annotation
    :param num_clusters: number of clusters/concepts
    :return: new concept based annotation matrix
    """
    cluster_ids = utilites.loadVariableFromFile("Corel5k/concepts_ids.pkl")
    # all tags
    words = utilites.loadVariableFromFile("Corel5k/terms_corel5k_filtered.pkl")
    # all tag ids from 1 to length of cluster_ids
    word_ids = range(len(cluster_ids))
    # get number of clusters by counting unique cluster ids
    num_clusters = len(set(cluster_ids))
    # construct to indicate which cluster does the given word belong to
    cluster_map = dict(zip(word_ids, cluster_ids))
    # load the original tag annotation matrix
    word_anno = utilites.loadVariableFromFile("Corel5k/train_anno_filtered.pkl")

    # initialize a zero matrix as concept matrix
    anno = np.zeros((len(word_anno), num_clusters), dtype=np.int)

    # for every instance in the anno
    for i in range(len(word_anno)):
        print('This is instance ' + str(i) + '.')
        # for every tag in all tags
        for j in range(len(cluster_ids)):
            # if this tag appears in the original tag annotation  matrix
            if word_anno[i][j] == 1:
                # we first find which concept this tag belongs to
                # and then set the occurrence of this concept is 1
                anno[i][cluster_map[j]] = 1
                print("The words is " + words[j] + ", and the concept is " + str(cluster_map[j]))

    utilites.saveVariableToFile(anno, "Corel5k/train_anno_concept.pkl")

    return anno


def display_clusters(num_clusters):
    cluster_ids = utilites.loadVariableFromFile("Corel5k/concepts_ids.pkl")
    words = utilites.loadVariableFromFile("Corel5k/terms_corel5k_filtered.pkl")
    word_centroid_map = dict(zip(words, cluster_ids))
    cluster_contents = []
    # For the first 10 clusters
    for cluster in range(0, num_clusters):
        # print the cluster number
        print("\nCluster %d" % cluster)
        # Find all of the words for that cluster number, and print them out
        r_words = []
        for i in range(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                r_words.append(word_centroid_map.keys()[i])

        print (r_words)
        cluster_contents.append(r_words)

