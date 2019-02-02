# Import modules

import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import sys

def eat_scene(pic_name):
    #TODO: finish this

def read_scene(pic_name):
	data_x = misc.imread(pic_name)

	return (data_x)


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = 3
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def create_sketch(pic_name):
    data_x = read_scene(pic_name)
    flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    #kmeans
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(flattened_image)
    clusters = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    labels = np.asarray(kmeans.labels_,dtype=np.uint8)
    # TODO: image size should be dynamic
    reconstructed_image = recreate_image(clusters,labels,400,400)

    # TODO: save or generate image without show it in console
    # solved? test needed
    plt.imsave("./pics/test_sketch.png", reconstructed_image/255)
    # or in some other format
    # plt.imsave("output.jpg", mat, format="jpg")
