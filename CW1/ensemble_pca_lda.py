from random import randrange
from mat4py import loadmat
import numpy as np
import image_data_processor as idp
from pca import *
from pca_lda import *


def bagging(train_samples, train_results, T, N):
    data_set = np.zeros((T, train_samples.shape[0], N))
    train_results_bag = np.zeros((T, train_results.shape[0]))
    for i in range(0, T):
        for j in range(0, N):
            index = randrange(train_samples.shape[-1])
            data_set[i, :, j] = train_samples[:, index]
            train_results_bag[i, j] = train_results[index]
    return data_set, train_results_bag


def random_in_feature(bag, M0, Ntrain):
    M1 = randrange(Ntrain - M0 - 1)
    M_feature = M0 + M1

def nearest_neighbour(pca):
    learning_result = np.zeros(pca.num_of_test_samples)
    i = 0
    # Compute learning result by using Nearest Neighbour classification
    for test_projection in pca.test_sample_projection:
        error = np.zeros((pca.train_sample_projection.shape[0]))
        index = 0
        for train_projection in pca.train_sample_projection:
            error[index] = np.linalg.norm(test_projection - train_projection)
            index += 1
        learning_result[i] = pca.train_results[np.argmin(error)]
        i += 1

    return learning_result

# Loading face information in .mat data file
data = loadmat('face.mat')

# Data stored in key 'X' in the library with size 2576*520 and results stored in key 'l' with size 520*1
faces = np.asarray(data.get("X"))
results = np.asarray(data.get("l"))

# State test image per face
test_image_per_face = 2

# State size of M, 128 is 95% of the covariances
M_pca = 128

resolution = faces.shape[0]
num_of_faces = faces.shape[-1]
images_per_face = idp.images_per_person(results)
num_of_distinct_face = idp.distinct_faces_num(num_of_faces, images_per_face)

num_of_train_samples, num_of_test_samples, train_samples, test_samples, train_results, test_results = idp.split_train_test(
    num_of_faces, test_image_per_face, images_per_face, num_of_distinct_face, resolution, faces, results)

T = 10
N = train_samples.shape[-1]
bag, train_results_bag = bagging(train_samples, train_results, T, N)

M0 = 100

for i in range(T):
    M1 = randrange(N - M0 - 1)
    M_pca = M0 + M1
    # Get low-dimension PCA training projections
    train_samples = bag[i, :, :]
    train_results = train_results_bag[i, :]

    M_lda = 51

    pca_lda_method = PCA_LDA(test_samples,
                             train_samples,
                             train_results,
                             num_of_test_samples,
                             num_of_train_samples,
                             num_of_distinct_face,
                             resolution,
                             M_pca,
                             M_lda)

    pca_lda_method.fit()
    results = nearest_neighbour(pca_lda_method)
