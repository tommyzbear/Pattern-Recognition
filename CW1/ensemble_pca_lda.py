from random import randrange
from mat4py import loadmat
import numpy as np
import image_data_processor as idp


def bagging(train_samples, T, N):
    data_set = []
    for i in range(0, T):
        subset = []
        for j in range(0, N):
            index = randrange(train_samples.shape[-1])
            temp = train_samples[:, index]
            subset.append(temp)
        data_set.append(temp)
    return data_set


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

bag = bagging(train_samples, 10, train_samples.shape[-1])

M_pca = 128
M_lda = 51

