from pca_lda import *
import time
from mat4py import loadmat


# Nearest Neighbour classification
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


# Calculating classified accuracy
def compute_accuracy(test_result, actual_result):
    correct = 0
    for i in range(0, len(test_result)):
        if test_result[i] == actual_result[i]:
            correct += 1
    return correct / len(test_result)


# Loading face information in .mat data file
data = loadmat('face.mat')

# Data stored in key 'X' in the library with size 2576*520 and results stored in key 'l' with size 520*1
faces = np.asarray(data.get("X"))
results = np.asarray(data.get("l"))

# State test image per face
test_image_per_face = 2

# Define M_pca values and M_lda values
M_pca_list = [67, 128, 290, 364]
M_lda_list = [5, 10, 20, 30, 40, 51]

resolution = faces.shape[0]
num_of_faces = faces.shape[-1]
images_per_face = idp.images_per_person(results)
num_of_distinct_face = idp.distinct_faces_num(num_of_faces, images_per_face)

num_of_train_samples, num_of_test_samples, train_samples, test_samples, train_results, test_results = idp.split_train_test(
    num_of_faces, test_image_per_face, images_per_face, num_of_distinct_face, resolution, faces, results)

# Varying M_pca while keeping M_lda constant
M_lda = 51
print("-----Varying M_pca keeping M_lda = %i-----" % M_lda)
for M_pca in M_pca_list:
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
    print("M_pca = %i, " % M_pca, "Accuracy: ", "{:.2%}".format(compute_accuracy(results, test_results)))

# Varying M_lda while keeping M_pca constant
M_pca = 290
print("-----Varying M_lda keeping M_pca = %i-----" % M_pca)
for M_lda in M_lda_list:
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
    print("M_lda = %i, " % M_lda, "Accuracy: ", "{:.2%}".format(compute_accuracy(results, test_results)))