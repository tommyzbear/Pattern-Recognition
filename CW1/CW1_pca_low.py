from mat4py import loadmat
import time
from pca import *
import os
import psutil


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_full_info().uss / float(1 << 20)
    return mem


# Calculating classified accuracy
def compute_accuracy(test_result, actual_result):
    correct = 0
    for i in range(0, len(test_result)):
        if test_result[i] == actual_result[i]:
            correct += 1
    return correct / len(test_result)


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

# PCA method low dimension
print("----- PCA_Low-Dimension NN_Classification -----")

start_time = time.time()
pca_method_low = PCA(test_samples,
                     train_samples,
                     train_results,
                     num_of_test_samples,
                     num_of_train_samples,
                     resolution,
                     M_pca,
                     True)

pca_method_low.projection()

end_time = time.time()

print("Compute Time: %s seconds" % (end_time - start_time))

results = nearest_neighbour(pca_method_low)

print("Accuracy: ", "{:.2%}".format(compute_accuracy(results, test_results)))

print("Memory Usage: %0.2f MB" % memory_usage_psutil())
