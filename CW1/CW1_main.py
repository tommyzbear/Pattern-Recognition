from mat4py import loadmat
import time
from pca import *


# Calculating classified accuracy
def compute_accuracy(test_result, actual_result):
    correct = 0
    for i in range(0, len(test_result)):
        if test_result[i] == actual_result[i]:
            correct += 1
    return correct / len(test_result)


start_time = time.time()

# Loading face information in .mat data file
data = loadmat('face.mat')

# Data stored in key 'X' in the library with size 2576*520 and results stored in key 'l' with size 520*1
faces = np.asarray(data.get("X"))
results = np.asarray(data.get("l"))

# State test image per face
test_image_per_face = 2

# State size of M
M = 128

resolution = faces.shape[0]
num_of_faces = faces.shape[-1]
images_per_face = idp.images_per_person(results)
num_of_distinct_face = idp.distinct_faces_num(num_of_faces, images_per_face)

num_of_train_samples, num_of_test_samples, train_samples, test_samples, train_results, test_results = idp.split_train_test(
    num_of_faces, test_image_per_face, images_per_face, num_of_distinct_face, resolution, faces, results)

# PCA method
print("----- PCA -----")
pca_method = PCA(test_samples,
                 train_samples,
                 train_results,
                 num_of_test_samples,
                 num_of_train_samples,
                 resolution,
                 M)

start_time = time.time()

pca_method.projection()

end_time = time.time()

print("Compute Time: %s seconds" % (end_time - start_time))

pca_method.compute_result()

print("Accuracy: ", "{:.2%}".format(compute_accuracy(pca_method.learning_result, test_results)))

pca_method.test_sample_reconstruction()

# PCA method low dimension
print("-----PCA_Low-Dimension-----")
pca_method_low = PCA(test_samples,
                     train_samples,
                     train_results,
                     num_of_test_samples,
                     num_of_train_samples,
                     resolution,
                     M,
                     True)

start_time = time.time()

pca_method_low.projection()

end_time = time.time()

print("Compute Time: %s seconds" % (end_time - start_time))

pca_method_low.compute_result()

print("Accuracy: ", "{:.2%}".format(compute_accuracy(pca_method.learning_result, test_results)))

pca_method_low.test_sample_reconstruction()
