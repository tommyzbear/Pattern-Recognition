from mat4py import loadmat
import time
from pca_lda import *
from pca import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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

# PCA method using Nearest Neighbour classification
print("----- PCA NN_Classification -----")
pca_method = PCA(test_samples,
                 train_samples,
                 train_results,
                 num_of_test_samples,
                 num_of_train_samples,
                 resolution,
                 M_pca)

start_time = time.time()

pca_method.projection()

end_time = time.time()

print("Compute Time: %s seconds" % (end_time - start_time))

results = nearest_neighbour(pca_method)

print("Accuracy: ", "{:.2%}".format(compute_accuracy(results, test_results)))

pca_method.test_sample_reconstruction()

idp.print_image(pca_method.train_avg_vector)

idp.image_comparison(pca_method)

print("Covariance memory usage: ", pca_method.cov_mem_usage, " bytes")
print("Eigen vectors memory usage: ", pca_method.eig_vec_mem_usage, " bytes")
print("Eigen values memory usage: ", pca_method.eig_val_mem_usage, " bytes")

# PCA method low dimension
print("----- PCA_Low-Dimension NN_Classification -----")
pca_method_low = PCA(test_samples,
                     train_samples,
                     train_results,
                     num_of_test_samples,
                     num_of_train_samples,
                     resolution,
                     M_pca,
                     True)

start_time = time.time()

pca_method_low.projection()

end_time = time.time()

print("Compute Time: %s seconds" % (end_time - start_time))

results = nearest_neighbour(pca_method_low)

print("Accuracy: ", "{:.2%}".format(compute_accuracy(results, test_results)))

pca_method_low.test_sample_reconstruction()

idp.image_comparison(pca_method_low)

idp.false_correct_image(results, test_results, test_samples, pca_method_low)

print("Covariance memory usage: ", pca_method_low.cov_mem_usage, " bytes")
print("Eigen vectors memory usage: ", pca_method_low.eig_vec_mem_usage, " bytes")
print("Eigen values memory usage: ", pca_method_low.eig_val_mem_usage, " bytes")

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_results.tolist(), results.tolist())
np.set_printoptions(precision=2)

plt.figure(figsize=(38.4, 21.6))

idp.plot_confusion_matrix(cnf_matrix, classes=list(range(0, 53)),
                          title="Confusion matrix, without normalization \n PCA NN method")

plt.show()

# Alternative method
print("----- PCA_Low-Dimension Alternative method -----")
M_Alternative_pca = 5

train_image_per_face = images_per_face - test_image_per_face

# Initialize error matrix
errors = np.zeros((num_of_test_samples, num_of_distinct_face))
index = 0

start_time = time.time()

for i in range(0, num_of_train_samples, train_image_per_face):
    pca_method_low = PCA(test_samples,
                         train_samples[:, i: i + train_image_per_face],
                         train_results,
                         num_of_test_samples,
                         train_image_per_face,
                         resolution,
                         M_Alternative_pca,
                         True)
    pca_method_low.projection()
    pca_method_low.test_sample_reconstruction()
    test_sample_reconstructed = pca_method_low.test_sample_reconstructed

    for j in range(test_sample_reconstructed.shape[-1]):
        errors[j, index] = np.linalg.norm(test_samples[:, j] - test_sample_reconstructed[:, j])

    index += 1

results = np.zeros(num_of_test_samples)
for i in range(0, num_of_test_samples):
    results[i] = np.argmin(errors[i]) + 1

end_time = time.time()

print("Compute Time: %s seconds" % (end_time - start_time))

print("Accuracy: ", "{:.2%}".format(compute_accuracy(results, test_results)))

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_results.tolist(), results.tolist())
np.set_printoptions(precision=2)

plt.figure(figsize=(38.4, 21.6))

idp.plot_confusion_matrix(cnf_matrix, classes=list(range(0, 53)),
                          title="Confusion matrix, without normalization \n PCA Alternative method")

plt.show()

# PCA_LDA method
print("----- PCA_LDA NN_Classification -----")
# Define M for LDA
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

start_time = time.time()

pca_lda_method.fit()

end_time = time.time()

print("Compute Time: %s seconds" % (end_time - start_time))

results = nearest_neighbour(pca_lda_method)

print("Accuracy: ", "{:.2%}".format(compute_accuracy(results, test_results)))

idp.false_correct_image(results, test_results, test_samples, pca_lda_method, is_lda=True)

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_results.tolist(), results.tolist())
np.set_printoptions(precision=2)

plt.figure(figsize=(38.4, 21.6))

idp.plot_confusion_matrix(cnf_matrix, classes=list(range(0, 53)),
                          title="Confusion matrix, without normalization \n PCA_LDA NN method")

plt.show()
