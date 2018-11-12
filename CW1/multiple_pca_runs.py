from mat4py import loadmat
from pca import *
from operator import add
import matplotlib.pyplot as plt


def pca_with_diff_M(M_list, test_samples, train_samples, train_results, num_of_test_samples, num_of_train_samples, resolution):
    accuracy_list = []
    for M in M_list:
        pca_method_low = PCA(test_samples,
                             train_samples,
                             train_results,
                             num_of_test_samples,
                             num_of_train_samples,
                             resolution,
                             M,
                             True)

        pca_method_low.projection()

        results = nearest_neighbour(pca_method_low)

        accuracy = compute_accuracy(results, test_results)

        accuracy_list.append(accuracy)

    return accuracy_list


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

resolution = faces.shape[0]
num_of_faces = faces.shape[-1]
images_per_face = idp.images_per_person(results)
num_of_distinct_face = idp.distinct_faces_num(num_of_faces, images_per_face)
'''
M_list = [27, 41, 67, 128, 290]

run_num = 10

accuracy_list = [0]*len(M_list)

for i in range(0, run_num):
    num_of_train_samples, num_of_test_samples, train_samples, test_samples, train_results, test_results = idp.split_train_test(
        num_of_faces, test_image_per_face, images_per_face, num_of_distinct_face, resolution, faces, results)

    accuracy_temp = pca_with_diff_M([27, 41, 67, 128, 290],
                                    test_samples,
                                    train_samples,
                                    train_results,
                                    num_of_test_samples,
                                    num_of_train_samples,
                                    resolution)

    accuracy_list = list(map(add, accuracy_list, accuracy_temp))

accuracy_avg = [element / run_num for element in accuracy_list]

print(accuracy_avg)
'''
# PCA accuracy versus M
num_of_train_samples, num_of_test_samples, train_samples, test_samples, train_results, test_results = idp.split_train_test(
        num_of_faces, test_image_per_face, images_per_face, num_of_distinct_face, resolution, faces, results)

M_list = range(0, 415)
accuracy_list = []

for M in M_list:
    pca_method_low = PCA(test_samples,
                         train_samples,
                         train_results,
                         num_of_test_samples,
                         num_of_train_samples,
                         resolution,
                         M,
                         True)

    pca_method_low.projection()

    results = nearest_neighbour(pca_method_low)

    accuracy = compute_accuracy(results, test_results)

    accuracy_list.append(accuracy)

plt.figure()
accuracy_list = [i * 100 for i in accuracy_list]
plt.plot(M_list, accuracy_list)
plt.xlabel(r'$M_{pca}$')
plt.ylabel('Accuracy %')
plt.title(r'PCA Result Accuracy With different $M_{pca}$')
plt.show()

print('Finished')
