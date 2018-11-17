from random import randrange
from mat4py import loadmat
from pca_lda import *
from pca import *
from fusion_rules import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def bagging(train_samples, train_results, T, N):
    data_set = np.zeros((T, train_samples.shape[0], N))
    train_results_bag = np.zeros((T, train_results.shape[0]))
    for i in range(0, T):
        for j in range(0, N):
            index = randrange(N)
            data_set[i, :, j] = train_samples[:, index]
            train_results_bag[i, j] = train_results[index]
    return data_set, train_results_bag


# Calculating classified accuracy
def compute_accuracy(test_result, actual_result):
    correct = 0
    for i in range(0, len(test_result)):
        if test_result[i] == actual_result[i]:
            correct += 1
    return correct / len(test_result)


def nearest_neighbour(num_of_test_samples, test_sample_projection, train_sample_projection, train_results):
    learning_result = np.zeros(num_of_test_samples)
    i = 0
    # Compute learning result by using Nearest Neighbour classification
    for test_projection in test_sample_projection:
        error = np.zeros((train_sample_projection.shape[0]))
        index = 0
        for train_projection in train_sample_projection:
            error[index] = np.linalg.norm(test_projection - train_projection)
            index += 1
        learning_result[i] = train_results[np.argmin(error)]
        i += 1

    return learning_result


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

num_of_train_samples, num_of_test_samples, train_samples, test_samples, train_results, test_results = idp.split_train_test(
    num_of_faces, test_image_per_face, images_per_face, num_of_distinct_face, resolution, faces, results)

T = 10
bag_train_samples, train_results_bag = bagging(train_samples, train_results, T, num_of_train_samples)

M0 = 100
results_array_random_feature = np.zeros((T, num_of_test_samples))
results_array_random_data = np.zeros((T, num_of_test_samples))

M_lda = 51

test_sample_projection_array_feature = np.zeros((T, num_of_test_samples, M_lda))
test_sample_projection_array_data = np.zeros((T, num_of_test_samples, M_lda))

probability_array_random_feature = np.zeros((T, num_of_test_samples, num_of_distinct_face))
probability_array_random_data = np.zeros((T, num_of_test_samples, num_of_distinct_face))

for i in range(T):
    M1 = randrange(num_of_train_samples - M0 - 1)
    M_pca = M0 + M1

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

    test_sample_projection_array_feature[i, :, :] = pca_lda_method.test_sample_projection

    # 10,104,51
    results_array_random_feature[i, :] = nearest_neighbour(pca_lda_method.num_of_test_samples,
                                                           pca_lda_method.test_sample_projection,
                                                           pca_lda_method.train_sample_projection,
                                                           pca_lda_method.train_results)

    probability_array_random_feature[i, :] = probability_given_classifier(pca_lda_method)

pca = PCA(test_samples,
          train_samples,
          train_results,
          num_of_test_samples,
          num_of_train_samples,
          resolution,
          128,
          True)

pca.projection()

for i in range(T):
    bag_train_samples, bag_train_results = bagging(train_samples, train_results, T, num_of_train_samples)

    lda_method = LDA(test_samples,
                     bag_train_samples[i],
                     bag_train_results[i],
                     num_of_test_samples,
                     num_of_train_samples,
                     num_of_distinct_face,
                     resolution,
                     M_lda,
                     pca)

    lda_method.fit()

    test_sample_projection_array_data[i, :, :] = lda_method.test_sample_projection

    # 10,104,51
    results_array_random_data[i, :] = nearest_neighbour(lda_method.num_of_test_samples,
                                                        lda_method.test_sample_projection,
                                                        lda_method.train_sample_projection,
                                                        lda_method.train_results)

    probability_array_random_data[i, :] = probability_given_classifier(lda_method)

# Majority voting
majority_result = majority_voting(np.concatenate((results_array_random_feature, results_array_random_data), axis=0))
print("Majority voting Accuracy: ", "{:.2%}".format(compute_accuracy(majority_result, test_results)))

# Sum rule
sum_rule_result = sum_rule(np.concatenate((probability_array_random_feature, probability_array_random_data), axis=0))
print("Sum rule Accuracy: ", "{:.2%}".format(compute_accuracy(sum_rule_result, test_results)))

# Prod rule
# prod_rule_result = prod_rule(np.concatenate((probability_array_random_feature, probability_array_random_data), axis=0))
# print("Prod rule Accuracy: ", "{:.2%}".format(compute_accuracy(prod_rule_result, test_results)))

# Max rule
# max_rule_result = max_rule(np.concatenate((probability_array_random_feature, probability_array_random_data), axis=0))
# print("Max rule Accuracy: ", "{:.2%}".format(compute_accuracy(prod_rule_result, test_results)))


#
# # Averaging
# test_projections_averaging = np.sum(test_sample_projection_array, axis=0)
# test_projections_averaging = np.array(test_projections_averaging) / T
# pca_lda_method = PCA_LDA(test_samples,
#                          train_samples,
#                          train_results,
#                          num_of_test_samples,
#                          num_of_train_samples,
#                          num_of_distinct_face,
#                          resolution,
#                          M_pca,
#                          M_lda)
# pca_lda_method.fit()
# result = nearest_neighbour(pca_lda_method.num_of_test_samples,
#                            pca_lda_method.test_sample_projection,
#                            pca_lda_method.train_sample_projection,
#                            pca_lda_method.train_results)
# print("Averaging Accuracy: ", "{:.2%}".format(compute_accuracy(result, test_results)))

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_results, sum_rule_result)
np.set_printoptions(precision=2)

plt.figure(figsize=(38.4, 21.6))

idp.plot_confusion_matrix(cnf_matrix, classes=list(range(0, 53)),
                          title="Confusion matrix, without normalization \n Ensemble_PCA_LDA Fusion rule")

plt.show()


