from random import randrange
from mat4py import loadmat
from pca_lda import *
from pca import *
from fusion_rules import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


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


def ensemble_random_feature(num_of_train_samples,
                            num_of_test_samples,
                            num_of_distinct_face,
                            T,
                            M0,
                            M_lda):
    print('-----T = {:d}, M0 = {:d}, M_lda = {:d}-----'.format(T, M0, M_lda))
    results_array_random_feature = np.zeros((T, num_of_test_samples))
    probability_array_random_feature = np.zeros((T, num_of_test_samples, num_of_distinct_face))
    # test_sample_projection_array_feature = np.zeros((T, num_of_test_samples, M_lda))
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

        # test_sample_projection_array_feature[i, :, :] = pca_lda_method.test_sample_projection

        # 10,104,51
        results_temp = nearest_neighbour(pca_lda_method.num_of_test_samples,
                                         pca_lda_method.test_sample_projection,
                                         pca_lda_method.train_sample_projection,
                                         pca_lda_method.train_results)
        results_array_random_feature[i, :] = results_temp

        print('Random feature #%i Accuracy:' % (i + 1), '{:.2%}'.format(compute_accuracy(results_temp, test_results)))

        probability_array_random_feature[i, :] = probability_given_classifier(pca_lda_method)

    return results_array_random_feature, probability_array_random_feature


def ensemble_random_data(num_of_train_samples,
                         num_of_test_samples,
                         num_of_distinct_face,
                         train_samples,
                         train_results,
                         T,
                         M_lda):
    pca = PCA(test_samples,
              train_samples,
              train_results,
              num_of_test_samples,
              num_of_train_samples,
              resolution,
              128,
              True)

    pca.projection()

    print('-----T = {:d}, M_pca = {:d}, M_lda = {:d}-----'.format(T, pca.M, M_lda))
    results_array_random_data = np.zeros((T, num_of_test_samples))
    # test_sample_projection_array_data = np.zeros((T, num_of_test_samples, M_lda))
    probability_array_random_data = np.zeros((T, num_of_test_samples, num_of_distinct_face))
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

        # test_sample_projection_array_data[i, :, :] = lda_method.test_sample_projection

        # 10,104,51
        results_temp = nearest_neighbour(lda_method.num_of_test_samples,
                                         lda_method.test_sample_projection,
                                         lda_method.train_sample_projection,
                                         lda_method.train_results)

        results_array_random_data[i, :] = results_temp

        print('Bag #%i Accuracy:' % (i + 1), '{:.2%}'.format(compute_accuracy(results_temp, test_results)))

        probability_array_random_data[i, :] = probability_given_classifier(lda_method)

    return results_array_random_data, probability_array_random_data


def ensemble_random_data_and_feature_in_sequence(num_of_train_samples,
                                                 num_of_test_samples,
                                                 num_of_distinct_face,
                                                 train_samples,
                                                 train_results,
                                                 T,
                                                 M0,
                                                 M_lda):
    print('-----T = {:d}, M_0 = {:d}, M_lda = {:d}-----'.format(T, M0, M_lda))
    results_array_random_data = np.zeros((T ** 2, num_of_test_samples))
    probability_array_random_data = np.zeros((T ** 2, num_of_test_samples, num_of_distinct_face))
    for i in range(T):
        bag_train_samples, bag_train_results = bagging(train_samples, train_results, T, num_of_train_samples)
        results_sub_array_random_data = np.zeros((T, num_of_test_samples))
        probability_sub_array_random_data = np.zeros((T, num_of_test_samples, num_of_distinct_face))
        for j in range(T):
            M1 = randrange(num_of_train_samples - M0 - 1)
            M_pca = M0 + M1

            pca = PCA(test_samples,
                      train_samples,
                      train_results,
                      num_of_test_samples,
                      num_of_train_samples,
                      resolution,
                      M_pca,
                      True)

            pca.projection()

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

            results_sub_array_random_data[j, :] = nearest_neighbour(lda_method.num_of_test_samples,
                                                                    lda_method.test_sample_projection,
                                                                    lda_method.train_sample_projection,
                                                                    lda_method.train_results)

            probability_sub_array_random_data[j, :] = probability_given_classifier(lda_method)

        results_array_random_data[T*i: T*i + T, :] = results_sub_array_random_data

        probability_array_random_data[T*i: T*i + T, :] = probability_sub_array_random_data

    return results_array_random_data, probability_array_random_data


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

# Randomization in feature space only, varying randomness parameter M
print('-----Ensemble learning using randomization in feature space-----')
# Randomization in feature space
M0_list = [24, 30, 40, 50, 100]
for M0 in M0_list:
    results_array_random_feature, probability_array_random_feature = ensemble_random_feature(num_of_train_samples,
                                                                                             num_of_test_samples,
                                                                                             num_of_distinct_face,
                                                                                             T=10,
                                                                                             M0=M0,
                                                                                             M_lda=25)

    # Majority voting
    majority_result = majority_voting(results_array_random_feature)
    print("Majority voting Accuracy: ", "{:.2%}".format(compute_accuracy(majority_result, test_results)))

# Randomization in data samples only, varying base model T
print('\n-----Ensemble learning by randomizing data samples-----')
# Randomization in sample data
T_list = [5, 10, 15, 20, 30]
for T in T_list:
    results_array_random_data, probability_array_random_data = ensemble_random_data(num_of_train_samples,
                                                                                    num_of_test_samples,
                                                                                    num_of_distinct_face,
                                                                                    train_samples,
                                                                                    train_results,
                                                                                    T=T,
                                                                                    M_lda=25)

    # Majority voting
    majority_result = majority_voting(results_array_random_data)
    print("Majority voting Accuracy: ", "{:.2%}".format(compute_accuracy(majority_result, test_results)))

# Randomization in both feature space and data samples in parallel
print('\n-----Ensemble learning by randomizing both feature space and data samples in parallel-----')
results_array_random_feature, probability_array_random_feature = ensemble_random_feature(num_of_train_samples,
                                                                                         num_of_test_samples,
                                                                                         num_of_distinct_face,
                                                                                         T=10,
                                                                                         M0=40,
                                                                                         M_lda=25)

results_array_random_data, probability_array_random_data = ensemble_random_data(num_of_train_samples,
                                                                                num_of_test_samples,
                                                                                num_of_distinct_face,
                                                                                train_samples,
                                                                                train_results,
                                                                                T=10,
                                                                                M_lda=25)

# Majority voting
majority_result = majority_voting(np.concatenate((results_array_random_feature, results_array_random_data), axis=0))
print("Majority voting Accuracy: ", "{:.2%}".format(compute_accuracy(majority_result, test_results)))

# Sum rule
sum_rule_result = sum_rule(np.concatenate((probability_array_random_feature, probability_array_random_data), axis=0))
print("Sum rule Accuracy: ", "{:.2%}".format(compute_accuracy(sum_rule_result, test_results)))

# Prod rule
prod_rule_result = prod_rule(np.concatenate((probability_array_random_feature, probability_array_random_data), axis=0))
print("Prod rule Accuracy: ", "{:.2%}".format(compute_accuracy(prod_rule_result, test_results)))

# Max rule
max_rule_result = max_rule(np.concatenate((probability_array_random_feature, probability_array_random_data), axis=0))
print("Max rule Accuracy: ", "{:.2%}".format(compute_accuracy(prod_rule_result, test_results)))

sns.heatmap(sum(probability_array_random_data), linewidths=0.5)
plt.ylabel('Test sample index')
plt.xlabel('Class index')
plt.title('Heat map for probability of choosing class $c_i$ given test sample $x_i$' + '\nRandomization in sample data')
plt.show()

sns.heatmap(sum(probability_array_random_feature), linewidths=0.5)
plt.ylabel('Test sample index')
plt.xlabel('Class index')
plt.title('Heat map for probability of choosing class $c_i$ given test sample $x_i$' + '\nRandomization in feature space')
plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_results, sum_rule_result)
np.set_printoptions(precision=2)

plt.figure(figsize=(38.4, 21.6))

idp.plot_confusion_matrix(cnf_matrix, classes=list(range(0, 53)),
                          title="Confusion matrix, without normalization \n Ensemble_PCA_LDA Fusion rule")
plt.show()

# Randomization in both feature space and data samples in sequence
print('\n-----Ensemble learning by randomizing both data samples and feature space in sequence-----')
results_array_randomisation_sequence, probability_array_randomisation_sequence = \
    ensemble_random_data_and_feature_in_sequence(num_of_train_samples,
                                                 num_of_test_samples,
                                                 num_of_distinct_face,
                                                 train_samples,
                                                 train_results,
                                                 T=10,
                                                 M0=50,
                                                 M_lda=25)

# Majority voting
majority_result = majority_voting(results_array_randomisation_sequence)
print("Majority voting Accuracy: ", "{:.2%}".format(compute_accuracy(majority_result, test_results)))

# Sum rule
sum_rule_result = sum_rule(probability_array_randomisation_sequence)
print("Sum rule Accuracy: ", "{:.2%}".format(compute_accuracy(sum_rule_result, test_results)))

# Prod rule
prod_rule_result = prod_rule(probability_array_randomisation_sequence)
print("Prod rule Accuracy: ", "{:.2%}".format(compute_accuracy(prod_rule_result, test_results)))

# Max rule
max_rule_result = max_rule(probability_array_randomisation_sequence)
print("Max rule Accuracy: ", "{:.2%}".format(compute_accuracy(prod_rule_result, test_results)))

# # committee machine
# test_sample_projection_array_concat = np.concatenate((test_sample_projection_array_feature,
#                                                       test_sample_projection_array_data), axis=0)
# committee_machine_sum = np.sum(test_sample_projection_array_concat, axis=0)
# committee_machine = np.array(committee_machine_sum)/T
# print('hi')

