from collections import Counter
import numpy as np


#def probabilistic_summation(results_array):
#def probabilistic_multiplication(results_array):

def majority_voting(results_array):
    majority_results = np.zeros(results_array.shape[-1])
    for i in range(0, results_array.shape[-1]):
        majority_results[i] = Counter(results_array[:, i]).most_common(1)[0][0]
    return majority_results


def probability_given_classifier(lda):
    w_i = lda.opt_eig_vec @ lda.class_mean
    w_x = lda.opt_eig_vec @ lda.test_sample
    numerator = w_x.transpose() @ w_i
    denominator = np.linalg.norm(w_x) * np.linalg.norm(w_i)
    return ((1 + (numerator / denominator)) / 2).real


#def sum_rule(results_array):


#def probabilistic_max(results_array)：


#def probabilistic_min(results_array)：