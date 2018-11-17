from collections import Counter
import numpy as np



def probabilistic_summation(results_array):
def probabilistic_multiplication(results_array):
def majority_voting(results_array):
    majority_results = np.zeros(results_array.shape[-1])
    for i in range(0, results_array.shape[-1]):
        majority_results[i] = Counter(results_array[:, i]).most_common(1)ssss
    return majority_results

def probabilistic_max(results_array)：


def probabilistic_min(results_array)：