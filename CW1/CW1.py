from pr_method import *
from mat4py import loadmat
import time

start_time = time.time()

# Loading face information in .mat data file
data = loadmat('face.mat')

# Data stored in key 'X' in the library with size 2576*520 and results stored in key 'l' with size 520*1
faces = data.get("X")
results = data.get("l")

# Initialize for classification
pattern_recognition = PRFactory(faces, results, 1)


# Calculating classified accuracy
def compute_accuracy(test_result, actual_result):
    correct = 0
    for i in range(0, len(test_result)):
        if test_result[i] == actual_result[i]:
            correct += 1
    return correct / len(test_result)


# Compute learning result by using Nearest Neighbour classification
learning_result, reconstructed_face = pattern_recognition.pca()

end_time = time.time()

print("Accuracy: ", "{:.2%}".format(compute_accuracy(learning_result, pattern_recognition.test_results)))

print("----- %s seconds -----" % (end_time - start_time))

start_time = time.time()

learning_result = pattern_recognition.low_dim_pca()

end_time = time.time()

print("Accuracy (low-dimension): ", "{:.2%}".format(compute_accuracy(learning_result, pattern_recognition.test_results)))

print("----- %s seconds -----" % (end_time - start_time))

