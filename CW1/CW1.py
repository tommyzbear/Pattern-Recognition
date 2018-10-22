from mat4py import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
import time

start_time = time.time()

# Loading face information in .mat data file
data = loadmat('face(1).mat')

# Data stored in key 'X' in the library with size 2576*520 and results stored in key 'l' with size 520*1
faces = data.get("X")
results = data.get("l")

faces_arr = np.asarray(faces)
results_arr = np.asarray(results)

# Transpose into matrix with dimension 520*2576
faces_arr = faces_arr.transpose()

# Get resolutions of the images
image_pixels = faces_arr.shape[-1]

# Compute number of images per person (assuming each face has same num of images)
images_per_face = 0
image_count = 0
while images_per_face == 0:
    if results[image_count] == results[image_count + 1]:
        image_count += 1
    else:
        images_per_face = image_count + 1

# Compute number of distinct faces in the data
num_of_distinct_faces = int(faces_arr.shape[0] / images_per_face)

# Define train/test images per face
test_image_per_face = 1
train_image_per_face = images_per_face - test_image_per_face

num_of_total_faces = faces_arr.shape[0]

num_of_train_faces = train_image_per_face * num_of_distinct_faces
num_of_test_faces = test_image_per_face * num_of_distinct_faces

test_ratio = num_of_test_faces / num_of_total_faces

faces_train = np.zeros((num_of_train_faces, image_pixels))
faces_test = np.zeros((num_of_test_faces, image_pixels))
results_train = np.zeros(num_of_train_faces)
results_test = np.zeros(num_of_test_faces)

# Split training samples and test samples
for i in range(0, num_of_distinct_faces):
    start = i * images_per_face
    end = start + images_per_face
    single_face_arr = faces_arr[start: end]
    single_face_result = results_arr[start: end]
    faces_train_temp, faces_test_temp, results_train_temp, results_test_temp = train_test_split(single_face_arr, single_face_result, test_size=test_ratio, random_state=10)
    start_train = start if start == 0 else start - test_image_per_face * i
    end_train = end - test_image_per_face * (i + 1)
    start_test = start if start == 0 else start - train_image_per_face * i
    end_test = end - train_image_per_face * (i + 1)
    faces_train[start_train: end_train] = faces_train_temp
    faces_test[start_test: end_test] = faces_test_temp
    results_train[start_train: end_train] = results_train_temp
    results_test[start_test: end_test] = results_test_temp



# Initialize sum of training faces
face_train_sum = np.zeros((1, image_pixels))

for face in range(num_of_train_faces):
    face_train_sum += faces_train[face]

# Compute average face vector
face_train_avg = face_train_sum / num_of_train_faces

# Initialize covariance matrix
faces_train_covariance = 1 / num_of_train_faces

# Normalized training faces by subtracting average face vector to each face
normalized_faces_train = faces_train - face_train_avg

# Compute covariance matrix
faces_train_covariance *= np.matmul(normalized_faces_train.transpose(), normalized_faces_train)

# Compute all eigen values and corresponding eigen vectors
eigen_values, eigen_vectors = np.linalg.eig(faces_train_covariance)

# num of best eigen vectors
M = 200

# Retrieve largest M eigen value indices in the array
largest_eigen_value_indices = np.argsort(eigen_values)[-M:]

# Initialize best eigen vectors
best_eigen_vectors = np.zeros((M, image_pixels), dtype=np.complex)

# Retrieve corresponding eigen vectors mapping to top M eigen values
for i in range(0, M):
    best_eigen_vectors[i] = eigen_vectors[largest_eigen_value_indices[i]]

# Compute projections of training faces onto eigen space
projections_of_train_faces = np.matmul(normalized_faces_train, best_eigen_vectors.transpose())

train_faces_reconstructed = np.zeros((num_of_train_faces, image_pixels), dtype=np.complex)

# Reconstruct training faces as linear combination of the best M eigen vectors
for i in range(0, projections_of_train_faces.shape[-1]):
    linear_combination_of_eigen_vectors = np.zeros((1, image_pixels), dtype=np.complex)
    for j in range(0, M):
        projection = projections_of_train_faces[i][j]
        eigen_vector = best_eigen_vectors[j]
        linear_combination_of_eigen_vectors += [eig * projection for eig in eigen_vector]
    train_faces_reconstructed[i] = face_train_avg + linear_combination_of_eigen_vectors

# Normalized test samples by subtracting average training face vector
normalized_faces_test = faces_test - face_train_avg

# Compute projections of testing faces onto eigen space
projections_of_test_faces = np.matmul(normalized_faces_test, best_eigen_vectors.transpose())

print(projections_of_train_faces.shape, projections_of_test_faces.shape)


# Initialize learning_results
learning_result = np.zeros(num_of_test_faces)

i = 0

# Compute learning result by using Nearest Neighbour classification
for test_projection in projections_of_test_faces:
    error = np.zeros((projections_of_train_faces.shape[0]))
    index = 0
    for train_projection in projections_of_train_faces:
        error[index] = np.linalg.norm(test_projection - train_projection)
        index += 1
    learning_result[i] = results_train[np.argmin(error)]
    i += 1

print(learning_result)
print(results_test)


def compute_accuracy (test_result, actual_result):
    correct = 0
    for i in range(0, len(test_result)):
        if test_result[i] == actual_result[i]:
            correct += 1
    return correct / len(test_result)


print("Accuracy: ", "{:.2%}".format(compute_accuracy(learning_result, results_test)))

print("----- %s seconds -----" % (time.time() - start_time))
