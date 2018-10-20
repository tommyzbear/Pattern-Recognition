from mat4py import loadmat
import numpy as np
from sklearn.model_selection import train_test_split

data = loadmat('face(1).mat')

faces = data.get("X")
results = data.get("l")

faces_arr = np.asarray(faces)
results_arr = np.asarray(results)

faces_arr = faces_arr.transpose()

image_pixels = faces_arr.shape[-1]

#print(faces_arr.shape, results_arr.shape)

#print(type(faces_arr), type(results_arr))

faces_train, faces_test, results_train, results_test = train_test_split(faces_arr, results_arr, test_size=0.3, random_state=10)

#print(faces_train.shape, faces_test.shape)

face_train_sum = np.zeros((1, image_pixels))

faces_train_num = faces_train.shape[0]

for face in range(faces_train_num):
    face_train_sum += faces_train[face]

face_train_avg = face_train_sum / faces_train_num

#print(face_avg)

faces_train_covariance = 1 / faces_train_num

normalized_faces_train = faces_train - face_train_avg

faces_train_covariance *= np.matmul(normalized_faces_train.transpose(), normalized_faces_train)

#print(faces_train_covariance.shape, '\n', faces_train_covariance)

eigen_values, eigen_vectors = np.linalg.eig(faces_train_covariance)

#print(eigen_values, "\n", eigen_vectors)

# num of best eigen vectors
M = 50

largest_eigen_value_indices = np.argsort(eigen_values)[-50:]

best_eigen_vectors = np.zeros((M, image_pixels), dtype=np.complex)

for i in range(0, M):
    best_eigen_vectors[i] = eigen_vectors[largest_eigen_value_indices[i]]

projections_of_train_faces = np.matmul(normalized_faces_train, best_eigen_vectors.transpose())

train_faces_reconstructed = np.zeros((faces_train_num, image_pixels))

for i in range(0, projections_of_train_faces.shape[-1]):
    linear_combination_of_eigen_vectors = np.zeros((1, image_pixels), dtype=np.complex)
    for j in range(0, M):
        projection = projections_of_train_faces[i][j]
        eigen_vector = best_eigen_vectors[j]
        linear_combination_of_eigen_vectors += [eig * projection for eig in eigen_vector]
    train_faces_reconstructed[i] = face_train_avg + linear_combination_of_eigen_vectors

print(train_faces_reconstructed.shape)
