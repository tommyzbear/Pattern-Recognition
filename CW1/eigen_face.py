import numpy as np


# This class is for obtaining eigenface, this includes calculating avg face vector, normalizing face vector, obtaining
# covariance matrix, eigenvectors and eigenvalues, projecting training faces onto eigen space
class EigenFace:
    def __init__(self, samples, resolutions, num_of_faces, low_dimension):
        self.face_avg_vector = avg_face_vector(samples, resolutions, num_of_faces)
        self.normalized_face = samples - self.face_avg_vector
        self.covariance = \
            (1 / num_of_faces) * np.matmul(self.normalized_face.transpose(), self.normalized_face) \
            if low_dimension is False \
            else (1 / num_of_faces) * np.matmul(self.normalized_face, self.normalized_face.transpose())
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.covariance)
        self.best_eigen_vectors = compute_best_eigen_vectors(self.eigen_values, self.eigen_vectors, len(self.covariance))
        self.projections_of_faces = np.matmul(self.normalized_face, self.best_eigen_vectors.transpose())


# Initialize sum of training faces
def avg_face_vector(samples, resolutions, num_of_faces):
    faces_sum = np.zeros((1, resolutions))

    for i in range(num_of_faces):
        faces_sum += samples[i]

    return faces_sum / num_of_faces


def compute_best_eigen_vectors(eigen_values, eigen_vectors, size):
    # finding how many eigenfaces needed to represent 95% total variance
    eig_value_sum = sum(eigen_values)
    sum_temp = 0
    M = 0
    firstfound = 0
    for i in range(0, len(eigen_values)):
        sum_temp = sum_temp + eigen_values[i]
        tv = sum_temp / eig_value_sum
        if tv > 0.95 and firstfound == 0:
            M = i + 1
            firstfound = 1

    # Retrieve largest M eigen value indices in the array
    largest_eigen_value_indices = np.argsort(eigen_values)[-M:]

    # Initialize best eigen vectors
    best_eigen_vectors = np.zeros((M, size), dtype=np.complex)

    # Retrieve corresponding eigen vectors mapping to top M eigen values
    for i in range(0, M):
        best_eigen_vectors[i] = eigen_vectors[largest_eigen_value_indices[i]]

    return best_eigen_vectors
