import numpy as np
import image_data_processor as idp
import matplotlib.pyplot as plt
import sys


class PCA:
    """Principle Component Analysis"""
    def __init__(self,
                 test_samples,
                 train_samples,
                 train_results,
                 num_of_test_samples,
                 num_of_train_samples,
                 resolution,
                 M,
                 low_dimension=False):
        self.test_sample = test_samples
        self.train_sample = train_samples
        self.train_results = train_results
        self.num_of_test_samples = num_of_test_samples
        self.num_of_train_samples = num_of_train_samples
        self.resolution = resolution
        self.M = M
        self.low_dimension = low_dimension
        self.train_avg_vector = avg_face_vector(train_samples,
                                                self.resolution,
                                                self.num_of_train_samples).reshape(self.resolution, 1)
        self.best_eig_vectors = False
        self.projected = False
        self.train_sample_projection = None
        self.test_sample_projection = None
        self.train_sample_reconstructed = None
        self.test_sample_reconstructed = None
        self.learning_result = None
        # Only used for low dimension case
        self.dimensioned_eig_vectors = None

    def projection(self):
        # normalize training samples
        normalized_train_samples = self.train_sample - self.train_avg_vector

        # normalize test samples
        normalized_test_samples = self.test_sample - self.train_avg_vector

        # Calculate covariance matrix
        if self.low_dimension is False:
            cov = (1 / self.num_of_train_samples) * np.matmul(normalized_train_samples,
                                                              normalized_train_samples.transpose())
        else:
            cov = (1 / self.num_of_train_samples) * np.matmul(normalized_train_samples.transpose(),
                                                              normalized_train_samples)

        # Compute eigen values and eigen vectors of the covariance matrix
        eig_values, eig_vectors = np.linalg.eig(cov)

        # Check memory usage for Covariance matrix and eigen values, eigen vectors
        print("Covariance memory usage: ", sys.getsizeof(cov), " bytes")
        print("Eigen vectors memory usage: ", sys.getsizeof(eig_vectors), " bytes")
        print("Eigen values memory usage: ", sys.getsizeof(eig_values), " bytes")

        # Retrieve largest M eigen value indices in the array
        largest_eig_value_indices = np.argsort(eig_values)[-self.M:]

        # Initialize best eigen vectors
        self.best_eig_vectors = np.zeros((len(cov), self.M), dtype=np.complex)

        # Retrieve corresponding eigen vectors mapping to top M eigen values
        for i in range(0, self.M):
            self.best_eig_vectors[:, i] = eig_vectors[:, largest_eig_value_indices[i]]

        # Compute projections of training samples onto eigen space
        if self.low_dimension is False:
            self.train_sample_projection = np.matmul(normalized_train_samples.transpose(), self.best_eig_vectors)
            self.test_sample_projection = np.matmul(normalized_test_samples.transpose(), self.best_eig_vectors)
        else:
            # Compute eigen vector that matches the dimension using relationship u = Av,
            # where u is eigen vector of size D, v is eigen vector of size N<<D, A is normalized training faces
            self.dimensioned_eig_vectors = np.matmul(normalized_train_samples, self.best_eig_vectors).transpose()
            for v in self.dimensioned_eig_vectors:
                idp.normalization(v)
            self.dimensioned_eig_vectors = self.dimensioned_eig_vectors.transpose()
            self.train_sample_projection = np.matmul(normalized_train_samples.transpose(), self.dimensioned_eig_vectors)
            self.test_sample_projection = np.matmul(normalized_test_samples.transpose(), self.dimensioned_eig_vectors)

        self.projected = True

    def test_sample_reconstruction(self):
        if self.low_dimension is False:
            self.test_sample_reconstructed = idp.sample_reconstruction(self.num_of_test_samples,
                                                                       self.test_sample_projection,
                                                                       self.resolution,
                                                                       self.best_eig_vectors.transpose(),
                                                                       self.train_avg_vector,
                                                                       self.M)
        else:
            self.test_sample_reconstructed = idp.sample_reconstruction(self.num_of_test_samples,
                                                                       self.test_sample_projection,
                                                                       self.resolution,
                                                                       self.dimensioned_eig_vectors.transpose(),
                                                                       self.train_avg_vector,
                                                                       self.M)

        first_reconstructed_image = self.test_sample_reconstructed[:, 0].real.reshape(46, 56).T
        second_reconstructed_image = self.test_sample_reconstructed[:, 1].real.reshape(46, 56).T
        third_reconstructed_image = self.test_sample_reconstructed[:, 2].real.reshape(46, 56).T
        first_test_image = self.test_sample[:, 0].reshape(46, 56).T
        second_test_image = self.test_sample[:, 1].reshape(46, 56).T
        third_test_image = self.test_sample[:, 2].reshape(46, 56).T

        plt.subplot(321)
        plt.title('Actual')
        plt.imshow(first_test_image, cmap='gist_gray')
        plt.subplot(323)
        plt.imshow(second_test_image, cmap='gist_gray')
        plt.subplot(325)
        plt.imshow(third_test_image, cmap='gist_gray')
        plt.subplot(322)
        plt.title('Reconstructed')
        plt.imshow(first_reconstructed_image, cmap='gist_gray')
        plt.subplot(324)
        plt.imshow(second_reconstructed_image, cmap='gist_gray')
        plt.subplot(326)
        plt.imshow(third_reconstructed_image, cmap='gist_gray')
        plt.show()

    def train_sample_reconstruction(self):
        self.train_sample_reconstructed = idp.sample_reconstruction(self.num_of_train_samples,
                                                                    self.test_sample_projection,
                                                                    self.resolution,
                                                                    self.best_eig_vectors,
                                                                    self.train_avg_vector,
                                                                    self.M)

    def compute_result(self):
        self.learning_result = np.zeros(self.num_of_test_samples)
        i = 0
        # Compute learning result by using Nearest Neighbour classification
        for test_projection in self.test_sample_projection:
            error = np.zeros((self.train_sample_projection.shape[0]))
            index = 0
            for train_projection in self.train_sample_projection:
                error[index] = np.linalg.norm(test_projection - train_projection)
                index += 1
            self.learning_result[i] = self.train_results[np.argmin(error)]
            i += 1


def avg_face_vector(samples, resolution, num_of_samples):
    faces_sum = np.zeros(resolution)

    for i in range(num_of_samples):
        temp = samples[:, i]
        faces_sum += temp

    return faces_sum / num_of_samples

