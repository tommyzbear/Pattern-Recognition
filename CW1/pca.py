import numpy as np
import image_data_processor as idp
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
        self.cov_mem_usage = None
        self.eig_vec_mem_usage = None
        self.eig_val_mem_usage = None
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

        # Plot Eigenvalues
        # idp.plot_eig_values(eig_values.real)

        # Check memory usage for Covariance matrix and eigen values, eigen vectors
        self.cov_mem_usage = sys.getsizeof(cov)
        self.eig_vec_mem_usage = sys.getsizeof(eig_vectors)
        self.eig_val_mem_usage = sys.getsizeof(eig_values)

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

    def train_sample_reconstruction(self):
        self.train_sample_reconstructed = idp.sample_reconstruction(self.num_of_train_samples,
                                                                    self.test_sample_projection,
                                                                    self.resolution,
                                                                    self.best_eig_vectors,
                                                                    self.train_avg_vector,
                                                                    self.M)


def avg_face_vector(samples, resolution, num_of_samples):
    faces_sum = np.zeros(resolution)

    for i in range(num_of_samples):
        temp = samples[:, i]
        faces_sum += temp

    return faces_sum / num_of_samples

