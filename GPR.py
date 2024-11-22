import numpy as np
import GPy


class GPR:

    def __init__(self, X: np.array = None, Y: np.array = None, kernel_var=1.0, noise=1.0):
        self.X = X
        self.Y = Y
        self.noise = noise
        self.kernel_var = kernel_var

    def optimize(self):
        kernel = GPy.kern.Linear(input_dim=self.X.shape[1])
        model = GPy.models.GPRegression(self.X, self.Y, kernel)
        model.optimize()

        self.kernel_var = model.param_array[0]
        self.noise = model.param_array[1]

        return self.kernel_var, self.noise

    def predict(self, x_new):
        kernel = GPy.kern.Linear(input_dim=self.X.shape[1], variances=self.kernel_var)

        K = kernel.K(self.X)  # Training data covariance matrix
        K_star = kernel.K(x_new, self.X)  # Covariance matrix between test and train data
        K_star_star = kernel.K(x_new)  # Test data covariance matrix

        inverse_of_covariance_matrix_of_input = np.linalg.inv(K + self.noise * np.eye(len(self.X)))

        # Mean.
        mean = np.dot(K_star, np.dot(self.Y.ravel(), inverse_of_covariance_matrix_of_input.T).T).flatten()

        # Variance.
        cov = np.diag(K_star_star) - np.dot(K_star, np.dot(inverse_of_covariance_matrix_of_input, K_star.T))
        var = np.sqrt(np.diag(cov))

        return mean, var

    def predict2(self, x_new):
        kernel = GPy.kern.Linear(input_dim=self.X.shape[1], variances=self.kernel_var)

        # Compute kernel matrices
        K = kernel.K(self.X)  # Training data covariance matrix
        K_star = kernel.K(self.X, x_new)  # Covariance matrix between test and train data
        K_star_star = kernel.K(x_new)  # Test data covariance matrix

        L = np.linalg.cholesky(K + self.noise * np.eye(len(self.X)))  # line 1

        Lk = np.linalg.solve(L, K_star)  # k_star = kernel(X, Xtest), calculating v := l\k_star
        mu = np.dot(Lk.T, np.linalg.solve(L, self.Y.ravel()))  # \alpha = np.linalg.solve(L, y)

        s2 = np.diag(K_star_star) - np.sum(Lk ** 2, axis=0)
        s = np.sqrt(s2)

        return mu, s

    def predict_from_matrices(self, K, K_star, K_star_star):
        inverse_of_covariance_matrix_of_input = np.linalg.inv(K + self.noise * np.eye(len(self.X)))

        # Mean.
        mean = np.dot(K_star, np.dot(self.Y, inverse_of_covariance_matrix_of_input.T).T).flatten()

        # Variance.
        cov = np.diag(K_star_star) - np.dot(K_star, np.dot(inverse_of_covariance_matrix_of_input, K_star.T))
        var = np.sqrt(np.diag(cov))

        return mean, var
