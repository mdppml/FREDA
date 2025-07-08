import itertools
import os
import numpy as np
from GPR import GPR
from scipy.linalg import sqrtm
from scipy.stats import norm
import data as data_mod
from models import *
import util
import pandas as pd
import sklearn.linear_model as lm
import keras.backend as K


class SourceClient:

    def __init__(self, home_path, total_clients, dist, client_id, seed):
        """
        Source client class for FREDA.
        :param home_path: Working directory
        :param total_clients: Number of source clients for the current simulation setup
        :param dist: Which data distribution is currently being used
        :param client_id: id of the client
        :param seed: Random seed for common random mask generation
        """
        setup_dir = os.path.join(home_path, f'data/{total_clients}_client/')
        dist_dir = os.path.join(setup_dir, f'dist_{dist}/')
        self.data_dir = os.path.join(dist_dir, f'{client_id}/')

        self.total_clients = total_clients
        self.id = client_id

        self.X = np.loadtxt(os.path.join(self.data_dir, "x_train.txt"))
        self.Y = np.loadtxt(os.path.join(self.data_dir, "y_train.txt"))

        self.noises = list()
        self.kernel_vars = list()

        self.local_noise = None
        self.local_kernel = None

        self.local_WEN_model = None

        self.seed = seed
        np.random.seed(seed)

        self.no_features = self.X.shape[1]
        self.k = self.no_features + 5  # Can be set as high as desired depending on the privacy requirements

    def get_no_samples(self):
        return self.X.shape[0]

    def compute_masked_hyperparameters(self, feature, peer_ids):
        """
        Computes masked kernel and noise sigmas for a given feature using zero-sum masking.
        :param feature: the index of the feature
        :param peer_ids: list of peer client IDs to generate masks for
        :return: masked kernel sigma, masked noise sigma, masks to share (for debugging/logging)
        """
        kernel_sig, noise_sig = self.compute_hyperparameters(feature)

        total_mask_kernel = 0.0
        total_mask_noise = 0.0
        shared_masks_kernel = {}
        shared_masks_noise = {}

        for peer_id in peer_ids:
            if peer_id == self.id:
                continue

            rng = np.random.default_rng(self.seed + feature)  # deterministic for reproducibility
            r_kernel = rng.normal()
            r_noise = rng.normal()

            # Enforce consistent masking direction
            if self.id < peer_id:
                # We subtract this mask (sending)
                total_mask_kernel += r_kernel
                total_mask_noise += r_noise
                shared_masks_kernel[peer_id] = r_kernel
                shared_masks_noise[peer_id] = r_noise
            else:
                # We add this mask (receiving)
                total_mask_kernel -= r_kernel
                total_mask_noise -= r_noise
                shared_masks_kernel[peer_id] = -r_kernel
                shared_masks_noise[peer_id] = -r_noise

        # Final masked hyperparameters
        self.local_kernel = kernel_sig - total_mask_kernel
        self.local_noise = noise_sig - total_mask_noise

        return shared_masks_kernel, shared_masks_noise

    def get_masked_hyperparameters(self):
        return self.local_kernel, self.local_noise

    def compute_hyperparameters(self, feature):
        """
        Computes the optimal sigma for the linear kernel as well as the optimal sigma for the noise for a given feature
        based on own local data.
        :param feature: The feature for which to compute the kernel's sigma and the noise's sigma
        :return: the optimal kernel and noise sigmas based on own local data
        """
        # Paths to precomputed hyperparameter files
        kernel_sig_path = os.path.join(self.data_dir, "x_kernel_sig.txt")
        noise_sig_path = os.path.join(self.data_dir, "x_noise_sig.txt")

        # Check if precomputed hyperparameters exist
        if os.path.exists(kernel_sig_path) and os.path.exists(noise_sig_path):
            # Load precomputed hyperparameters
            kernel_sig = np.loadtxt(kernel_sig_path)
            noise_sig = np.loadtxt(noise_sig_path)
            return kernel_sig[feature], noise_sig[feature]

        # Compute hyperparameters from scratch if not precomputed
        is_target_feature = np.in1d(np.arange(self.X.shape[1]), feature)
        data_train_x = self.X[:, ~is_target_feature]  # Input features
        data_train_y = self.X[:, is_target_feature]  # Target feature

        gpr = GPR(data_train_x, data_train_y)
        kernel_sig, noise_sig = gpr.optimize()

        return kernel_sig, noise_sig

    def generate_masked_data(self, feature):
        """
        Generates the masked version of the data matrix after taking out the given feature
        :param feature: which feature to extract before masking
        :return: the masked data
        """
        is_feature = np.in1d(np.arange(self.no_features), feature)

        N = np.random.rand(self.k, self.no_features - 1)  # random common matrix for masking generated by all clients
        L = np.linalg.pinv(N)  # left inverse of the random matrix N

        data_train_x = self.X[:, ~is_feature]

        masked_data = data_train_x @ L @ np.real(sqrtm(N @ N.T))

        return masked_data

    def compute_mean_piece(self, mean_piece, feature):
        """
        Computes the dot product between the piece of the mean matrix and own local feature slice.
        :param mean_piece: A sub-matrix of the mean matrix with a dimension of
        N x M where N is the number of local samples
        :param feature: The index of the feature slice
        :return: the resulting vector after taking the dot product
        """
        is_feature = np.in1d(np.arange(self.no_features), feature)

        data_train_y = self.X[:, is_feature]

        return np.dot(mean_piece, data_train_y).flatten()

    def compute_masked_weights(self, weights):
        """
        Applies zero-sum masking to WEN model weights.
        :param weights: List of numpy arrays (model weights)
        :return: masked weights (with this client's mask applied)
        """

        masked_weights = []

        for i, weight in enumerate(weights):
            total_mask = np.zeros_like(weight)

            for peer_id in range(self.total_clients):
                if peer_id == self.id:
                    continue

                # Use deterministic seed per layer for reproducibility
                rng = np.random.default_rng(self.seed + i)

                mask = rng.normal(size=weight.shape)

                if self.id < peer_id:
                    total_mask += mask
                else:
                    total_mask -= mask

            masked_weight = weight - total_mask
            masked_weights.append(masked_weight)

        return masked_weights

    def train_WEN_locally(self, global_model_weights, feature_weights, alpha, lam, current_lr, epochs):
        """
        Performs local training of the Weighted Elastic Net model for the federated learning system. Takes as input
        the hyperparameters for the training and the current global model weights. Updates the weights of the current
        global model for a given number of epochs on own local data and gives back the updated model weights.

        :param global_model_weights: the initial global model weights for the current iteration
        :param feature_weights: weight vector of the weighted elastic net computed from the confidence scores
        :param alpha: penalties of the WEN
        :param lam: the regularization parameter to use during training
        :param current_lr: current global learning rate to use
        :param epochs: how many epochs to run on the local data
        :return: the updated model weights after training
        """

        if not self.local_WEN_model:
            self.local_WEN_model = create_model(self.no_features, alpha, lam, feature_weights, current_lr)

        self.local_WEN_model.set_weights(global_model_weights)

        K.set_value(self.local_WEN_model.optimizer.lr, current_lr)

        self.local_WEN_model.fit(self.X,
                                 self.Y,
                                 epochs=epochs,
                                 batch_size=32,
                                 verbose=0,
                                 shuffle=False
                                 )

        updated_weights = self.local_WEN_model.get_weights()

        masked_weights = self.compute_masked_weights(updated_weights)

        return masked_weights


class TargetClient:

    def __init__(self, home_path, setup, dist, seed, k_value):
        """
        Target client class for FREDA.
        :param home_path: Working directory
        :param setup: Number of source clients for the current simulation setup
        :param dist: Which data distribution is currently being used
        :param seed: Random seed for common random mask generation
        :param k_value: Exponent of the weight function for transforming confidences into weights.
        """
        self.home_path = home_path
        data_dir = os.path.join(home_path, f'data/target_client/')

        setup_dir = os.path.join(home_path, f'data/{setup}_client/')
        self.dist_dir = os.path.join(setup_dir, f'dist_{dist}/')

        self.X = np.loadtxt(os.path.join(data_dir, "x_test.txt"))
        self.Y = np.loadtxt(os.path.join(data_dir, "y_test.txt"))

        self.k_value = k_value

        self.weighted_elastic_net = None
        self.confidences = None

        self.seed = seed
        np.random.seed(seed)

        self.no_features = self.X.shape[1]
        self.k = self.no_features + 5  # Can be set as high as desired depending on the privacy requirements

        # Load phenotype data with actual age and tissue type
        pheno_data_path = os.path.join(os.path.join(self.home_path, "dna_data"), "target_phenotypes.txt")
        pheno_data = pd.read_csv(pheno_data_path, delimiter="\t")
        self.test_tissues = pheno_data['tissue_complete']
        test_tissues_aggregated = self.test_tissues.copy()

        test_tissues_aggregated[test_tissues_aggregated == "whole blood"] = "blood"
        test_tissues_aggregated[test_tissues_aggregated == "menstrual blood"] = "blood"
        test_tissues_aggregated[test_tissues_aggregated == "Brain MedialFrontalCortex"] = "Brain Frontal"

        self.grouping = test_tissues_aggregated
        self.groups = np.unique(self.grouping)

        # Load source and target data
        data_dir = os.path.join(self.home_path, "dna_data")

        # Load source data labels
        y_file = os.path.join(data_dir, "source_y.tsv")
        y_table = pd.read_csv(y_file, header=None)
        y_matrix = np.asfortranarray(y_table)

        normalizer_y = util.HorvathNormalizer

        self.norm_y = normalizer_y(y_matrix)

        self.similarity = None
        self.tissue_combinations = None

    def set_confidences(self, confidences):
        self.confidences = confidences

    def get_groups(self):
        return self.groups

    def get_no_features(self):
        return self.no_features

    def weight_func(self, x):
        return np.power(1 - x, self.k_value)

    def compute_weights(self):
        """
        computes the weights for the WEN training. Can only be called if self.confidences are already computed
        :return: the weight vectors for each tissue in the target domain computed from the confidences scores
        """
        tissue_weights = dict()

        for group in self.groups:
            feature_weights = self.confidences[np.array(group == self.grouping)]
            feature_weights = self.weight_func(np.mean(feature_weights, axis=0))

            current_sum = np.sum(feature_weights)

            # Calculate the scaling factor
            scaling_factor = self.no_features / current_sum
            # Scale each element in the list
            weights = [element * scaling_factor for element in feature_weights]
            tissue_weights[group] = weights

        return tissue_weights

    def generate_masked_data(self, feature):
        """
        Generates the masked version of the data matrix after taking out the given feature
        :param feature: which feature to extract before masking
        :return: the masked data
        """

        is_feature = np.in1d(np.arange(self.no_features), feature)

        N = np.random.rand(self.k, self.no_features - 1)  # random common matrix for masking generated by all clients
        L = np.linalg.pinv(N)  # left inverse of the random matrix N

        data_train_x = self.X[:, ~is_feature]

        masked_data = data_train_x @ L @ np.real(sqrtm(N @ N.T))

        return masked_data

    def compute_confidence_score(self, feature, mean, var):
        """
        Computes the confidence score for the current feature
        :param feature: The feature to compute the confidence score on
        :param mean: the mean of the predicted GPR distribution
        :param var: The variance of the predicted GPR distribution
        :return: The confidence vector
        """
        is_feature = np.in1d(np.arange(self.no_features), feature)

        target_label = self.X[:, is_feature].ravel()

        res_normed = (target_label - mean) / var
        confs = (1 - abs(norm.cdf(res_normed) - norm.cdf(-res_normed)))

        return confs

    def compute_MAE(self, models):
        """
        For a given dict of models, computes the Mean Absolute Error (MAE) on the target data
        :param models: A dict of model objects, where the keys are the tissue names
        :return: The MAE of the models on the target tissues
        """

        results = dict()

        for tissue in models:
            predictions = models[tissue].predict(self.X[self.grouping == tissue, :]).ravel()
            predictions = self.norm_y.denormalize(predictions)
            performances = util.printTestErrors(predictions, self.Y[self.grouping == tissue])[0]
            results[tissue] = performances

        return results

    def eval_model(self, tissue, model):
        """
        Evaluates the given model on the target domain tissue specified and prints the performance of the model
        :param tissue: The tissue data to use
        :param model: The model to evaluate
        :return: the performance of the model on the given tissue samples in a list
        """

        predictions = model.predict(self.X[self.grouping == tissue, :]).ravel()
        predictions = self.norm_y.denormalize(predictions)
        errors_t = util.printTestErrors(predictions, self.Y[self.grouping == tissue],
                                        "Performance on {}:".format(tissue),
                                        indent=4)

        return errors_t

    def read_tissue_similarities(self):
        """
        Reads the tissue similarity translated from the similarities in the GTEx Nature paper. The tissue
        similarities must be in the form of a csv file inside a directory names "tissueSimilarityFromNaturePaper".
        @see: Aguet, F. et al. Genetic effects on gene expression across human tissues. NATURE 550, 204–213 (2017).
        """

        # read similarities translated from GTEx paper
        tissue_similarity = data_mod.TissueSimilarity(self.home_path)

        # Compute similarity of each tissue with training data
        test_tissue_frequencies = self.grouping.value_counts()
        test_tissues_unique = test_tissue_frequencies.index.values
        similarity = pd.Series(index=test_tissues_unique, dtype=np.float64)

        pheno_data_path = os.path.join(os.path.join(self.home_path, "dna_data"), "source_phenotypes.txt")
        pheno_data = pd.read_csv(pheno_data_path, delimiter="\t")

        for t in test_tissues_unique:
            similarity[t] = tissue_similarity.compute_similarity(pheno_data['tissue_detailed'],
                                                                 self.test_tissues[self.grouping == t])

        self.tissue_combinations = list(itertools.combinations(test_tissues_unique[test_tissue_frequencies > 20], 3))
        self.similarity = similarity

    def get_tissue_combinations(self):
        return self.tissue_combinations

    def predict_optimal_lambdas(self, best_lambdas, fit_tissues, holdout_tissues):
        """
        For a given dictionary of the best lambdas, performs optimal lambda prediction using the fit tissues to predict
        the optimal lambda values for the holdout tissues.
        :param best_lambdas: Best lambda values for tissues in the form of a dict
        :param fit_tissues: Which tissues to use during the optimal lambda prediction
        :param holdout_tissues: The tissues to predict the best lambdas on
        :return: The predicted optimal lambda values on the holdout tissues
        """

        # ... fit a line to similarity-lambda relationship
        optimal_lambda_model = lm.LinearRegression()

        optimal_lambda_model.fit(self.similarity[list(fit_tissues)].values.reshape(-1, 1),
                                 np.log([best_lambdas[t] for t in fit_tissues]))

        # ... predict optimal lambdas on hold-out tissues
        predicted_lambdas = np.exp(optimal_lambda_model.predict(self.similarity[holdout_tissues].values.reshape(-1, 1)))
        predicted_lambdas = pd.Series(data=predicted_lambdas, index=holdout_tissues, dtype=np.float64)

        return predicted_lambdas
