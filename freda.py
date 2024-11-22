import os

import numpy as np
from clients import SourceClient, TargetClient
from models import *
import pandas as pd


class FREDA:

    def __init__(self, home_path, setup, dist, seed, k_value):
        """
        The main class which implements the FREDA framework.
        :param home_path: Current working directory
        :param setup: The number of source clients
        :param dist: Which data distribution to run
        :param seed: The common random seed for the clients for masking
        :param k_value: Exponent of the weight function for transforming confidences into weights.
        """
        self.no_clients = setup

        self.seed = seed

        self.dist_dir = os.path.join(os.path.join(home_path, f'data/{setup}_client/'), f'dist_{dist}/')

        self.source_clients = [SourceClient(home_path, setup, dist, cid, self.seed) for cid in range(self.no_clients)]
        self.target_client = TargetClient(home_path, setup, dist, self.seed, k_value)

        self.source_client_sample_sizes = [client.get_no_samples() for client in self.source_clients]

        self.total_source_samples = sum(self.source_client_sample_sizes)

        self.kernel_sig = None
        self.noise_sig = None

        self.confidences = None
        self.target_groups = self.target_client.get_groups()

        self.best_lambdas = None

    def compute_global_hyperparameters(self):
        """
        When called, performs the federated hyperparameter optimization among the source clients for each feature.
        :return: The global hyperparameter vectors computed.
        """

        if self.kernel_sig and self.noise_sig:
            print("Hyperparameters are already computed")
            return
        elif self.confidences:
            print("Confidences are already computed, no need to compute hyperparameters")
            return

        kernel_sig = []
        noise_sig = []

        no_features = self.target_client.get_no_features()

        for i in range(no_features):
            ks, ns = zip(*(client.compute_hyperparameters(i) for client in self.source_clients))
            kernel_sig.append(sum(ks) / self.no_clients)
            noise_sig.append(sum(ns) / self.no_clients)
            print("computed hyperparameter for feature ", i)

        self.kernel_sig = kernel_sig
        self.noise_sig = noise_sig

        return kernel_sig, noise_sig

    def compute_federated_confidence_scores(self):
        """
        When called, performs the federated GPR training as well as the confidence score computation. The federated GPR
        training involves both the source clients and the target client. Randomized encoding and masking is used during
        the computation to protect the privacy of local data of the clients. The computed closed form solution of the
        GPR model is directly the predicted distribution. this distribution is then sent to the target client where
        the target client computes the confidence score.
        :return: None
        """
        if self.confidences is not None:
            print("Confidences already computed.")
            self.target_client.set_confidences(self.confidences)
            return

        if self.noise_sig is None or self.kernel_sig is None:
            raise ValueError(
                "Hyperparameters (noise_sig and kernel_sig) are not set. Cannot compute federated confidence scores.")

        preds_mean, preds_var = self.predict_with_federated_GPRs(self.kernel_sig, self.noise_sig)

        confidences = []

        no_features = self.target_client.get_no_features()

        for i in range(no_features):
            confs = self.target_client.compute_confidence_score(i, preds_mean[i], preds_var[i])
            confidences.append(confs)
            print(f"Confidence computed for feature: {i}")

        confidences = np.column_stack(confidences)
        np.savetxt(os.path.join(self.dist_dir, "all_confidences.csv"), confidences, delimiter=';')

        self.confidences = confidences
        self.target_client.set_confidences(self.confidences)

    def predict_with_federated_GPRs(self, global_kernel_sig, global_noise_sig):
        """
        Performs federated GPR training with the computed hyperparameters.
        :param global_kernel_sig: The global sigma of the linear kernel computed during federated hyperparameter
        optimization
        :param global_noise_sig: The global sigma of the noise of the GPR formula computed during federated
        hyperparameter optimization
        :return: The mean and variance of the predicted distribution
        """

        prediction_means = []
        prediction_vars = []

        no_features = self.target_client.get_no_features()

        for i in range(no_features):
            masked_source_data = [client.generate_masked_data(i) for client in self.source_clients]
            masked_target_data = self.target_client.generate_masked_data(i)

            rows = [np.concatenate([masked_data @ Y.T for Y in masked_source_data], axis=1) for masked_data in
                    masked_source_data]

            # Concatenate all rows vertically to form the Gram matrix
            gram_matrix = np.concatenate(rows, axis=0)

            K = global_kernel_sig[i] * gram_matrix

            K_star = global_kernel_sig[i] * np.concatenate([masked_target_data @ X.T for X in masked_source_data],
                                                           axis=1)

            K_star_star = global_kernel_sig[i] * (masked_target_data @ masked_target_data.T)

            # Computing the intermediate matrix products

            inverse_of_covariance_matrix_of_input = np.linalg.inv(
                K + global_noise_sig[i] * np.eye(self.total_source_samples))

            intermediate_mean_vector = np.dot(K_star, inverse_of_covariance_matrix_of_input)  # This is K* @ K^-1

            # now splitting the intermediate result between the source clients
            client_means = []
            start_idx = 0

            for idx in range(self.no_clients):
                # Calculate the end index for the current client based on the number of samples they have
                end_idx = start_idx + self.source_client_sample_sizes[idx]

                # Slice the mean matrix from start_idx to end_idx for this client
                client_means.append(intermediate_mean_vector[:, start_idx:end_idx])

                # Update the start index for the next client
                start_idx = end_idx

            # Each client computes the dot product between their slice and their label vector (current feature vector)

            final_mean_parts = [client.compute_mean_piece(mean_piece, i) for mean_piece, client in
                                zip(client_means, self.source_clients)]

            final_mean = sum(final_mean_parts).flatten()

            # computing the variance is straightforward
            cov = np.diag(K_star_star) - np.dot(K_star, np.dot(inverse_of_covariance_matrix_of_input, K_star.T))
            var = np.sqrt(np.diag(cov))

            # computed mean and var are then sent to the target for confidence score computation

            prediction_means.append(final_mean)
            prediction_vars.append(var)

        return prediction_means, prediction_vars

    def compute_best_lambdas(self, lambda_path, alpha, epochs, global_iterations, lr_init,
                             lr_final):
        """
        Performs the best lambda prediction phase. For a given lambda path, all source clients train separate WEN models
        in a federated learning setting for each tissue. All these models are then sent to the target client where the
        target client computes the best lambdas.
        :param lambda_path: A list of lambda values to perform training with
        :param alpha: Weighting factor for the loss function.
        :param epochs: Number of local training epochs.
        :param global_iterations: Number of global iterations.
        :param lr_init: Initial learning rate.
        :param lr_final: Final learning rate.
        :return: None
        """

        all_results = dict()

        tissue_weights = self.target_client.compute_weights()

        for lam in lambda_path:
            print("Training for Lamda = %s" % lam)
            global_models = dict()

            for tissue in tissue_weights:
                global_model = self.train_federated_WEN(lam, tissue_weights[tissue], alpha, epochs, global_iterations,
                                                        lr_init,
                                                        lr_final)
                global_models[tissue] = global_model

            results = self.target_client.compute_MAE(global_models)
            all_results[lam] = results

        df = pd.DataFrame.from_dict(all_results, orient='index')
        df = df.transpose()
        df.to_csv(os.path.join(self.dist_dir, f'all_results.csv'))

        best_lambdas = dict()

        for group in self.target_groups:
            min_error = float('inf')
            best_lambda = None

            for lam, results in all_results.items():
                if results[group] < min_error:
                    min_error = results[group]
                    best_lambda = lam

            # Store the best lambda for the current group
            best_lambdas[group] = best_lambda

        print("Best lambdas for each tissue:")
        for group, lam in best_lambdas.items():
            print(f"Group: {group}, Best Lambda: {lam}")

        df = pd.DataFrame(list(best_lambdas.items()), columns=['Group', 'Best Lambda'])

        # Save the DataFrame to a CSV file
        df.to_csv(os.path.join(self.dist_dir, 'best_lambdas.csv'), index=False)

        self.best_lambdas = best_lambdas

    def use_precomputed_best_lambdas(self):
        """
        Reads precomputed lambda values from the working directory if they exist. The file should be named
        best_lambdas.csv and should be inside the current distribution directory
        :return: None
        """
        file_path = os.path.join(self.dist_dir, "best_lambdas.csv")

        if os.path.exists(file_path):  # Check if precomputed lambdas exist
            # Read the CSV file into a DataFrame
            best_lambdas_df = pd.read_csv(file_path)

            # Convert the DataFrame to a dictionary (if needed)
            self.best_lambdas = best_lambdas_df.set_index('Group')['Best Lambda'].to_dict()
        else:
            # Raise an error if the file does not exist
            raise FileNotFoundError(f"Could not find 'best_lambdas.csv' in the directory: {self.dist_dir}")

    def use_precomputed_confidences(self):
        """
        Reads precomputed confidence scores from the working directory if they exist. The file should be named
        all_confidences.csv and should be inside the current distribution directory
        :return: None
        """
        file_path = os.path.join(self.dist_dir, "all_confidences.csv")

        if os.path.exists(file_path):  # Check if precomputed confidences exist
            # Load precomputed confidences
            confidences = np.loadtxt(file_path, delimiter=';')
            self.confidences = np.asfortranarray(confidences)
            self.target_client.set_confidences(self.confidences)
        else:
            # Raise an error if the file does not exist
            raise FileNotFoundError(f"Could not find 'all_confidences.csv' in the directory: {self.dist_dir}")

    def train_federated_WEN(self, lam, feature_weights, alpha=0.8, epochs=100, global_iterations=10,
                            lr_init=1e-4,
                            lr_final=1e-5):
        """
        Performs federated Weighted Elastic Net (WEN) training with the source clients for a given set of parameters.
        :param lam: The regularization parameter to use
        :param feature_weights: The feature weights computed using the confidence scores
        :param alpha: Weighting factor for the loss function
        :param epochs: Number of local training epochs
        :param global_iterations: Number of global iterations
        :param lr_init: Initial learning rate
        :param lr_final: Final learning rate
        :return: The global model after the federated training is completed
        """

        if self.confidences is None:
            print("Cannot perform training without confidences")
            return

        no_features = self.target_client.get_no_features()

        global_model = create_model(no_features, alpha, lam, feature_weights, lr_init)

        for iteration in range(global_iterations):
            print(".", end="")

            current_lr = lr_schedule(iteration, global_iterations, lr_init, lr_final)

            global_model_weights = global_model.get_weights()

            local_updates = []

            for client in self.source_clients:
                local_update = client.train_WEN_locally(global_model_weights, feature_weights, alpha, lam, current_lr,
                                                        epochs)
                local_updates.append(local_update)

            # Average the weights
            average_weights = [np.mean(weight_list, axis=0) for weight_list in zip(*local_updates)]

            global_model.set_weights(average_weights)

        return global_model

    def train_final_adaptive_models(self, alpha, epochs, global_iterations, lr_init,
                                    lr_final):
        """
        Trains and prints the performance of the final adaptive models by using the optimal lambdas predicted
        by the target client
        :param alpha: Weighting factor for the loss function
        :param epochs: Number of local training epochs
        :param global_iterations: Number of global iterations
        :param lr_init: Initial learning rate
        :param lr_final: Final learning rate
        :return: None
        """

        self.target_client.read_tissue_similarities()
        tissue_combinations = self.target_client.get_tissue_combinations()

        tissue_weights = self.target_client.compute_weights()

        # set up tables for hold-out errors
        error_table = pd.DataFrame(columns=["combination", "tissue", "mean", "median", "corr", "std", "iqr"],
                                   dtype=np.float64)
        error_table["combination"] = error_table["combination"].astype(np.int64)
        error_table.set_index(["combination", "tissue"], inplace=True)

        for i in range(len(tissue_combinations)):
            print("  - combination", i, end="\n")

            fit_tissues = tissue_combinations[i]
            holdout_tissues = [t for t in self.target_groups if t not in fit_tissues]

            predicted_lambdas = self.target_client.predict_optimal_lambdas(self.best_lambdas,
                                                                           fit_tissues,
                                                                           holdout_tissues)
            print("predicted lambdas:")
            print(predicted_lambdas, "\n")

            for tissue in holdout_tissues:
                lam = predicted_lambdas[tissue]

                global_model = self.train_federated_WEN(lam, tissue_weights[tissue], alpha, epochs, global_iterations,
                                                        lr_init,
                                                        lr_final)

                errors_t = self.target_client.eval_model(tissue, global_model)

                error_table.loc[(i, tissue), :] = errors_t
                print()
