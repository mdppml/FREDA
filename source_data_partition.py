import os
import numpy as np
import data as data_mod
import argparse
import util

parser = argparse.ArgumentParser()

parser.add_argument('--home_path', type=str, default=os.getcwd(),
                    help="Root directory for the project (default: current working directory).")

parser.add_argument('--setup', type=int, default=2,
                    help="Number of source clients to split data")

parser.add_argument('--dist', type=int, default=17,
                    help="Distribution ID")


args = parser.parse_args()

# set paths and parameters
home_path = args.home_path
no_clients = args.setup
dist = args.dist

partition_path = os.path.join(home_path, f'data/{no_clients}_client/dist_{dist}')


def main():
    data = data_mod.DataDNAmethPreprocessed(home_path)

    normalizer_x = util.StandardNormalizer
    normalizer_y = util.HorvathNormalizer

    norm_x = normalizer_x(data.training.meth_matrix)
    norm_y = normalizer_y(data.training.age)

    x_source = norm_x.normalize(data.training.meth_matrix)
    y_source = norm_y.normalize(data.training.age)

    # Uniform random partitioning
    total_samples = len(x_source)
    samples_per_party = total_samples // no_clients

    # Create indices for random shuffling
    indices = np.random.permutation(total_samples)

    for party in range(no_clients):
        party_dir = os.path.join(partition_path, f'{party}/')

        if not os.path.exists(party_dir):
            os.makedirs(party_dir)

        start_index = party * samples_per_party
        end_index = (party + 1) * samples_per_party if party < no_clients - 1 else total_samples

        # Get partitioned data
        x_partitioned = x_source[indices[start_index:end_index]]
        y_partitioned = y_source[indices[start_index:end_index]]

        # Save partitioned data to txt files
        np.savetxt(os.path.join(party_dir, 'x_train.txt'), x_partitioned)
        np.savetxt(os.path.join(party_dir, 'y_train.txt'), y_partitioned)


if __name__ == "__main__":
    main()
