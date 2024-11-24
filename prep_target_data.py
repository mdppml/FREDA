import os
import numpy as np
import argparse
import util
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--home_path', type=str, default=os.getcwd(),
                    help="Root directory for the project (default: current working directory).")

args = parser.parse_args()

# set paths and parameters
home_path = args.home_path

partition_path = os.path.join(home_path, f'data/target_client/')


def main():
    # Load source and target data
    data_dir = os.path.join(args.home_path, "dna_data")

    source_file = os.path.join(data_dir, "source_data.txt")
    source_table = pd.read_csv(source_file, sep="\t", header=None)
    source_matrix = np.asfortranarray(source_table.values)

    normalizer_x = util.StandardNormalizer
    norm_x = normalizer_x(source_matrix)

    # Load target data
    data_dir = os.path.join(args.home_path, "dna_data")

    source_file = os.path.join(data_dir, "target_data.txt")
    target_table = pd.read_csv(source_file, sep="\t", header=None)
    target_matrix = np.asfortranarray(target_table.values)

    # Load source data labels
    y_file = os.path.join(data_dir, "target_y.tsv")
    y_table = pd.read_csv(y_file, header=None)

    y_target = np.asfortranarray(y_table).ravel()
    x_target = norm_x.normalize(target_matrix)

    # Save partitioned data to txt files
    np.savetxt(os.path.join(partition_path, 'x_test.txt'), x_target)
    np.savetxt(os.path.join(partition_path, 'y_test.txt'), y_target)


if __name__ == "__main__":
    main()
