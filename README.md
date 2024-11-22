# FREDA

This repository contains the source code of our research article

**Privacy Preserving Federated Unsupervised Domain Adaptation with Application to Age Prediction from DNA Methylation
Data**

### Software

- Python 3.8.18

### Libraries

To install the required Python libraries, run:

```bash
pip install -r requirements.txt
```

---

### Usage

The following arguments can be configured when running the `main.py` script:

| Argument                    | Description                                                                             | Default Value       |
|-----------------------------|-----------------------------------------------------------------------------------------|---------------------|
| `--setup`                   | Number of source clients to simulate.                                                   | `2`                 |
| `--dist`                    | Distribution identifier for the experiment.                                             | `0`                 |
| `--use_precomputed_confs`   | Whether to use precomputed confidence scores.                                           | `True`              |
| `--use_precomputed_lambdas` | Whether to use precomputed optimal lambdas.                                             | `True`              |
| `--lambda_path`             | Path to a text file containing lambda values. If not provided, default values are used. | `None`              |
| `--home_path`               | Root directory for the project. Can be set to any desired path.                         | `Current directory` |
| `--alpha`                   | Weighting factor for the loss function.                                                 | `0.8`               |
| `--epochs`                  | Number of local training epochs.                                                        | `20`                |
| `--global_iterations`       | Number of global iterations.                                                            | `100`               |
| `--lr_init`                 | Initial learning rate.                                                                  | `0.0001`            |
| `--lr_final`                | Final learning rate.                                                                    | `0.00001`           |
| `--k_value`                 | Exponent of the weight function for transforming confidences into weights.              | `3`                 |

### Example Command

Here’s an example of how to run the experiment with sample arguments:

```bash
python main.py --setup 2 --dist 0 --use_precomputed_confs False --use_precomputed_lambdas False --lambda_path ./lambdas.txt --home_path ./FREDA/ --alpha 0.8 --epochs 20 --global_iterations 100 --lr_init 0.0001 --lr_final 0.00001 --k_value 3
```

---

## Data

We utilized DNA methylation data and donor age information from two main sources:

1. **The Cancer Genome Atlas (TCGA)**
   - **Reference**:  
     Weinstein, J.N., Collisson, E.A., Mills, G.B., Shaw, K.R., Ozenberger, B.A., Ellrott, K., Shmulevich, I., Sander, C., Stuart, J.M.: The cancer genome atlas pan-cancer analysis project. *Nature Genetics*, 45(10), 1113–1120 (2013).  
     DOI: [10.1038/ng.2764](https://doi.org/10.1038/ng.2764)

2. **The Gene Expression Omnibus (GEO)** 
   - **Reference**:  
     Edgar, R., Domrachev, M., Lash, A.E.: Gene expression omnibus: NCBI gene expression and hybridization array data repository. *Nucleic Acids Research*, 30(1), 207–210 (2002).  
     DOI: [10.1093/nar/30.1.207](https://doi.org/10.1093/nar/30.1.207)

---

## Tissue Similarities

We used tissue similarities data translated from the following paper:  
   - **Reference**:  
     Aguet, F. et al. Genetic effects on gene expression across human tissues. NATURE 550, 204–213 (2017).  
     DOI: [10.1038/nature24277](https://doi.org/10.1038/nature24277)







