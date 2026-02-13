# Grover Acceleration for Attention Computation

This repository focuses on how Grover search can be applied to accelerate the attention mechanism in transformer models. Specifically, we explore the use of Grover search to speed up the computation of attention weights in visual transformer models on MNIST dataset. 

# Project Structure

- `entrance.py`: The main script to run the experiments.
- `qvit_test\`: The package for qvit test.
- - `evaluation.py`: Contains the evaluation functions for ViT and QVIT models.
- - `feature_extraction.py`: Contains the feature extraction functionalities for transformers, i.e., tokenization and patchification.
- - `qiskit_grover.py`: Grover search implementation using Qiskit. Only internally used in `qvit.py` to implement Grover search acceleration.
- - `qvit.py`: The implementation of QVIT model with Grover search acceleration.
- - `vit.py`: The implementation of traditinal ViT models.
- `data\`: The directory to store the dataset.
- `pyproject.toml`: The configuration file for the project. 
- `uv.lock`: The lock file for the project. Please ensure to install the dependencies specified in this file. **uv** is strongly suggested to use.

# TODOs
- Implement percentile record during training. Use the average of percentiles during training as the percentile when during prediction. (Solution for looking for a appropriate percentile threshold)

- Implement the local connection mask in parallel with 

# References

[Grover Search for Acceleration of Attention Computation](https://arxiv.org/abs/arXiv:2307.08045) (arXiv:2307.08045)

[Sublinear Time Quantum Algorithm for Attention Approximation](https://arXiv.org/abs/2602.00874) (arXiv:2602.00874).
