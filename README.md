# Harmful Brain Activity Classification – MLiP Project – by Noodle Nappers

Welcome to the Noodle Nappers project repository, dedicated to the 2024 Machine Learning in Practice (MLiP) course at Radboud University. This repository focuses solely on the [Kaggle competition](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) aimed at detecting and classifying seizures and other types of harmful brain activity using electroencephalography (EEG) signals. Our work here aims to contribute to the advancement of neurocritical care, epilepsy treatment, and drug development through improved EEG pattern classification accuracy.

**Authors: Luppo Sloup, Dick Blankvoort, Tygo Francissen (MLiP Group 9)**

## Overview

The goal of this competition is to develop a model trained on EEG signals recorded from critically ill hospital patients. By accurately detecting and classifying seizures and other harmful brain activity, this project can aid doctors and brain researchers in providing faster and more accurate treatments, potentially unlocking transformative benefits for neurocritical care, epilepsy management, and drug development.

## Our Work

Our work includes multiple notebooks, files, and data sets, which are explained in more detail below. This also includes an explanation of this Github repository's structure.

### Notebooks

In this Github repository, four notebooks are present that were the main notebooks in this project. They are self-explanatory and will be shortly mentioned in the project structure section below.

However, in our project, many more notebooks were created. We list the most important ones from our accounts on Kaggle below.

We utilized these notebooks to modify the EfficientNet B0 starter for training and inference:

- [Train Notebook](https://www.kaggle.com/code/tygofrancissen/train-notebook)
- [Inference Notebook](https://www.kaggle.com/code/tygofrancissen/inference-notebook)

Furthermore, we modified the HMS ensemble of EfficientNet and ResNet to fit our ensemble model with these notebooks:

- [HMS Multiple Model Ensemble Original](https://www.kaggle.com/code/luepoe/hms-multiple-model-ensemble-4-notebooks-19b844)
- [IIACT Training HMS Workflow](https://www.kaggle.com/code/luepoe/iiact-training-hms-workflow)
- [Master Training HMS Workflow](https://www.kaggle.com/code/luepoe/master-training-hms-workflow)

As we added more models to the ensemble, we needed to modify and search through more notebooks:

- [IIACT Ensamble Features Head Starter](https://www.kaggle.com/code/luepoe/iiact-ensamble-features-head-starter)
- [Wavenet Starter](https://www.kaggle.com/code/luepoe/wavenet-starter-lb-0-66)
- [WaveNet Training](https://www.kaggle.com/code/luepoe/lb-0-46-dilatedinception-wavenet-training)
- [WaveNet Inference](https://www.kaggle.com/code/luepoe/lb-0-46-dilatedinception-wavenet-inference)
- [Catboost Starter](https://www.kaggle.com/code/luepoe/catboost-starter-lb-0-67)

Finally, this notebook we created combines all models to create an ensemble and the submission:

- [IIACT Ensemble 7 Models](https://www.kaggle.com/code/luepoe/iiact-ensamble-7-models)

Note that this notebook has not been made public in order to keep our best-working code private, but it is available in this repository as `ensemble-7-models.ipynb`.

### Datasets

To be able to store augmented data, packages, and models, we created several data sets on Kaggle:

- [Brain Solver](https://www.kaggle.com/datasets/luepoe/brain-solver): This data set contains the whole Python package with all the code being pushed into this using GitHub workflows. There are almost 125 versions available of this package.
- [D2L Package](https://www.kaggle.com/datasets/tygofrancissen/d2l-package): This data set contains the necessary files to properly import the d2l package, as it is not available in Kaggle using offline mode.
- [Trained Model EfficientNet](https://www.kaggle.com/datasets/tygofrancissen/trained-model-effnet-mlip9): This data set contains all versions of trained models for our modified EfficientNet notebook.
- [Wav2vec/Filter EEGs and Spectograms](https://www.kaggle.com/datasets/dickblankvoort/w2v-specs): This data set contains the raw EEG data and spectograms after being processed with wav2vec, filters, or a combination.
- [Models Wav2Vec/Filter Training](https://www.kaggle.com/datasets/dickblankvoort/models-first-wav2vec-training): This data set contains a wide range of models that were stored after being trained with the `Wav2vec/Filter EEGs and Spectograms` data set mentioned above.
- [Catboost Model](https://www.kaggle.com/datasets/luepoe/catboost-model): This data set stores the trained models for CatBoost.
- [Dilated WaveNet](https://www.kaggle.com/datasets/luepoe/dilated-wavenet): This data set stores the trained models for WaveNet.

### Project Structure

This is the core structure of our Github repository:

```txt
NoodleBrainActivityClassification/
├── brain_solver/
│   ├── brain_model.py
│   ├── config.py
│   ├── eeg_dataset.py
│   ├── filter.py
│   ├── helpers.py
│   ├── network.py
│   ├── trainer.py
│   └── wav2vec2.py
├── data/
├── images/
├── dataset-metadata.json
├── ensemble-7-models.ipynb
├── inference.ipynb
├── preprocessing.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.py
└── training.ipynb
```

Below is a small explanation for the usage of the most important files:

- **brain_solver/brain_model.py**: Model definition for the trained EfficientNet, including training functions and other model-specific operations.
- **brain_solver/trainer.py**: Trainer class that encapsulates the training logic, used to manage the training process of models.
- **brain_solver/eeg_dataset.py**: DataLoader compatibility layer, providing a dataset class that enables efficient data handling and preprocessing for PyTorch models.
- **brain_solver/helpers.py**: A collection of miscellaneous functions that serve as a utility library for various tasks throughout the project.
- **brain_solver/filter.py**: Implementation of preprocessing filters (low-pass, high-pass, band-pass, and band-stop) used for data preprocessing before feeding it into the model.
- **brain_solver/wav2vec2.py**: Class designed for data preprocessing, leveraging the Wav2Vec 2.0 model to process and transform data before model training or inference.
- **setup.py**: Standard setup script to manage project dependencies and environment setup.
- **inference.ipynb**: Notebook for conducting inference using our trained EfficientNet model, part of the ensemble approach in the competition.
- **training.ipynb**: Notebook dedicated to the training process of our EfficientNet model, which is later utilized within the ensemble for the competition.

TODO:



explaining the files ...........................

FIX FIGURES







### Installation

In order to reproduce our results and run our code, you should use _Python 3.10_ and install the requirements as specified with `pip`:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commit and PR Rules

We tried to stick to these rules during our project:

1. **"Protected" master branch**: Direct commits to the master branch are not recommended. The branch is not locked, but do not do it.
2. **Branching**: Always create a new branch for your changes. Name your branch based on the feature or fix you're working on, preferably linking to an issue number. E.g., `feature-12-add-new-filter` or `fix-15-resolve-this-really-annoying-bug`.
3. **Commit Messages**:
   - Commits should be categorized using prefixes like `feat:`, `fix:`, `chore:`, `docs:`, `style:`, `refactor:`, `perf:`, and `test:`.
   - Use meaningful commit messages that clearly describe the change.
4. **Pull Requests (PRs)**:
   - PRs should have descriptive titles and should explain the purpose and content of the changes.
   - Each PR must have at least one review before merging.
   - After reviews and any necessary adjustments, the PR can be merged into the master branch.
