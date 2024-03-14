# Harmful Brain Activity Classification – MLiP Project – by Noodle Nappers

Welcome to the Noodle Nappers project repository, dedicated to the Machine Learning in Practice (MLiP) course. This repository focuses solely on the [Kaggle competition](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) aimed at detecting and classifying seizures and other types of harmful brain activity using electroencephalography (EEG) signals. Our work here aims to contribute to the advancement of neurocritical care, epilepsy treatment, and drug development through improved EEG pattern classification accuracy.

## Overview

The goal of this competition is to develop a model trained on EEG signals recorded from critically ill hospital patients. By accurately detecting and classifying seizures and other harmful brain activity, this project can aid doctors and brain researchers in providing faster and more accurate treatments, potentially unlocking transformative benefits for neurocritical care, epilepsy management, and drug development.

### Our Notebooks:

1. [IIACT Ensamble 7 Models](https://www.kaggle.com/code/luepoe/iiact-ensamble-7-models)
2. [HMS Multiple Model Ensemble Original](https://www.kaggle.com/code/luepoe/hms-multiple-model-ensemble-4-notebooks-19b844) _(Updated link)_
3. [IIACT Ensamble Features Head Starter](https://www.kaggle.com/code/luepoe/iiact-ensamble-features-head-starter)
4. [Wavenet Starter LB 0.66](https://www.kaggle.com/code/luepoe/wavenet-starter-lb-0-66)
5. [Train Notebook by TygoFrancissen](https://www.kaggle.com/code/tygofrancissen/train-notebook)
6. [Master Training HMS Workflow](https://www.kaggle.com/code/luepoe/master-training-hms-workflow)
7. [IIACT Training HMS Workflow](https://www.kaggle.com/code/luepoe/iiact-training-hms-workflow)
8. [Inference Notebook by TygoFrancissen](https://www.kaggle.com/code/tygofrancissen/inference-notebook)
9. [LB 0.46 DilatedInception WaveNet Inference](https://www.kaggle.com/code/luepoe/lb-0-46-dilatedinception-wavenet-inference)
10. [LB 0.46 DilatedInception WaveNet Training](https://www.kaggle.com/code/luepoe/lb-0-46-dilatedinception-wavenet-training)
11. [Catboost Starter LB 0.67](https://www.kaggle.com/code/luepoe/catboost-starter-lb-0-67)

### Our Datasets:

1. [W2V Specs](https://www.kaggle.com/datasets/dickblankvoort/w2v-specs)
2. [Models First Wav2Vec Training](https://www.kaggle.com/datasets/dickblankvoort/models-first-wav2vec-training)
3. [Trained Model Effnet MLiP9](https://www.kaggle.com/datasets/tygofrancissen/trained-model-effnet-mlip9)
4. [Catboost Model](https://www.kaggle.com/datasets/luepoe/catboost-model)
5. [Brain Solver](https://www.kaggle.com/datasets/luepoe/brain-solver)
6. [D2L Package](https://www.kaggle.com/datasets/tygofrancissen/d2l-package)
7. [Dilated WaveNet](https://www.kaggle.com/datasets/luepoe/dilated-wavenet)

## Project Structure

```
NoodleNappers/
├── brain_solver/
│   ├── brain_model.py
│   ├── config.py
│   ├── eeg_dataset.py
│   ├── filter.py
│   ├── helpers.py
│   ├── trainer.py
│   └── wav2vec2.py
├── data/
├── dataset-metadata.json
├── inference.ipynb
├── notebooks/
├── preprocessing.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.py
└── training.ipynb
```

## Version

We are using _Python 3.10_ and the requirement installed as specified always with `pip`:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commit and PR Rules

1. **"Protected" master branch**: Direct commits to the master branch are not recommended. The branch is not locked, but do not do it.
2. **Branching**: Always create a new branch for your changes. Name your branch based on the feature or fix you're working on, preferably linking to an issue number. E.g., `feature-12-add-new-filter` or `fix-15-resolve-this-really-stupid-bug`.
3. **Commit Messages**:
   - Commits should be categorized using prefixes like `feat:`, `fix:`, `chore:`, `docs:`, `style:`, `refactor:`, `perf:`, and `test:`.
   - Use meaningful commit messages that clearly describe the change.
   - Incorporate the GitHub issue number at the beginning, e.g., `feat(#12): Added new functionality`. (optional)
4. **Pull Requests (PRs)**:
   - Each PR must be associated with a GitHub issue.
   - PRs should have descriptive titles and should explain the purpose and content of the changes.
   - Each PR must have at least one review before merging.
   - After reviews and any necessary adjustments, the PR can be merged into the master branch.

## Template for Commit Messages

```
[prefix(#GitHub Issue Number)]: Short description of the change
[prefix]: Short description of the change

For example:
feat(#12): Add new filter functionality
feat: Add new filter functionality

(Optional) A more detailed description can follow if required. It should provide context for the change, detail on the solution, or any other pertinent information.
```

This README should give a clear overview of the project and lay down some basic rules for collaboration.

## HMS Competition Structure

- **brain_solver/brain_model.py**: Model definition for the trained EfficientNet, including training functions and other model-specific operations.
- **brain_solver/trainer.py**: Trainer class that encapsulates the training logic, used to manage the training process of models.
- **brain_solver/eeg_dataset.py**: DataLoader compatibility layer, providing a dataset class that enables efficient data handling and preprocessing for PyTorch models.
- **brain_solver/helpers.py**: A collection of miscellaneous functions that serve as a utility library for various tasks throughout the project.
- **brain_solver/filter.py**: Implementation of preprocessing filters (low-pass, high-pass, band-pass, and band-stop) used for data preprocessing before feeding it into the model.
- **brain_solver/wav2vec2.py**: Class designed for data preprocessing, leveraging the Wav2Vec 2.0 model to process and transform data before model training or inference.
- **setup.py**: Standard setup script to manage project dependencies and environment setup.

### Notebooks:

- **inference.ipynb**: Notebook for conducting inference using our trained EfficientNet model, part of the ensemble approach in the competition.
- **training.ipynb**: Notebook dedicated to the training process of our EfficientNet model, which is later utilized within the ensemble for the competition.
