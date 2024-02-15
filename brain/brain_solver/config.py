class Config:
    VER = 5
    full_path = "./data/"
    competition_data_path = full_path

    output_path = full_path + "out/"

    data_train_csv = competition_data_path + "train.csv"
    data_test_csv = competition_data_path + "test.csv"
    data_eeg = competition_data_path + "train_eegs/"
    data_spectograms = competition_data_path + "train_spectrograms/"

    data_spectograms_test = competition_data_path + "test_spectrograms/"
    data_eeg_test = competition_data_path + "test_eegs/"

    trained_model_path = full_path + "hms-efficientnetb0-pt-ckpts/"
    trained_weight_file = trained_model_path + "efficientnet_b0_rwightman-7f5810bc.pth"

    num_classes = 6
    batch_size = 88
    epochs = 20
    seed = 2024
    weight_decay = 1e-2
    learning_rate = 8e-3

    use_wavelet = None

    USE_KAGGLE_SPECTROGRAMS = True
    USE_EEG_SPECTROGRAMS = True

    # 'Brain-Spectrograms' dataset on Kaggle.
    should_read_brain_spectograms = True
    path_to_brain_spectrograms_npy = full_path + "brain-spectrograms/specs.npy"
    path_to_brain_spectrograms_parquet = full_path + "brain-spectrograms/train.pqt"

    # Brain-EEG-Spectrograms' dataset folder.
    should_read_eeg_spectrogram_files = True
    base_path_to_brain_eeg_spectrograms = full_path + "brain-eeg-spectrograms/"
    path_to_eeg_spectrograms_folder = (
        base_path_to_brain_eeg_spectrograms + "EEG_Spectrograms/"
    )
    path_to_eeg_spectrograms_npy = base_path_to_brain_eeg_spectrograms + "eeg_specs.npy"
