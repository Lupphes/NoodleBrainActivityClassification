class Config:
    def __init__(
        self,
        full_path,
        output_path,
        VER=1,
        num_classes=6,
        batch_size=88,
        epochs=20,
        seed=2024,
        weight_decay=1e-2,
        learning_rate=8e-3,
        use_wavelet=None,
        USE_KAGGLE_SPECTROGRAMS=True,
        USE_EEG_SPECTROGRAMS=True,
        should_read_brain_spectograms=True,
        should_read_eeg_spectrogram_files=True,
        USE_PRETRAINED_MODEL=True,
    ):
        self.VER = VER
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.use_wavelet = use_wavelet
        self.USE_KAGGLE_SPECTROGRAMS = USE_KAGGLE_SPECTROGRAMS
        self.USE_EEG_SPECTROGRAMS = USE_EEG_SPECTROGRAMS
        self.should_read_brain_spectograms = should_read_brain_spectograms
        self.should_read_eeg_spectrogram_files = should_read_eeg_spectrogram_files

        # Paths setup with default parameters
        self.full_path = full_path
        self.output_path = output_path
        self.competition_data_path = (
            full_path + "hms-harmful-brain-activity-classification/"
        )
        self.data_train_csv = self.competition_data_path + "train.csv"
        self.data_test_csv = self.competition_data_path + "test.csv"
        self.data_eeg = self.competition_data_path + "train_eegs/"
        self.data_spectograms = self.competition_data_path + "train_spectrograms/"
        self.data_spectograms_test = self.competition_data_path + "test_spectrograms/"
        self.data_eeg_test = self.competition_data_path + "test_eegs/"
        self.trained_model_path = None
        self.trained_weight_file = None
        if USE_PRETRAINED_MODEL:
            self.trained_model_path = full_path + "hms-efficientnetb0-pt-ckpts/"
            self.trained_weight_file = (
                self.trained_model_path + "efficientnet_b0_rwightman-7f5810bc.pth"
            )
        self.path_to_brain_spectrograms_npy = full_path + "brain-spectrograms/specs.npy"
        self.path_to_brain_spectrograms_parquet = (
            full_path + "brain-spectrograms/train.pqt"
        )
        self.base_path_to_brain_eeg_spectrograms = full_path + "brain-eeg-spectrograms/"
        self.path_to_eeg_spectrograms_folder = (
            self.base_path_to_brain_eeg_spectrograms + "EEG_Spectrograms/"
        )
        self.path_to_eeg_spectrograms_npy = (
            self.base_path_to_brain_eeg_spectrograms + "eeg_specs.npy"
        )
