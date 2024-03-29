# Configuration file for the project
class Config:
    def __init__(
        self,
        full_path,
        output_path,
        VER=5,
        num_classes=6,
        batch_size=88,
        epochs=20,
        seed=2024,
        num_folds=5,
        weight_decay=1e-2,
        learning_rate=8e-3,
        use_wavelet=None,
        USE_KAGGLE_SPECTROGRAMS=True,
        USE_EEG_SPECTROGRAMS=True,
        should_read_brain_spectograms=True,
        should_read_eeg_spectrogram_files=True,
        USE_PRETRAINED_MODEL=True,
        FINE_TUNE=True,
    ):
        # Parameters
        self.VER = VER
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.num_folds = num_folds
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.use_wavelet = use_wavelet
        self.USE_KAGGLE_SPECTROGRAMS = USE_KAGGLE_SPECTROGRAMS
        self.USE_EEG_SPECTROGRAMS = USE_EEG_SPECTROGRAMS
        self.should_read_brain_spectograms = should_read_brain_spectograms
        self.should_read_eeg_spectrogram_files = should_read_eeg_spectrogram_files
        self.FINE_TUNE = FINE_TUNE

        # Paths to data sources and models
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
        self.data_w2v_eegs = self.output_path + "w2v_eegs_filter/"
        self.data_w2v_specs = self.output_path + "w2v_specs_filter/"
        self.data_w2v_specs_eeg = self.output_path + "w2v_specs_filter_eeg/"
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
        self.efficientnetb_tf_keras = (
            full_path + "efficientnetb-tf-keras/EfficientNetB2.h5"
        )
        self.futures_head_starters_models = full_path + "features-head-starter-models/"
        self.brain_eegs_npy = full_path + "brain-eegs/eegs.npy"
        self.brain_eegs = full_path + "brain-eegs/"
        self.catboost_path = full_path + "catboost-starter-lb-0-60/"
        self.brain_efficientnet_models = (
            full_path + "brain-efficientnet-models-v3-v4-v5/"
        )
        self.tf_efficientnet_imagenet = (
            full_path
            + "tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
        )
        self.resnet34d = full_path + "hms-baseline-resnet34d-512-512-training-5-folds/"
        self.train_resnet34d = full_path + "resnet34d/hms-train-resnet34d/"
        self.efficientnetb0 = full_path + "efficientnetb0/hms-train-efficientnetb0/"
        self.efficientnetb1 = full_path + "efficientnetb1/hms-train-efficientnetb1/"
