from transformers import Wav2Vec2Processor


class Wav2Vec2:
    @staticmethod
    def wav2vec2(spectrograms):
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        processed = {}
        for file_name, specs in spectrograms.items():
            features = processor(
                specs, sampling_rate=16000, return_tensors="pt", padding=True
            )
            processed[file_name] = features.input_values
        return processed
