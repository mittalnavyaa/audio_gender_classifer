import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Update constants
WAV_TRAIN_PATH = r"C:\Users\navya\OneDrive\Desktop\ICTC\Cocktail Party\Model\gender-audio-classification\VoxCeleb1\train"
WAV_TEST_PATH = r"C:\Users\navya\OneDrive\Desktop\ICTC\Cocktail Party\Model\gender-audio-classification\VoxCeleb1\test"
METADATA_PATH = r"C:\Users\navya\OneDrive\Desktop\ICTC\Cocktail Party\Model\gender-audio-classification\VoxCeleb1\vox1_meta.prn"

def load_metadata():
    """Load gender labels from VoxCeleb1 metadata"""
    try:
        # Read metadata with tab separator
        df = pd.read_csv(METADATA_PATH, sep='\t', 
                        names=['ID', 'Name', 'Gender', 'Country', 'Set'])
        gender_dict = dict(zip(df['ID'], df['Gender']))
        
        print("\nMetadata loaded successfully")
        print(f"Total speakers in metadata: {len(gender_dict)}")
        print("\nGender distribution:")
        print(df['Gender'].value_counts())
        return gender_dict
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return {}

def get_speaker_id(file_path):
    """Extract speaker ID from VoxCeleb1 file path"""
    try:
        parts = file_path.split(os.sep)
        return parts[-3]  # Get the 'id10270' part
    except Exception as e:
        print(f"Error extracting speaker ID from {file_path}: {str(e)}")
        return None

# ...existing pad_or_truncate function...
# ...existing preprocess_audio function...

def find_wav_files(root_dir):
    """Recursively find all WAV files"""
    wav_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files


def pad_or_truncate(array, target_length):
    """Standardize array length"""
    current_length = array.shape[1]
    if (current_length > target_length):
        return array[:, :target_length]
    elif (current_length < target_length):
        pad_width = ((0, 0), (0, target_length - current_length))
        return np.pad(array, pad_width, mode='constant')
    return array

def preprocess_audio(file_path, sr=22050, duration=3, target_length=130):
    """Extract audio features with fixed length output"""
    try:
        # Load and pre-emphasize audio
        audio, sr = librosa.load(file_path, sr=sr, duration=duration)
        emphasized_signal = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # Window parameters
        frame_length = int(sr * 0.025)  # 25ms window
        hop_length = int(sr * 0.01)    # 10ms hop
        
        # Extract MFCC features with corrected window parameter
        mfcc = librosa.feature.mfcc(
            y=emphasized_signal,
            sr=sr,
            n_mfcc=23,
            dct_type=2,
            norm='ortho',
            lifter=22,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=40,
            center=False,
            window=np.hanning(frame_length)  # Using numpy's hanning window
        )
        
        # Extract additional features with corrected window parameters
        spectral_centroids = librosa.feature.spectral_centroid(
            y=emphasized_signal,
            sr=sr,
            n_fft=frame_length,
            hop_length=hop_length,
            window=np.hanning(frame_length)
        )
        
        zero_crossings = librosa.feature.zero_crossing_rate(
            emphasized_signal,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Ensure all features have same length
        min_length = min(mfcc.shape[1], spectral_centroids.shape[1], zero_crossings.shape[1])
        mfcc = mfcc[:, :min_length]
        spectral_centroids = spectral_centroids[:, :min_length]
        zero_crossings = zero_crossings[:, :min_length]
        
        # Stack features
        features = np.vstack([mfcc])
        
        # Pad or truncate to fixed length
        features = pad_or_truncate(features, target_length)
        
        return features, {
            'mfcc_shape': mfcc.shape,
            'spectral_centroids_shape': spectral_centroids.shape,
            'zero_crossings_shape': zero_crossings.shape
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None

def process_dataset(wav_path, gender_dict, dataset_type):
    """Process either train or test dataset"""
    wav_files = find_wav_files(wav_path)
    print(f"\nFound {len(wav_files)} WAV files in {dataset_type} set")
    
    X, y, metadata = [], [], []
    processed_count = 0
    
    print(f"\nProcessing {dataset_type} audio files...")
    for wav_file in tqdm(wav_files):
        try:
            speaker_id = get_speaker_id(wav_file)
            if not speaker_id:
                continue

            gender = gender_dict.get(speaker_id)
            if not gender:
                continue

            features, info = preprocess_audio(wav_file)
            if features is None:
                continue

            X.append(features)
            y.append(1 if gender.lower() == 'm' else 0)
            metadata.append({
                'file_path': wav_file,
                'speaker_id': speaker_id,
                'gender': gender,
                'set': dataset_type
            })
            processed_count += 1

        except Exception as e:
            print(f"\nError processing {wav_file}: {str(e)}")
    
    return np.array(X), np.array(y), metadata, processed_count

def main():
    # Verify paths exist
    for path, name in [(WAV_TRAIN_PATH, 'train'), (WAV_TEST_PATH, 'test'), (METADATA_PATH, 'metadata')]:
        if not os.path.exists(path):
            print(f"Error: {name} path does not exist: {path}")
            return

    # Load metadata
    print("Loading metadata...")
    gender_dict = load_metadata()
    if not gender_dict:
        print("Error: Failed to load metadata")
        return

    # Process train and test sets
    X_train, y_train, train_metadata, train_count = process_dataset(WAV_TRAIN_PATH, gender_dict, "train")
    X_test, y_test, test_metadata, test_count = process_dataset(WAV_TEST_PATH, gender_dict, "test")

    # Save results if files were processed
    if train_count > 0 or test_count > 0:
        output_dir = "preprocessed_data_VoxCeleb1"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train features and labels
        np.save(os.path.join(output_dir, 'preprocess_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'labels_train.npy'), y_train)
        
        # Save test features and labels
        np.save(os.path.join(output_dir, 'preprocess_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'labels_test.npy'), y_test)
        
        # Also save combined labels for compatibility
        all_labels = np.concatenate([y_train, y_test])
        np.save(os.path.join(output_dir, 'labels_VoxCeleb1.npy'), all_labels)
        
        # Save metadata
        all_metadata = train_metadata + test_metadata
        pd.DataFrame(all_metadata).to_csv(os.path.join(output_dir, 'metadata_VoxCeleb1.csv'), index=False)
        
        print("\nProcessing Summary:")
        print(f"Train set: {train_count} files processed")
        print(f"Test set: {test_count} files processed")
        print(f"\nTrain features shape: {X_train.shape}")
        print(f"Train labels shape: {y_train.shape}")
        print(f"Test features shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Total labels shape: {all_labels.shape}")
        print("\nGender distribution in train set:")
        print(pd.Series(y_train).value_counts().map({1: 'Male', 0: 'Female'}))
        print("\nGender distribution in test set:")
        print(pd.Series(y_test).value_counts().map({1: 'Male', 0: 'Female'}))
    else:
        print("\nNo files were successfully processed!")

if __name__ == "__main__":
    main()