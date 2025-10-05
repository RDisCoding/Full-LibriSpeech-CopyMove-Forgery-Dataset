import os
import time
import random
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from collections import Counter
import psutil  # For memory monitoring

class FullDatasetCopyMoveGenerator:
    def __init__(self, librispeech_path, output_path, target_duration=10.0):
        """
        Full dataset generator for copy-move forgery detection
        """
        self.librispeech_path = librispeech_path
        self.output_path = output_path
        self.target_duration = target_duration
        self.sr = 16000  # Standard sample rate
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        self.create_directories()
    
    def create_directories(self):
        """Create framework-friendly output structure"""
        splits = ['train', 'val', 'test']
        categories = ['original', 'forged']
        
        for split in splits:
            for category in categories:
                # Audio files
                audio_dir = os.path.join(self.output_path, 'audio', split, category)
                os.makedirs(audio_dir, exist_ok=True)
                
                # Spectrogram images
                spec_dir = os.path.join(self.output_path, 'spectrograms', split, category)
                os.makedirs(spec_dir, exist_ok=True)
        
        # Metadata directory
        os.makedirs(os.path.join(self.output_path, 'metadata'), exist_ok=True)
        
        # Progress tracking
        os.makedirs(os.path.join(self.output_path, 'progress'), exist_ok=True)
    
    def log_progress(self, message, split_name=None):
        """Log progress to file and console"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        # Write to log file
        log_file = os.path.join(self.output_path, 'progress', 'processing_log.txt')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def get_system_stats(self):
        """Get current system resource usage"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        return {
            'memory_used_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
            'cpu_percent': cpu
        }
    
    def normalize_audio_length(self, audio, target_samples):
        """Ensure consistent audio length to prevent data leakage"""
        if len(audio) > target_samples:
            start_idx = random.randint(0, len(audio) - target_samples)
            return audio[start_idx:start_idx + target_samples]
        else:
            padding = target_samples - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
    
    def add_subtle_modifications(self, segment, sr):
        """Add subtle modifications to copied segments to make them more detectable"""
        from scipy.signal import butter, filtfilt
        from scipy.ndimage import shift
        
        modified_segment = segment.copy()
        modification_type = random.choice(['noise', 'filter', 'gain', 'phase_shift', 'compression'])
        
        if modification_type == 'noise':
            # Add very subtle noise
            noise_level = random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_level, len(segment))
            modified_segment += noise
            
        elif modification_type == 'filter':
            # Apply subtle filtering
            if random.choice([True, False]):
                # High-pass filter
                cutoff = random.uniform(50, 200) / (sr / 2)
                b, a = butter(2, cutoff, btype='high')
                modified_segment = filtfilt(b, a, modified_segment)
            else:
                # Low-pass filter
                cutoff = random.uniform(6000, 7500) / (sr / 2)
                b, a = butter(2, cutoff, btype='low')
                modified_segment = filtfilt(b, a, modified_segment)
                
        elif modification_type == 'gain':
            # Subtle gain change
            gain_factor = random.uniform(0.95, 1.05)
            modified_segment *= gain_factor
            
        elif modification_type == 'phase_shift':
            # Subtle phase shift through time delay
            delay_samples = random.randint(-2, 2)
            if delay_samples != 0:
                modified_segment = shift(modified_segment, delay_samples, cval=0)
                
        elif modification_type == 'compression':
            # Simple dynamic range compression
            threshold = 0.5
            ratio = random.uniform(2, 4)
            compressed = np.where(
                np.abs(modified_segment) > threshold,
                np.sign(modified_segment) * (threshold + (np.abs(modified_segment) - threshold) / ratio),
                modified_segment
            )
            modified_segment = compressed
        
        return modified_segment, modification_type

    def create_diverse_forgery(self, audio, sr, forgery_strength='medium'):
        """Create enhanced copy-move forgeries with multiple segments and modifications"""
        duration = len(audio)
        forged_audio = audio.copy()
        
        # Determine number of forgeries and parameters based on strength
        if forgery_strength == 'subtle':
            num_forgeries = random.randint(1, 2)
            seg_duration_range = (0.5, 1.2)
            min_gap = int(sr * 0.5)
            modification_prob = 0.5
        elif forgery_strength == 'medium':
            num_forgeries = random.randint(2, 3)
            seg_duration_range = (0.8, 2.0)
            min_gap = int(sr * 0.3)
            modification_prob = 0.7
        else:  # obvious
            num_forgeries = random.randint(3, 4)
            seg_duration_range = (1.0, 2.8)
            min_gap = int(sr * 0.2)
            modification_prob = 0.8
        
        # Track used regions to avoid overlaps
        used_regions = []
        forgery_details = []
        modifications_applied = []
        
        for forgery_idx in range(num_forgeries):
            # Random segment duration
            seg_duration = random.uniform(*seg_duration_range)
            seg_len = int(sr * seg_duration)
            
            if duration < seg_len * 3:
                seg_len = duration // 6  # Smaller segments for shorter audio
            
            # Find non-overlapping source region
            attempts = 0
            while attempts < 15:
                max_source_start = int(duration * 0.6) - seg_len
                if max_source_start <= 0:
                    max_source_start = duration // 4
                
                source_start = random.randint(0, max_source_start)
                source_end = source_start + seg_len
                
                # Check for overlap with used regions
                overlap = False
                for used_start, used_end in used_regions:
                    if not (source_end <= used_start or source_start >= used_end):
                        overlap = True
                        break
                
                if not overlap:
                    break
                attempts += 1
            
            if attempts >= 15:
                continue  # Skip this forgery if can't find non-overlapping region
            
            # Find target region
            attempts = 0
            while attempts < 15:
                min_target_start = source_end + min_gap
                max_target_start = duration - seg_len
                
                if min_target_start >= max_target_start:
                    min_target_start = int(duration * 0.7)
                    max_target_start = duration - seg_len
                    if min_target_start >= max_target_start:
                        break
                
                target_start = random.randint(min_target_start, max_target_start)
                target_end = target_start + seg_len
                
                # Check for overlap with used regions
                overlap = False
                for used_start, used_end in used_regions:
                    if not (target_end <= used_start or target_start >= used_end):
                        overlap = True
                        break
                
                if not overlap:
                    break
                attempts += 1
            
            if attempts >= 15:
                continue  # Skip this forgery if can't find non-overlapping target
            
            # Extract segment
            segment = audio[source_start:source_end].copy()
            modification_type = 'none'
            
            # Apply modifications with probability
            if random.random() < modification_prob:
                try:
                    segment, modification_type = self.add_subtle_modifications(segment, sr)
                except:
                    modification_type = 'failed'
            
            # Apply forgery
            forged_audio[target_start:target_end] = segment
            
            # Record details
            forgery_details.append({
                'source_start': source_start / sr,
                'source_end': source_end / sr,
                'target_start': target_start / sr,
                'target_end': target_end / sr,
                'duration': seg_duration,
                'modification': modification_type,
                'forgery_idx': forgery_idx
            })
            
            modifications_applied.append(modification_type)
            
            # Mark regions as used
            used_regions.append((source_start, source_end))
            used_regions.append((target_start, target_end))
        
        # Return info about the first forgery for compatibility
        if forgery_details:
            first_forgery = forgery_details[0]
            return forged_audio, {
                'source_start': first_forgery['source_start'],
                'source_end': first_forgery['source_end'],
                'target_start': first_forgery['target_start'],
                'target_end': first_forgery['target_end'],
                'duration': first_forgery['duration'],
                'strength': forgery_strength,
                'num_forgeries': len(forgery_details),
                'modifications': modifications_applied,
                'total_forged_duration': sum([f['duration'] for f in forgery_details])
            }
        else:
            # Fallback to single forgery if multi-forgery fails
            seg_duration = random.uniform(*seg_duration_range)
            seg_len = int(sr * seg_duration)
            
            if duration < seg_len * 3:
                seg_len = duration // 4
            
            max_source_start = int(duration * 0.6) - seg_len
            if max_source_start <= 0:
                max_source_start = duration // 4
                
            source_start = random.randint(0, max_source_start)
            source_end = source_start + seg_len
            segment = audio[source_start:source_end].copy()
            
            min_target_start = source_end + min_gap
            max_target_start = duration - seg_len
            
            if min_target_start >= max_target_start:
                min_target_start = int(duration * 0.7)
                max_target_start = duration - seg_len
                if min_target_start >= max_target_start:
                    min_target_start = max_target_start - 1
            
            target_start = random.randint(min_target_start, max_target_start)
            target_end = target_start + seg_len
            
            # Apply modification
            modification_type = 'none'
            if random.random() < modification_prob:
                try:
                    segment, modification_type = self.add_subtle_modifications(segment, sr)
                except:
                    modification_type = 'failed'
            
            forged_audio[target_start:target_end] = segment
            
            return forged_audio, {
                'source_start': source_start / sr,
                'source_end': source_end / sr,
                'target_start': target_start / sr,
                'target_end': target_end / sr,
                'duration': seg_duration,
                'strength': forgery_strength,
                'num_forgeries': 1,
                'modifications': [modification_type],
                'total_forged_duration': seg_duration
            }
    
    def audio_to_spectrogram(self, audio, sr, save_path=None):
        """Convert audio to CLEAN spectrogram for CNN models"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, hop_length=512, n_fft=2048
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if save_path:
            plt.figure(figsize=(10, 4))
            plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        return mel_spec_db
    
    def prevent_speaker_leakage(self, file_paths):
        """Ensure train/val/test splits don't share speakers"""
        speaker_files = {}
        
        for path in file_paths:
            try:
                speaker_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
                if speaker_id not in speaker_files:
                    speaker_files[speaker_id] = []
                speaker_files[speaker_id].append(path)
            except:
                continue
        
        speakers = list(speaker_files.keys())
        self.log_progress(f"Found {len(speakers)} unique speakers")
        
        if len(speakers) < 3:
            # Fallback split
            n = len(file_paths)
            return file_paths[:n//2], file_paths[n//2:3*n//4], file_paths[3*n//4:]
        
        train_speakers, temp_speakers = train_test_split(speakers, test_size=0.3, random_state=42)
        val_speakers, test_speakers = train_test_split(temp_speakers, test_size=0.5, random_state=42)
        
        train_files = []
        val_files = []
        test_files = []
        
        for speaker in train_speakers:
            train_files.extend(speaker_files[speaker])
        for speaker in val_speakers:
            val_files.extend(speaker_files[speaker])
        for speaker in test_speakers:
            test_files.extend(speaker_files[speaker])
        
        self.log_progress(f"Speaker split - Train: {len(train_speakers)}, Val: {len(val_speakers)}, Test: {len(test_speakers)}")
        return train_files, val_files, test_files
    
    def save_checkpoint(self, all_labels, split_name, processed_idx):
        """Save progress checkpoint"""
        checkpoint_file = os.path.join(self.output_path, 'progress', f'checkpoint_{split_name}_{processed_idx}.csv')
        if all_labels:
            checkpoint_df = pd.DataFrame(all_labels)
            checkpoint_df.to_csv(checkpoint_file, index=False)
            self.log_progress(f"Checkpoint saved: {checkpoint_file}")
    
    def generate_full_dataset(self, include_spectrograms=True, checkpoint_interval=500):
        """Generate the complete dataset from entire LibriSpeech"""
        
        self.start_time = time.time()
        self.log_progress("=" * 60)
        self.log_progress("STARTING FULL LIBRISPEECH COPY-MOVE DATASET GENERATION")
        self.log_progress("=" * 60)
        
        # Collect all files
        self.log_progress("🔍 Collecting all audio files...")
        all_files = glob(os.path.join(self.librispeech_path, "**/*.flac"), recursive=True)
        self.log_progress(f"📁 Found {len(all_files)} total audio files")
        
        if len(all_files) == 0:
            self.log_progress("❌ No .flac files found!")
            return None
        
        # Split by speakers
        train_files, val_files, test_files = self.prevent_speaker_leakage(all_files)
        
        self.log_progress(f"📊 Final split sizes:")
        self.log_progress(f"   Train: {len(train_files)} files")
        self.log_progress(f"   Val: {len(val_files)} files") 
        self.log_progress(f"   Test: {len(test_files)} files")
        
        all_labels = []
        target_samples = int(self.sr * self.target_duration)
        
        # Process each split
        for split_name, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
            self.log_progress(f"\n🎵 Processing {split_name} split ({len(file_list)} files)...")
            
            # Create balanced forgery strength distribution
            split_size = len(file_list)
            strengths = (['subtle'] * (split_size // 3) + 
                        ['medium'] * (split_size // 3) + 
                        ['obvious'] * (split_size - 2 * (split_size // 3)))
            random.shuffle(strengths)
            
            split_start_time = time.time()
            
            for i, file_path in enumerate(file_list):
                base_name = f"{split_name}_{i:04d}"
                
                try:
                    # Load and normalize audio
                    audio, sr = librosa.load(file_path, sr=self.sr)
                    audio = self.normalize_audio_length(audio, target_samples)
                    
                    # Save original audio
                    orig_audio_path = os.path.join(self.output_path, 'audio', split_name, 'original', f"{base_name}.wav")
                    sf.write(orig_audio_path, audio, self.sr)
                    
                    # Create spectrogram for original
                    if include_spectrograms:
                        orig_spec_path = os.path.join(self.output_path, 'spectrograms', split_name, 'original', f"{base_name}.png")
                        self.audio_to_spectrogram(audio, self.sr, orig_spec_path)
                    
                    # Add original to labels
                    all_labels.append({
                        'filename': f"{base_name}.wav",
                        'split': split_name,
                        'label': 'original',
                        'forgery_type': 'none',
                        'source_start': None,
                        'source_end': None,
                        'target_start': None,
                        'target_end': None,
                        'forgery_strength': None,
                        'audio_path': f"audio/{split_name}/original/{base_name}.wav",
                        'spectrogram_path': f"spectrograms/{split_name}/original/{base_name}.png" if include_spectrograms else None
                    })
                    
                    # Create forgery
                    forgery_strength = strengths[i] if i < len(strengths) else random.choice(['subtle', 'medium', 'obvious'])
                    forged_audio, forgery_info = self.create_diverse_forgery(audio, self.sr, forgery_strength)
                    
                    # Save forged audio
                    forged_audio_path = os.path.join(self.output_path, 'audio', split_name, 'forged', f"{base_name}.wav")
                    sf.write(forged_audio_path, forged_audio, self.sr)
                    
                    # Create spectrogram for forged
                    if include_spectrograms:
                        forged_spec_path = os.path.join(self.output_path, 'spectrograms', split_name, 'forged', f"{base_name}.png")
                        self.audio_to_spectrogram(forged_audio, self.sr, forged_spec_path)
                    
                    # Add forged to labels
                    all_labels.append({
                        'filename': f"{base_name}.wav",
                        'split': split_name,
                        'label': 'forged',
                        'forgery_type': 'copymove',
                        'source_start': forgery_info['source_start'],
                        'source_end': forgery_info['source_end'],
                        'target_start': forgery_info['target_start'],
                        'target_end': forgery_info['target_end'],
                        'forgery_strength': forgery_info['strength'],
                        'audio_path': f"audio/{split_name}/forged/{base_name}.wav",
                        'spectrogram_path': f"spectrograms/{split_name}/forged/{base_name}.png" if include_spectrograms else None
                    })
                    
                    self.processed_count += 1
                    
                    # Progress reporting
                    if (i + 1) % 100 == 0:
                        elapsed = time.time() - split_start_time
                        rate = (i + 1) / elapsed if elapsed > 0 else 0
                        eta = (len(file_list) - i - 1) / rate if rate > 0 else 0
                        
                        stats = self.get_system_stats()
                        self.log_progress(
                            f"  {split_name}: {i+1}/{len(file_list)} "
                            f"({(i+1)/len(file_list)*100:.1f}%) - "
                            f"Rate: {rate:.1f} files/sec - "
                            f"ETA: {eta/60:.1f}min - "
                            f"RAM: {stats['memory_percent']:.1f}% - "
                            f"CPU: {stats['cpu_percent']:.1f}%"
                        )
                    
                    # Save checkpoint
                    if (i + 1) % checkpoint_interval == 0:
                        self.save_checkpoint(all_labels, split_name, i + 1)
                
                except Exception as e:
                    self.error_count += 1
                    self.log_progress(f"⚠️ Error processing {file_path}: {e}")
                    continue
            
            # End of split summary
            split_elapsed = time.time() - split_start_time
            self.log_progress(f"✅ Completed {split_name} split in {split_elapsed/60:.1f} minutes")
        
        # Save final labels
        self.log_progress("\n💾 Saving final labels and metadata...")
        labels_df = pd.DataFrame(all_labels)
        
        # Save comprehensive labels
        labels_path = os.path.join(self.output_path, 'metadata', 'labels.csv')
        labels_df.to_csv(labels_path, index=False)
        
        # Save split-specific labels
        for split in ['train', 'val', 'test']:
            split_labels = labels_df[labels_df['split'] == split]
            split_path = os.path.join(self.output_path, 'metadata', f'{split}_labels.csv')
            split_labels.to_csv(split_path, index=False)
        
        # Create final summary
        self.create_final_summary(labels_df)
        
        total_elapsed = time.time() - self.start_time
        self.log_progress("\n" + "=" * 60)
        self.log_progress("🎉 FULL DATASET GENERATION COMPLETED!")
        self.log_progress(f"⏱️  Total time: {total_elapsed/3600:.2f} hours")
        self.log_progress(f"📊 Total files processed: {self.processed_count}")
        self.log_progress(f"📊 Total files generated: {len(labels_df)}")
        self.log_progress(f"❌ Errors encountered: {self.error_count}")
        self.log_progress("=" * 60)
        
        return labels_df
    
    def create_final_summary(self, labels_df):
        """Create comprehensive final summary"""
        summary_path = os.path.join(self.output_path, 'metadata', 'final_dataset_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=== FULL LIBRISPEECH COPY-MOVE FORGERY DATASET ===\n\n")
            f.write(f"Generation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Time: {(time.time() - self.start_time)/3600:.2f} hours\n")
            f.write(f"Target Duration: {self.target_duration} seconds\n")
            f.write(f"Sample Rate: {self.sr} Hz\n")
            f.write(f"Files Processed: {self.processed_count}\n")
            f.write(f"Files Generated: {len(labels_df)}\n")
            f.write(f"Errors: {self.error_count}\n\n")
            
            f.write("=== DATASET STATISTICS ===\n")
            for split in ['train', 'val', 'test']:
                split_data = labels_df[labels_df['split'] == split]
                orig_count = len(split_data[split_data['label'] == 'original'])
                forged_count = len(split_data[split_data['label'] == 'forged'])
                f.write(f"{split.title()}: {len(split_data)} total ({orig_count} original + {forged_count} forged)\n")
            
            f.write(f"\n=== FORGERY STRENGTH DISTRIBUTION ===\n")
            forged_data = labels_df[labels_df['label'] == 'forged']
            strength_counts = forged_data['forgery_strength'].value_counts()
            for strength, count in strength_counts.items():
                f.write(f"{strength}: {count}\n")

def main():
    """Process the entire LibriSpeech dataset"""
    
    print("🚀 FULL LIBRISPEECH COPY-MOVE DATASET GENERATOR")
    print("=" * 60)
    print("⚠️  This will process ALL files in LibriSpeech train-clean-100")
    print("⚠️  Expected processing time: 6-12 hours")
    print("⚠️  Required disk space: ~50-100 GB")
    print("⚠️  System resources: High CPU and RAM usage")
    print("=" * 60)
    
    # Get user confirmation
    response = input("\nDo you want to continue with full dataset generation? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Operation cancelled by user")
        return
    
    # Create generator
    generator = FullDatasetCopyMoveGenerator(
        librispeech_path="D:\\Audio Fraud Detection System\\Dataset\\original_dataset\\LibriSpeech\\train-clean-100",
        output_path="full_librispeech_copymove_dataset",
        target_duration=10.0
    )
    
    # Generate full dataset
    labels_df = generator.generate_full_dataset(
        include_spectrograms=True,
        checkpoint_interval=500  # Save checkpoint every 500 files
    )
    
    if labels_df is not None:
        print(f"\n✅ SUCCESS! Generated {len(labels_df)} total files")
        print(f"📁 Output: full_librispeech_copymove_dataset/")
        print(f"📋 Check processing_log.txt for detailed progress")

if __name__ == "__main__":
    main()

def apply_forgery_to_file(input_file, output_file, forgery_strength='medium'):
    """Apply copy-move forgery to a specific audio file and save the forged version."""
    generator = FullDatasetCopyMoveGenerator(None, None)
    
    # Load the audio file
    audio, sr = librosa.load(input_file, sr=generator.sr)
    
    # Apply forgery
    forged_audio, forgery_info = generator.create_diverse_forgery(audio, sr, forgery_strength)
    
    # Save the forged audio
    sf.write(output_file, forged_audio, sr)
    
    print(f"Forged audio saved to: {output_file}")
    print("Forgery details:", forgery_info)

# Example usage
if __name__ == "__main__":
    input_file = "not_using/test.mp3"
    output_file = "not_using/test_forged.wav"
    apply_forgery_to_file(input_file, output_file, forgery_strength='medium')
