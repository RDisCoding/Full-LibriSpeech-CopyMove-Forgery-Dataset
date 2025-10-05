import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt

class CopyMoveForgeryGenerator:
    def __init__(self, input_audio_path, output_dir="copymove_dataset"):
        """
        Initialize the Copy-Move Forgery Generator
        
        Args:
            input_audio_path (str): Path to the input .aac audio file
            output_dir (str): Directory to save the generated dataset
        """
        self.input_audio_path = input_audio_path
        self.output_dir = output_dir
        self.create_output_directories()
    
    def create_output_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "forged"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
    
    def load_audio(self, file_path):
        """
        Load audio file (supports .aac, .wav, .mp3, etc.)
        
        Returns:
            audio (np.array): Audio samples
            sr (int): Sample rate
        """
        try:
            # First try with librosa (handles most formats)
            audio, sr = librosa.load(file_path, sr=None)
            return audio, sr
        except:
            # Fallback to pydub for AAC files
            audio_segment = AudioSegment.from_file(file_path)
            # Convert to mono and get raw audio data
            audio_segment = audio_segment.set_channels(1)
            sr = audio_segment.frame_rate
            # Convert to numpy array
            audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            # Normalize to [-1, 1]
            audio = audio / np.max(np.abs(audio))
            return audio, sr
    
    def create_simple_copymove_forgery(self, audio, sr, min_segment_duration=1.0, max_segment_duration=3.0):
        """
        Create a simple copy-move forgery by copying a segment to another location
        
        Args:
            audio (np.array): Original audio samples
            sr (int): Sample rate
            min_segment_duration (float): Minimum segment duration in seconds
            max_segment_duration (float): Maximum segment duration in seconds
        
        Returns:
            tuple: (forged_audio, source_start, source_end, target_start, target_end)
        """
        duration = len(audio) / sr
        
        # Choose segment duration
        seg_duration = random.uniform(min_segment_duration, 
                                    min(max_segment_duration, duration * 0.3))
        seg_len_samples = int(sr * seg_duration)
        
        # Choose source segment (from first half of audio)
        max_source_start = int(len(audio) * 0.6) - seg_len_samples
        source_start_sample = random.randint(0, max_source_start)
        source_end_sample = source_start_sample + seg_len_samples
        
        # Extract the segment to copy
        segment = audio[source_start_sample:source_end_sample].copy()
        
        # Choose target location (ensuring no overlap with source)
        min_target_start = source_end_sample + int(sr * 0.5)  # 0.5 sec gap
        max_target_start = len(audio) - seg_len_samples
        
        if min_target_start >= max_target_start:
            # If not enough space, place in second half
            min_target_start = int(len(audio) * 0.7)
            max_target_start = len(audio) - seg_len_samples
        
        target_start_sample = random.randint(min_target_start, max_target_start)
        target_end_sample = target_start_sample + seg_len_samples
        
        # Create forged audio by replacing target region with copied segment
        forged_audio = audio.copy()
        forged_audio[target_start_sample:target_end_sample] = segment
        
        # Convert to time stamps
        source_start_time = source_start_sample / sr
        source_end_time = source_end_sample / sr
        target_start_time = target_start_sample / sr
        target_end_time = target_end_sample / sr
        
        return (forged_audio, source_start_time, source_end_time, 
                target_start_time, target_end_time)
    
    def create_advanced_copymove_forgery(self, audio, sr):
        """
        Create a more sophisticated copy-move forgery with multiple segments
        
        Returns:
            tuple: (forged_audio, forgery_info)
        """
        forged_audio = audio.copy()
        forgery_info = []
        
        # Create 2-3 copy-move operations
        num_operations = random.randint(2, 3)
        
        for i in range(num_operations):
            # Segment duration between 0.5 and 2 seconds
            seg_duration = random.uniform(0.5, 2.0)
            seg_len_samples = int(sr * seg_duration)
            
            # Choose source segment
            max_start = len(forged_audio) - seg_len_samples - 1
            if max_start <= 0:
                continue
                
            source_start = random.randint(0, max_start // 2)
            source_end = source_start + seg_len_samples
            segment = forged_audio[source_start:source_end].copy()
            
            # Choose target location
            target_start = random.randint(source_end + int(sr * 0.3), max_start)
            target_end = target_start + seg_len_samples
            
            # Apply forgery
            forged_audio[target_start:target_end] = segment
            
            forgery_info.append({
                'operation': i + 1,
                'source_start': source_start / sr,
                'source_end': source_end / sr,
                'target_start': target_start / sr,
                'target_end': target_end / sr,
                'duration': seg_duration
            })
        
        return forged_audio, forgery_info
    
    def visualize_forgery(self, original_audio, forged_audio, sr, forgery_info, output_filename):
        """
        Create visualization of the copy-move forgery
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time axis
        time = np.linspace(0, len(original_audio) / sr, len(original_audio))
        
        # Plot original audio
        ax1.plot(time, original_audio, alpha=0.7, color='blue')
        ax1.set_title('Original Audio Waveform')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Plot forged audio with highlights
        ax2.plot(time[:len(forged_audio)], forged_audio, alpha=0.7, color='red')
        ax2.set_title('Forged Audio Waveform (Copy-Move Forgery)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # Highlight forged regions
        if isinstance(forgery_info, list):
            # Multiple forgeries
            colors = ['yellow', 'green', 'orange', 'purple']
            for i, info in enumerate(forgery_info):
                color = colors[i % len(colors)]
                # Highlight source
                ax2.axvspan(info['source_start'], info['source_end'], 
                           alpha=0.3, color=color, label=f'Source {i+1}')
                # Highlight target
                ax2.axvspan(info['target_start'], info['target_end'], 
                           alpha=0.5, color=color, label=f'Target {i+1}')
        else:
            # Single forgery (5-tuple format)
            source_start, source_end, target_start, target_end = forgery_info[1:5]
            ax2.axvspan(source_start, source_end, alpha=0.3, color='yellow', label='Source Segment')
            ax2.axvspan(target_start, target_end, alpha=0.5, color='orange', label='Forged Segment')
        
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualizations", output_filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_dataset(self, num_simple_forgeries=3, num_advanced_forgeries=2):
        """
        Generate a complete copy-move forgery dataset
        
        Args:
            num_simple_forgeries (int): Number of simple copy-move forgeries to create
            num_advanced_forgeries (int): Number of advanced copy-move forgeries to create
        """
        print(f"Loading audio file: {self.input_audio_path}")
        original_audio, sr = self.load_audio(self.input_audio_path)
        
        # Get base filename
        base_filename = os.path.splitext(os.path.basename(self.input_audio_path))[0]
        
        # Save original audio
        original_wav_path = os.path.join(self.output_dir, "original", f"{base_filename}_original.wav")
        sf.write(original_wav_path, original_audio, sr)
        
        labels = []
        
        # Add original file to labels
        labels.append({
            "filename": f"{base_filename}_original.wav",
            "label": "original",
            "forgery_type": "none",
            "source_start": None,
            "source_end": None,
            "target_start": None,
            "target_end": None,
            "num_operations": 0
        })
        
        print(f"Original audio duration: {len(original_audio)/sr:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        
        # Generate simple copy-move forgeries
        print(f"\nGenerating {num_simple_forgeries} simple copy-move forgeries...")
        for i in range(num_simple_forgeries):
            forged_audio, source_start, source_end, target_start, target_end = \
                self.create_simple_copymove_forgery(original_audio, sr)
            
            # Save forged audio
            forged_filename = f"{base_filename}_simple_forgery_{i+1}.wav"
            forged_path = os.path.join(self.output_dir, "forged", forged_filename)
            sf.write(forged_path, forged_audio, sr)
            
            # Add to labels
            labels.append({
                "filename": forged_filename,
                "label": "forged",
                "forgery_type": "simple_copymove",
                "source_start": source_start,
                "source_end": source_end,
                "target_start": target_start,
                "target_end": target_end,
                "num_operations": 1
            })
            
            # Create visualization
            vis_filename = f"{base_filename}_simple_forgery_{i+1}_visualization.png"
            forgery_info = (forged_audio, source_start, source_end, target_start, target_end)
            self.visualize_forgery(original_audio, forged_audio, sr, forgery_info, vis_filename)
            
            print(f"  Simple forgery {i+1}: Source [{source_start:.2f}-{source_end:.2f}s] -> Target [{target_start:.2f}-{target_end:.2f}s]")
        
        # Generate advanced copy-move forgeries
        print(f"\nGenerating {num_advanced_forgeries} advanced copy-move forgeries...")
        for i in range(num_advanced_forgeries):
            forged_audio, forgery_info = self.create_advanced_copymove_forgery(original_audio, sr)
            
            # Save forged audio
            forged_filename = f"{base_filename}_advanced_forgery_{i+1}.wav"
            forged_path = os.path.join(self.output_dir, "forged", forged_filename)
            sf.write(forged_path, forged_audio, sr)
            
            # Add to labels (simplified for advanced forgeries)
            labels.append({
                "filename": forged_filename,
                "label": "forged",
                "forgery_type": "advanced_copymove",
                "source_start": forgery_info[0]['source_start'] if forgery_info else None,
                "source_end": forgery_info[0]['source_end'] if forgery_info else None,
                "target_start": forgery_info[0]['target_start'] if forgery_info else None,
                "target_end": forgery_info[0]['target_end'] if forgery_info else None,
                "num_operations": len(forgery_info)
            })
            
            # Create visualization
            vis_filename = f"{base_filename}_advanced_forgery_{i+1}_visualization.png"
            self.visualize_forgery(original_audio, forged_audio, sr, forgery_info, vis_filename)
            
            print(f"  Advanced forgery {i+1}: {len(forgery_info)} copy-move operations")
        
        # Save labels CSV
        labels_df = pd.DataFrame(labels)
        labels_csv_path = os.path.join(self.output_dir, "labels.csv")
        labels_df.to_csv(labels_csv_path, index=False)
        
        # Create summary report
        self.create_summary_report(labels_df, original_audio, sr)
        
        print(f"\n✅ Dataset creation completed!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Total files: {len(labels_df)} ({len(labels_df[labels_df['label']=='original'])} original, {len(labels_df[labels_df['label']=='forged'])} forged)")
        print(f"📋 Labels saved to: {labels_csv_path}")
        
        return labels_df
    
    def create_summary_report(self, labels_df, original_audio, sr):
        """Create a summary report of the generated dataset"""
        report_path = os.path.join(self.output_dir, "dataset_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("=== COPY-MOVE FORGERY DATASET SUMMARY ===\n\n")
            f.write(f"Source Audio File: {self.input_audio_path}\n")
            f.write(f"Original Audio Duration: {len(original_audio)/sr:.2f} seconds\n")
            f.write(f"Sample Rate: {sr} Hz\n")
            f.write(f"Total Samples: {len(original_audio)}\n\n")
            
            f.write("=== DATASET STATISTICS ===\n")
            f.write(f"Total Files: {len(labels_df)}\n")
            f.write(f"Original Files: {len(labels_df[labels_df['label']=='original'])}\n")
            f.write(f"Forged Files: {len(labels_df[labels_df['label']=='forged'])}\n\n")
            
            f.write("=== FORGERY BREAKDOWN ===\n")
            forgery_types = labels_df[labels_df['label']=='forged']['forgery_type'].value_counts()
            for forgery_type, count in forgery_types.items():
                f.write(f"{forgery_type}: {count} files\n")
            
            f.write("\n=== FILE DETAILS ===\n")
            for _, row in labels_df.iterrows():
                f.write(f"\nFile: {row['filename']}\n")
                f.write(f"  Label: {row['label']}\n")
                if row['label'] == 'forged':
                    f.write(f"  Forgery Type: {row['forgery_type']}\n")
                    f.write(f"  Operations: {row['num_operations']}\n")
                    if row['source_start'] is not None:
                        f.write(f"  Source: {row['source_start']:.2f}s - {row['source_end']:.2f}s\n")
                        f.write(f"  Target: {row['target_start']:.2f}s - {row['target_end']:.2f}s\n")


def main():
    """
    Main function to demonstrate usage
    """
    # Example usage - update this path to your AAC file
    input_aac_file = "sample_audio.aac"  # <-- Change this to your AAC file path
    
    # Check if file exists
    if not os.path.exists(input_aac_file):
        print(f"❌ Error: Audio file '{input_aac_file}' not found!")
        print("Please update the 'input_aac_file' variable with the correct path to your .aac file")
        return
    
    # Create forgery generator
    generator = CopyMoveForgeryGenerator(
        input_audio_path=input_aac_file,
        output_dir="copymove_dataset"
    )
    
    # Generate dataset
    labels_df = generator.generate_dataset(
        num_simple_forgeries=3,    # Number of simple copy-move forgeries
        num_advanced_forgeries=2   # Number of advanced copy-move forgeries
    )
    
    print("\n=== Dataset Preview ===")
    print(labels_df.head(10))


if __name__ == "__main__":
    main()
