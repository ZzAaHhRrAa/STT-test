import os
import pandas as pd
import soundfile as sf
from datasets import load_dataset

def prepare_data(output_dir="data/audio_files", csv_path="input.csv"):
    print("📦 Loading dataset directly from local snapshot directory...")
    
    # Path to your successful CLI snapshot download
    local_snapshot_path = "/home/kavano/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/5be91486e11a2d616f4ec5db8d3fd248585ac07a"
    
    # Load completely offline using the local cache configuration
    dataset = load_dataset(local_snapshot_path, "clean", split="validation")
    
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    print(f"💾 Extracting and saving audio files to {output_dir}...")
    for idx, item in enumerate(dataset):
        filename = f"sample_{idx}.wav"
        local_path = os.path.join(output_dir, filename)
        
        # Pull the audio array and ground truth text
        audio_array = item["audio"]["array"]
        sample_rate = item["audio"]["sampling_rate"]
        text = item["text"]
        
        # Save file locally as .wav
        sf.write(local_path, audio_array, sample_rate)
        
        metadata.append({
            "audio_path": os.path.abspath(local_path),
            "ground_truth": text
        })
        

    df = pd.DataFrame(metadata)
    df.to_csv(csv_path, index=False)
    print(f"✅ Data preparation complete! Created {csv_path}")

if __name__ == "__main__":
    prepare_data()