import os
import torch
import torchaudio
from pydub import AudioSegment
from pydub.playback import play
from speechbrain.pretrained import SpectralMaskEnhancement

# Convert M4A to WAV and make it mono
def convert_to_wav(input_file, output_file):
    print(f"Converting {input_file} to WAV...")
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)  # Convert to mono
    audio.export(output_file, format="wav")
    print(f"Converted file saved at {output_file}")

# Split audio into chunks
def split_audio(file_path, chunk_length_ms=60000):
    print("Splitting audio into chunks...")
    audio = AudioSegment.from_file(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    output_folder = os.path.join(os.path.dirname(file_path), "chunks")
    os.makedirs(output_folder, exist_ok=True)
    chunk_files = []
    for idx, chunk in enumerate(chunks):
        chunk_path = os.path.join(output_folder, f"chunk_{idx + 1}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_files.append(chunk_path)
    print(f"Audio split into {len(chunks)} chunks.")
    return chunk_files

def preprocess_audio(input_path, output_path):
    """Convert input audio to 16 kHz, mono WAV format."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # Convert to 16 kHz mono
    audio.export(output_path, format="wav")
    return output_path

def process_chunks(input_files, output_folder):
    print("Enhancing audio chunks...")

    # Path to locally available pretrained model
    local_model_dir = "E:\\Abin Roy\\Codes\\miniproject\\noise_removal_abin\\pretrained_models\\metricgan-plus-voicebank"

    # Load the pretrained SpeechBrain model using the local directory
    enhancer = SpectralMaskEnhancement.from_hparams(
        source=local_model_dir,
        hparams_file=os.path.join(local_model_dir, "hyperparams.yaml"),
        savedir=local_model_dir
    )

    enhanced_files = []
    for idx, file in enumerate(input_files):
        print(f"Processing chunk {idx + 1} of {len(input_files)}...")

        # Preprocess input audio
        temp_preprocessed_path = os.path.join(output_folder, f"preprocessed_chunk_{idx + 1}.wav")
        preprocess_audio(file, temp_preprocessed_path)

        # Load the preprocessed audio
        waveform, sample_rate = torchaudio.load(temp_preprocessed_path)
        print(f"Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")

        if sample_rate != 16000:
            raise ValueError(f"Sample rate mismatch: Expected 16 kHz, got {sample_rate}")

        # Ensure waveform is 2D: [batch_size, time_steps]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension if missing

        if waveform.ndim > 2:
            raise ValueError(f"Waveform has unexpected dimensions: {waveform.shape}")

        # Enhance the audio
        enhanced_audio = enhancer.enhance_batch(
            waveform, lengths=torch.tensor([1.0])
        )
        enhanced_audio = enhanced_audio.squeeze(0)  # Remove batch dimension

        # Save the enhanced audio
        enhanced_path = os.path.join(output_folder, f"enhanced_chunk_{idx + 1}.wav")
        torchaudio.save(enhanced_path, enhanced_audio.unsqueeze(0), sample_rate=16000)
        enhanced_files.append(enhanced_path)

    print("Audio enhancement complete.")
    return enhanced_files

# Combine enhanced chunks into a single audio file
def combine_chunks(enhanced_files, output_file):
    print("Combining enhanced chunks into one file...")
    combined_audio = AudioSegment.empty()
    for file in enhanced_files:
        audio = AudioSegment.from_file(file)
        combined_audio += audio
    combined_audio.export(output_file, format="wav")
    print(f"Enhanced audio saved at {output_file}")

# Main function
def main():
    input_file = "SoumyaMiss.m4a"  # Replace with your input file
    temp_wav = "converted_input.wav"
    output_folder = "enhanced_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Convert M4A to WAV
    convert_to_wav(input_file, temp_wav)

    # Step 2: Split WAV into chunks
    chunk_files = split_audio(temp_wav)

    # Step 3: Enhance each chunk
    enhanced_files = process_chunks(chunk_files, output_folder)

    # Step 4: Combine enhanced chunks into a single WAV file
    final_output = "final_enhanced_audio.wav"
    combine_chunks(enhanced_files, final_output)

# Run the program
if __name__ == "__main__":
    main()
