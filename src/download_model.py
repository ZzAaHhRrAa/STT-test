import os
# Use the mirror just in case your network blocks the main model repo
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "openai/whisper-tiny"
save_directory = "./whisper-local"

print("📥 Downloading Whisper-Tiny weights and processor configuration...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

print(f"💾 Saving model files locally to '{save_directory}'...")
model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)

print("✅ Model downloaded and saved locally! You can now use it offline.")