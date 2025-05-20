import numpy as np
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from music21 import stream, note, metadata
import requests
import pretty_midi
import librosa
import os
import subprocess
from pathlib import Path
from moviepy.editor import VideoFileClip
from PIL import Image
import tempfile
import shutil

# === Dummy AI Model ===
class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 768)

    def forward(self, x):
        return self.proj(x)

class PolytuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_score = DummyEncoder()
        self.encoder_perf = DummyEncoder()
        self.project = nn.Linear(768 * 2, 512)
        self.decoder = T5ForConditionalGeneration.from_pretrained("t5-small")

    def forward(self, score_input, perf_input, labels=None):
        enc1 = self.encoder_score(score_input)
        enc2 = self.encoder_perf(perf_input)
        joint = torch.cat((enc1, enc2), dim=-1)
        joint = self.project(joint).unsqueeze(1)
        decoder_input_ids = torch.tensor([[0]], dtype=torch.long)
        return self.decoder(inputs_embeds=joint, decoder_input_ids=decoder_input_ids, labels=labels)

# === MusicXML Conversion and Feedback ===
def token_ids_to_musicxml(predicted_tokens, error_labels):
    s = stream.Stream()
    s.insert(0, metadata.Metadata())
    s.metadata.title = "Performance Feedback"
    s.metadata.composer = "Polytune"
    for idx, token in enumerate(predicted_tokens):
        pitch = 60 + (token % 12)
        n = note.Note(pitch)
        n.quarterLength = 0.5
        label = error_labels[idx] if idx < len(error_labels) else 'correct'
        n.style.color = {'missed': 'red', 'extra': 'blue'}.get(label, 'green')
        s.append(n)
    return s

def export_musicxml(score_stream, filename="annotated_output.xml"):
    return score_stream.write('musicxml', fp=filename)

def get_llm_feedback(note_errors):
    prompt = f"""A student played with the following errors:\n{note_errors}\nGive feedback on improving intonation, rhythm, and dynamics.\nAlso give a grade out of 100."""
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "deepseek-r1:8b",  # ✅ Just the model name as pulled by Ollama
            "prompt": prompt,
            "stream": False
        })
        data = response.json()
        if "response" in data:
            return data["response"]
        else:
            print("⚠️ No 'response' key in LLM reply:", data)
            return "⚠️ LLM did not return valid feedback."
    except Exception as e:
        return f"❌ LLM feedback error: {e}"


# === Utility Functions ===
def extract_audio_from_video(video_path: str, output_audio_path: str = "video_audio.wav") -> str:
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)
    return output_audio_path

def convert_pdf_or_image_to_musicxml(file_path: str) -> str:
    print(f"🧠 Running Audiveris OMR on: {file_path}")
    temp_dir = tempfile.mkdtemp()
    audiveris_jar = r"C:\\Users\\ashiq\\audiveris\\app\\build\\jar\\audiveris.jar"  # Update to your path

    cmd = ["java", "-jar", audiveris_jar, "-batch", "-export", "-output", temp_dir, file_path]

    try:
        subprocess.run(cmd, check=True, timeout=60)
    except subprocess.CalledProcessError as e:
        print("❌ Audiveris failed:", e)
        return "converted_score.musicxml"
    except subprocess.TimeoutExpired:
        print("⏱️ Audiveris timed out")
        return "converted_score.musicxml"

    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.endswith(".mxl") or f.endswith(".xml"):
                final_path = os.path.abspath(f)
                shutil.move(os.path.join(root, f), final_path)
                shutil.rmtree(temp_dir)
                return final_path

    shutil.rmtree(temp_dir)
    return "converted_score.musicxml"

def audio_to_feature(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=16000)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    return librosa.power_to_db(spec, ref=np.max)

def compare_audio_metrics(midi_audio_path: str, ref_audio_path: str) -> list:
    midi_y, sr = librosa.load(midi_audio_path, sr=None)
    ref_y, _ = librosa.load(ref_audio_path, sr=sr)

    midi_onsets = librosa.onset.onset_detect(y=midi_y, sr=sr, units='time')
    ref_onsets = librosa.onset.onset_detect(y=ref_y, sr=sr, units='time')
    rhythm_diff = abs(len(midi_onsets) - len(ref_onsets))

    midi_pitch = librosa.yin(y=midi_y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    ref_pitch = librosa.yin(y=ref_y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    min_len = min(len(midi_pitch), len(ref_pitch))
    pitch_diff = np.nanmean(np.abs(midi_pitch[:min_len] - ref_pitch[:min_len]))

    midi_rms = np.mean(librosa.feature.rms(y=midi_y))
    ref_rms = np.mean(librosa.feature.rms(y=ref_y))
    dynamics_diff = abs(midi_rms - ref_rms)

    return [
        f"Rhythm difference (onset count): {rhythm_diff}",
        f"Intonation deviation (Hz avg): {pitch_diff:.2f}",
        f"Dynamics difference (RMS): {dynamics_diff:.4f}"
    ]

def find_soundfont() -> str:
    for name in os.listdir():
        if name.lower().endswith(".sf2") and os.path.isfile(name):
            return os.path.abspath(name)
    raise FileNotFoundError("❌ SoundFont (.sf2) file not found in working directory.")

def midi_to_wav_with_fluidsynth(midi_path: str, sf2_path: str, output_wav: str = "file.wav") -> str:
    midi_path = os.path.abspath(midi_path)
    sf2_path = os.path.abspath(sf2_path)
    output_wav = os.path.abspath(output_wav)

    print(f"🎵 Converting MIDI to WAV using FluidSynth:\nMIDI: {midi_path}\nsf2: {sf2_path}\nOUT: {output_wav}")

    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"❌ MIDI file not found: {midi_path}")
    if not os.path.exists(sf2_path):
        raise FileNotFoundError(f"❌ SoundFont file not found: {sf2_path}")

    try:
        subprocess.run([
            "fluidsynth",
            "-F", output_wav,
            "-r", "44100",
            "-ni",
            sf2_path,
            midi_path
        ], check=True, timeout=30)
    except subprocess.CalledProcessError as e:
        print("❌ FluidSynth failed:", e)
    except subprocess.TimeoutExpired:
        print("⏱️ FluidSynth took too long and was killed.")

    return output_wav

# === Main Pipeline ===
def main():
    print("📥 Uploading and detecting file types...")

    input_file = next((f for f in os.listdir() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.mp4', '.avi'))), None)
    xml_file = next((f for f in os.listdir() if f.endswith('.mxl')), None)
    ref_audio = next((f for f in os.listdir() if f.endswith('.wav') and "video_audio" not in f and "file" not in f), None)
    midi_path = next((f for f in os.listdir() if f.endswith('.mid')), None)
    sf2_path = find_soundfont()

    if not all([input_file, ref_audio, midi_path]):
        print("❌ Missing required input files (.png/.mp4/.mxl/.mid/.wav)")
        return

    if input_file.lower().endswith(('.mp4', '.avi')):
        extract_audio_from_video(input_file)
    else:
        convert_pdf_or_image_to_musicxml(input_file)

    print("🎼 Converting MusicXML to MIDI and WAV...")
    synth_path = midi_to_wav_with_fluidsynth(midi_path, sf2_path)
    score_spec = audio_to_feature(synth_path)
    perf_spec = audio_to_feature(ref_audio)

    model = PolytuneModel()
    model.eval()

    with torch.no_grad():
        s = torch.tensor(score_spec.mean(axis=1), dtype=torch.float32).unsqueeze(0)
        p = torch.tensor(perf_spec.mean(axis=1), dtype=torch.float32).unsqueeze(0)
        result = model(s, p)
        predicted_tokens = torch.argmax(result.logits, dim=-1).squeeze().tolist()
        if not isinstance(predicted_tokens, list):
            predicted_tokens = [predicted_tokens]
        error_labels = ['correct', 'missed', 'correct', 'extra', 'correct', 'missed']

    score = token_ids_to_musicxml(predicted_tokens, error_labels)
    xml_path = export_musicxml(score)
    print(f"✅ Annotated MusicXML saved to: {xml_path}")

    audio_metrics = compare_audio_metrics(synth_path, ref_audio)
    print("\n📊 Audio Comparison Metrics:")
    for m in audio_metrics:
        print(f"- {m}")

    print("\n🤖 Local DeepSeek Feedback:")
    print(get_llm_feedback(audio_metrics))

if __name__ == "__main__":
    main()
