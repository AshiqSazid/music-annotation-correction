# music_comparator.py

import os
import json
import re
import librosa
import numpy as np
import music21 as m21
import requests
import streamlit as st
from typing import Tuple

def ensure_audio(input_path: str) -> Tuple[np.ndarray, int, m21.stream.Score]:
    ext = input_path.lower().split('.')[-1]
    if ext in ['wav', 'mp3', 'ogg', 'flac']:
        y, sr = librosa.load(input_path, sr=None)
        return y, sr, None
    elif ext in ['mid', 'midi', 'xml', 'mxl', 'musicxml']:
        score = m21.converter.parse(input_path)
        midi_fp = score.write('midi')
        y, sr = librosa.load(midi_fp, sr=None)
        return y, sr, score
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def extract_audio_features(ref_y, ref_sr, stu_y, stu_sr):
    features = {}
    f0_ref, vrf, _ = librosa.pyin(ref_y, sr=ref_sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_stu, vst, _ = librosa.pyin(stu_y, sr=stu_sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    ref_midi = librosa.hz_to_midi(f0_ref)
    stu_midi = librosa.hz_to_midi(f0_stu)
    voiced_idx = (vrf == True) & (vst == True)
    if voiced_idx.any():
        pitch_diff = stu_midi[voiced_idx] - ref_midi[voiced_idx]
        features['mean_pitch_error_semitones'] = float(np.nanmean(np.abs(pitch_diff)))
    else:
        features['mean_pitch_error_semitones'] = None

    tempo_ref, _ = librosa.beat.beat_track(y=ref_y, sr=ref_sr)
    tempo_stu, _ = librosa.beat.beat_track(y=stu_y, sr=stu_sr)
    features['tempo_ref_bpm'] = float(tempo_ref)
    features['tempo_student_bpm'] = float(tempo_stu)
    features['tempo_difference'] = float(abs(tempo_stu - tempo_ref))

    onset_ref = librosa.onset.onset_detect(y=ref_y, sr=ref_sr, units='time')
    onset_stu = librosa.onset.onset_detect(y=stu_y, sr=stu_sr, units='time')
    if len(onset_ref) > 0 and len(onset_stu) > 0:
        min_onsets = min(len(onset_ref), len(onset_stu))
        timing_diffs = [abs(onset_stu[i] - onset_ref[i]) for i in range(min_onsets)]
        features['avg_onset_deviation_sec'] = float(np.mean(timing_diffs))
    else:
        features['avg_onset_deviation_sec'] = None

    rms_ref = librosa.feature.rms(y=ref_y)[0]
    rms_stu = librosa.feature.rms(y=stu_y)[0]
    features['mean_rms_ref'] = float(np.mean(rms_ref))
    features['mean_rms_student'] = float(np.mean(rms_stu))
    features['mean_rms_difference'] = float(abs(np.mean(rms_stu) - np.mean(rms_ref)))
    features['dynamic_range_ref'] = float(np.max(rms_ref) - np.min(rms_ref))
    features['dynamic_range_student'] = float(np.max(rms_stu) - np.min(rms_stu))
    features['dynamic_range_difference'] = float(abs(features['dynamic_range_student'] - features['dynamic_range_ref']))
    return features

def compare_symbolic(score_ref, score_stu):
    comparison = {'missed_notes': [], 'extra_notes': [], 'pitch_mismatches': []}
    if score_ref is None or score_stu is None:
        return comparison

    ref_notes = [n for n in score_ref.recurse().notes]
    stu_notes = [n for n in score_stu.recurse().notes]
    i, j = 0, 0
    while i < len(ref_notes) and j < len(stu_notes):
        r = ref_notes[i]
        s = stu_notes[j]
        if r.nameWithOctave == s.nameWithOctave:
            i += 1
            j += 1
        else:
            comparison['pitch_mismatches'].append({
                'expected': r.nameWithOctave,
                'heard': s.nameWithOctave,
                'ref_measure': r.measureNumber
            })
            i += 1
            j += 1
    while i < len(ref_notes):
        r = ref_notes[i]
        comparison['missed_notes'].append({'expected': r.nameWithOctave, 'ref_measure': r.measureNumber})
        i += 1
    while j < len(stu_notes):
        s = stu_notes[j]
        comparison['extra_notes'].append({'heard': s.nameWithOctave, 'time': float(s.offset)})
        j += 1
    return comparison

def generate_feedback_and_grade(audio_metrics, symbolic_diff):
    prompt_parts = [
        "You are a music teacher evaluating a student's performance.",
        f"- Pitch error: {audio_metrics['mean_pitch_error_semitones']} semitones\n",
        f"- Tempo deviation: {audio_metrics['tempo_difference']} BPM\n",
        f"- Onset deviation: {audio_metrics['avg_onset_deviation_sec']} seconds\n",
        f"- RMS diff: {audio_metrics['mean_rms_difference']}\n",
        f"- Dynamic range diff: {audio_metrics['dynamic_range_difference']}\n",
    ]
    if symbolic_diff['missed_notes']:
        prompt_parts.append(f"Missed notes: {len(symbolic_diff['missed_notes'])}\n")
    if symbolic_diff['extra_notes']:
        prompt_parts.append(f"Extra notes: {len(symbolic_diff['extra_notes'])}\n")
    if symbolic_diff['pitch_mismatches']:
        prompt_parts.append(f"Pitch mismatches: {len(symbolic_diff['pitch_mismatches'])}\n")

    prompt_parts.append("Please provide feedback on intonation, rhythm, dynamics, and give a grade out of 100.")
    prompt = "\n".join(prompt_parts)

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "deepseek-r1:8b",
            "prompt": prompt,
            "stream": False
        })
        data = response.json()
        feedback = data.get("response", "No feedback received.")
        match = re.search(r'(\d{1,3})\s*/\s*100', feedback)
        grade = int(match.group(1)) if match else None
        return feedback, grade
    except Exception as e:
        return f"Error calling LLM: {e}", None

def save_results(path, audio_metrics, symbolic, feedback, grade):
    result = {
        "audio_metrics": audio_metrics,
        "symbolic_comparison": symbolic,
        "feedback": feedback,
        "grade": grade
    }
    with open(path, 'w') as f:
        json.dump(result, f, indent=4)
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("student_file")
    parser.add_argument("reference_file")
    parser.add_argument("--output", default="comparison_result.json")
    args = parser.parse_args()

    stu_y, stu_sr, stu_score = ensure_audio(args.student_file)
    ref_y, ref_sr, ref_score = ensure_audio(args.reference_file)

    audio_metrics = extract_audio_features(ref_y, ref_sr, stu_y, stu_sr)
    symbolic_diff = compare_symbolic(ref_score, stu_score)
    feedback, grade = generate_feedback_and_grade(audio_metrics, symbolic_diff)
    result = save_results(args.output, audio_metrics, symbolic_diff, feedback, grade)
    print(f"✅ Analysis saved to {args.output}")

# Streamlit GUI
if st._is_running_with_streamlit:
    st.title("Music Performance Comparator 🎶")
    uploaded_files = st.file_uploader("Upload the student performance and reference audio/score files",
                                      type=["wav", "mp3", "mid", "midi", "xml", "mxl"],
                                      accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) == 2:
        with open("temp_student", "wb") as f:
            f.write(uploaded_files[0].read())
        with open("temp_reference", "wb") as f:
            f.write(uploaded_files[1].read())

        stu_y, stu_sr, stu_score = ensure_audio("temp_student")
        ref_y, ref_sr, ref_score = ensure_audio("temp_reference")

        audio_metrics = extract_audio_features(ref_y, ref_sr, stu_y, stu_sr)
        symbolic_diff = compare_symbolic(ref_score, stu_score)
        feedback, grade = generate_feedback_and_grade(audio_metrics, symbolic_diff)
        result = save_results("comparison_result.json", audio_metrics, symbolic_diff, feedback, grade)

        st.success("✅ Analysis Complete!")
        st.json(result)
        st.download_button("Download JSON results", data=json.dumps(result, indent=4), file_name="comparison_result.json")
