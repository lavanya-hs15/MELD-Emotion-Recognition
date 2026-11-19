from pathlib import Path
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import soundfile as sf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import imageio_ffmpeg
from huggingface_hub import hf_hub_download

from faster_whisper import WhisperModel
from transformers import (
    AutoTokenizer, AutoModel,
    Wav2Vec2FeatureExtractor, HubertModel
)
from torchvision import models

st.set_page_config(page_title="MELD Video Emotion Recognition", page_icon="🎭", layout="wide")
st.title("MELD — Video Emotion Recognition")
st.write("Upload a video to view its transcript, audio, sampled frames, and segment-level emotion predictions.")

# Download the trained checkpoint from the private Hugging Face model repository.
# Add HF_TOKEN to Streamlit Community Cloud -> App -> Settings -> Secrets.
MODEL_REPO_ID = "Lavanya-hs15/meld-emotion-checkpoint"
CHECKPOINT_FILENAME = "cross_modal_temporal_transformer_best.pt"

@st.cache_resource(show_spinner="Downloading trained MELD checkpoint...")
def get_checkpoint_path():
    token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add your Hugging Face read token in "
            "Streamlit Community Cloud: App -> Settings -> Secrets."
        )
    return Path(hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=CHECKPOINT_FILENAME,
        token=token
    ))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossModalTemporalTransformer(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, num_classes,
                 hidden_dim=128, num_heads=4, temporal_layers=2, dropout=0.25):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.video_proj = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        modality_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.cross_modal_encoder = nn.TransformerEncoder(modality_layer, num_layers=1)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text, audio, video, padding_mask):
        text_tokens = self.text_proj(text)
        audio_tokens = self.audio_proj(audio)
        video_tokens = self.video_proj(video)
        batch_size, sequence_length, hidden_dim = text_tokens.shape
        modality_tokens = torch.stack([text_tokens, audio_tokens, video_tokens], dim=2)
        modality_tokens = modality_tokens.reshape(batch_size * sequence_length, 3, hidden_dim)
        attended = self.cross_modal_encoder(modality_tokens)
        fused = attended.mean(dim=1).reshape(batch_size, sequence_length, hidden_dim)
        contextual = self.temporal_encoder(fused, src_key_padding_mask=padding_mask)
        return self.classifier(contextual)


@st.cache_resource(show_spinner="Loading trained MELD model and feature extractors...")
def load_models(checkpoint_path_str, device_str):
    device = torch.device(device_str)
    checkpoint = torch.load(checkpoint_path_str, map_location=device, weights_only=False)
    label_to_id = checkpoint["label_to_id"]
    id_to_label = {int(v): str(k) for k, v in label_to_id.items()}

    model = CrossModalTemporalTransformer(
        checkpoint["text_dim"], checkpoint["audio_dim"], checkpoint["video_dim"],
        int(checkpoint["num_classes"]), **checkpoint["config"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta = AutoModel.from_pretrained("roberta-base").to(device).eval()
    for p in roberta.parameters():
        p.requires_grad = False

    hubert_name = "facebook/hubert-base-ls960"
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_name)
    hubert = HubertModel.from_pretrained(hubert_name).to(device).eval()
    for p in hubert.parameters():
        p.requires_grad = False

    resnet_weights = models.ResNet50_Weights.DEFAULT
    resnet_model = models.resnet50(weights=resnet_weights)
    resnet_encoder = nn.Sequential(*list(resnet_model.children())[:-1]).to(device).eval()
    for p in resnet_encoder.parameters():
        p.requires_grad = False
    return checkpoint, label_to_id, id_to_label, model, tokenizer, roberta, hubert_processor, hubert, resnet_encoder, resnet_weights.transforms()


def extract_audio(video_path, audio_path):
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wave, sample_rate = sf.read(audio_path)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16000 Hz audio after ffmpeg; received {sample_rate} Hz.")
    return wave.astype(np.float32), sample_rate


def get_segment_frames(video_path, start_s, end_s, max_frames=12):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("Could not read video frame rate or frame count.")
    start_idx = max(0, int(start_s * fps))
    end_idx = min(total_frames - 1, max(start_idx, int(end_s * fps)))
    indices = np.linspace(start_idx, end_idx, min(max_frames, end_idx - start_idx + 1), dtype=int)
    frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"No readable frames at {start_s:.2f}-{end_s:.2f}s.")
    return frames


def run_prediction(video_path, checkpoint, id_to_label, model, tokenizer, roberta,
                   hubert_processor, hubert, resnet_encoder, video_transform, device):
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "audio_16k_mono.wav"
        wave, sample_rate = extract_audio(video_path, audio_path)

        whisper_device = "cuda" if device.type == "cuda" else "cpu"
        whisper_compute = "float16" if whisper_device == "cuda" else "int8"
        whisper = WhisperModel("small", device=whisper_device, compute_type=whisper_compute)
        segments_iter, info = whisper.transcribe(str(video_path), beam_size=5, vad_filter=True)
        segments = [
            {"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
            for s in segments_iter
            if (s.text or "").strip() and float(s.end) > float(s.start)
        ]
        if not segments:
            raise ValueError("Whisper did not detect any speech segments in this video.")

        text_features, audio_features, video_features, valid_segments = [], [], [], []
        progress = st.progress(0, text="Extracting features for speech segments...")
        status = st.empty()

        for idx, row in enumerate(segments):
            start_s = max(0.0, row["start"])
            end_s = min(len(wave) / sample_rate, row["end"])
            if end_s <= start_s:
                continue
            segment_wave = wave[int(start_s * sample_rate):int(end_s * sample_rate)]
            if len(segment_wave) == 0:
                continue

            # RoBERTa mean-pooling, matching the notebook feature extraction.
            with torch.inference_mode():
                batch = tokenizer([row["text"]], padding=True, truncation=True,
                                  max_length=128, return_tensors="pt")
                batch = {k: v.to(device) for k, v in batch.items()}
                hidden = roberta(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                tx = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).squeeze(0).cpu().numpy().astype(np.float32)

                if len(segment_wave) < int(0.10 * sample_rate):
                    segment_wave = np.pad(segment_wave, (0, int(0.10 * sample_rate) - len(segment_wave)))
                inputs = hubert_processor(segment_wave.astype(np.float32), sampling_rate=16000,
                                          return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                ahidden = hubert(**inputs).last_hidden_state
                ax = ahidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

            frames = get_segment_frames(video_path, start_s, end_s, max_frames=12)
            with torch.inference_mode():
                frame_batch = torch.stack([video_transform(frame) for frame in frames]).to(device)
                vx = resnet_encoder(frame_batch).flatten(1).mean(dim=0).cpu().numpy().astype(np.float32)

            text_features.append(tx)
            audio_features.append(ax)
            video_features.append(vx)
            valid_segments.append({**row, "start": start_s, "end": end_s, "frames_used": len(frames), "preview_frames": frames})
            status.write(f"Processed segment {idx + 1}/{len(segments)}")
            progress.progress((idx + 1) / len(segments))

        if not valid_segments:
            raise RuntimeError("No speech segments could be converted into all three modality features.")

        X_text = torch.from_numpy(np.stack(text_features).astype(np.float32)).unsqueeze(0).to(device)
        X_audio = torch.from_numpy(np.stack(audio_features).astype(np.float32)).unsqueeze(0).to(device)
        X_video = torch.from_numpy(np.stack(video_features).astype(np.float32)).unsqueeze(0).to(device)
        expected = (checkpoint["text_dim"], checkpoint["audio_dim"], checkpoint["video_dim"])
        actual = (X_text.shape[-1], X_audio.shape[-1], X_video.shape[-1])
        if expected != actual:
            raise ValueError(f"Feature dimensions mismatch. Model expects {expected}; extracted {actual}.")

        padding_mask = torch.zeros((1, len(valid_segments)), dtype=torch.bool, device=device)
        with torch.inference_mode():
            logits = model(X_text, X_audio, X_video, padding_mask)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_ids = probabilities.argmax(axis=1)
        results = pd.DataFrame([
            {k: v for k, v in seg.items() if k != "preview_frames"} for seg in valid_segments
        ])
        results["predicted_emotion"] = [id_to_label[int(i)] for i in pred_ids]
        results["confidence"] = probabilities.max(axis=1)
        results["duration_seconds"] = results["end"] - results["start"]
        for class_id, emotion in id_to_label.items():
            results[f"prob_{emotion}"] = probabilities[:, int(class_id)]

        return wave, sample_rate, info, results, valid_segments, probabilities


try:
    CHECKPOINT_PATH = get_checkpoint_path()
except Exception as exc:
    st.error(f"Could not obtain the trained MELD checkpoint: {exc}")
    st.stop()

st.caption(f"Device: {DEVICE}" + (f" · {torch.cuda.get_device_name(0)}" if DEVICE.type == "cuda" else " · CPU"))
uploaded = st.file_uploader("Upload a video", type=["mp4", "mkv", "mov", "avi", "webm"])

if uploaded is not None:
    if st.button("Analyze video", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as f:
            f.write(uploaded.getbuffer())
            video_temp_path = Path(f.name)
        try:
            loaded = load_models(str(CHECKPOINT_PATH), str(DEVICE))
            checkpoint, label_to_id, id_to_label, model, tokenizer, roberta, hubert_processor, hubert, resnet_encoder, video_transform = loaded

            with st.spinner("Transcribing speech and predicting emotions..."):
                wave, sample_rate, info, results, valid_segments, probabilities = run_prediction(
                    video_temp_path, checkpoint, id_to_label, model, tokenizer, roberta,
                    hubert_processor, hubert, resnet_encoder, video_transform, DEVICE
                )

            st.success(f"Analysis complete · Detected language: {info.language} · Speech segments: {len(results)}")
            st.subheader("1. Extracted audio")
            st.audio(wave, sample_rate=sample_rate)

            st.subheader("2. Recognized speech transcript")
            transcript_view = results[["start", "end", "text"]].copy()
            st.dataframe(transcript_view, use_container_width=True, hide_index=True)

            st.subheader("3. Sampled video frames")
            for i, seg in enumerate(valid_segments):
                with st.expander(f"Segment {i+1}: {seg['start']:.2f}s–{seg['end']:.2f}s · {seg['text']}", expanded=(i < 2)):
                    cols = st.columns(min(4, len(seg["preview_frames"])))
                    for j, frame in enumerate(seg["preview_frames"]):
                        with cols[j % len(cols)]:
                            st.image(frame, use_container_width=True)

            st.subheader("4. Final emotion predictions")
            display_df = results[["start", "end", "text", "predicted_emotion", "confidence"]].copy()
            display_df["start"] = display_df["start"].map(lambda x: f"{x:.2f}s")
            display_df["end"] = display_df["end"].map(lambda x: f"{x:.2f}s")
            display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.subheader("5. Emotion timeline")
            emotions = list(dict.fromkeys(results["predicted_emotion"].tolist()))
            emotion_to_y = {emotion: i for i, emotion in enumerate(emotions)}
            fig, ax = plt.subplots(figsize=(14, max(3, len(emotions) * 0.55)))
            for _, row in results.iterrows():
                y = emotion_to_y[row["predicted_emotion"]]
                ax.plot([row["start"], row["end"]], [y, y], linewidth=9, solid_capstyle="butt")
                ax.text((row["start"] + row["end"]) / 2, y + 0.12, row["predicted_emotion"],
                        ha="center", va="bottom", fontsize=8)
            ax.set_yticks(range(len(emotions)), emotions)
            ax.set_xlabel("Video time (seconds)")
            ax.set_ylabel("Predicted emotion")
            ax.set_title("Segment-level emotion timeline")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("6. Download results")
            st.download_button(
                "Download predictions CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="external_video_emotion_predictions.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download transcript CSV",
                data=results[["start", "end", "text"]].to_csv(index=False).encode("utf-8"),
                file_name="transcript_segments.csv",
                mime="text/csv"
            )
            st.info("Predictions are experimental segment-level estimates. They are not verified ground-truth emotions and do not label silent portions of the video.")
        except subprocess.CalledProcessError:
            st.error("FFmpeg could not extract audio. Check that FFmpeg is installed and that the uploaded video is valid.")
        except Exception as exc:
            st.exception(exc)
        finally:
            try:
                video_temp_path.unlink(missing_ok=True)
            except Exception:
                pass
