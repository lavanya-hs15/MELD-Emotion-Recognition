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

