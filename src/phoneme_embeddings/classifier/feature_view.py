"""Phonetic feature inference in Streamlit."""

import json
from pathlib import Path
import pickle
import wave

import numpy as np
import streamlit as st


def render(root, local_path):
    st.title("Фонетические признаки")
    try:
        import altair as alt
        from .feature_inference import EXTRA_FEATURES, infer
    except ImportError as error:
        st.error(str(error))
        return

    directory = Path(__file__).resolve().parent
    source = st.radio("Источник", ["Эмбеддинги .npy", "Аудио WAV"], horizontal=True)
    wav = source == "Аудио WAV"
    model = st.selectbox("Модель", ["ResistantBias_2class", "ResistantBias_3class"])
    with st.form("feature_inference_form"):
        path = st.text_input(source, "data/examples/ata1101.wav" if wav else
                             "data/examples/ata1101_no_av_embs.npy", key=f"feature_source_{wav}")
        checkpoint = st.text_input("Путь к checkpoint", str(directory / "weights" / model),
                                   key=f"feature_weights_{model}")
        table = st.text_input("Таблица IPA (.csv)", str(directory / "ipa_all.csv"))
        with st.expander("Настройки поиска IPA"):
            extra = st.text_input("Дополнительные признаки", EXTRA_FEATURES)
            overrides = st.text_input("Переопределения")
        submit = st.form_submit_button("Распознать", type="primary")
    if submit:
        st.session_state.pop("feature_inference", None)
        try:
            with st.spinner("Предсказываю признаки и подбираю IPA…"):
                audio_path = local_path(path) if wav else None
                if wav:
                    from phoneme_embeddings.alignment import normalize
                    from phoneme_embeddings.audio import Audio, read_wav
                    from phoneme_embeddings.encoders.registry import create_encoder

                    audio = read_wav(audio_path)
                    mono = Audio(audio.samples.mean(axis=1, keepdims=True), audio.sample_rate)
                    options = json.loads((root / "configs/hubert_soft.json").read_text())["encoder"]
                    options["checkpoint"] = str(local_path(options["checkpoint"]))
                    vectors = normalize(create_encoder(**options).encode(mono).vectors, "l2")
                else:
                    vectors = np.load(local_path(path), allow_pickle=False)
                features, lookup, matches = infer(local_path(checkpoint), vectors, local_path(table),
                                                  extra, overrides)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError,
                EOFError, wave.Error, pickle.UnpicklingError) as error:
            st.error(f"Инференс остановлен. {error}")
            return
        st.session_state["feature_inference"] = model, source, audio_path, features, lookup, matches

    result = st.session_state.get("feature_inference")
    if not result or result[:2] != (model, source):
        return
    _, _, audio_path, features, lookup, matches = result
    if audio_path:
        st.audio(str(audio_path))
    st.text_area("Покадровая IPA-транскрипция", "_".join(matches["IPA"]), height=100)
    frames = features.rename_axis("Кадр").reset_index().melt("Кадр", var_name="Признак", value_name="Значение")
    heatmap = alt.Chart(frames).mark_rect().encode(
        x="Кадр:O", y=alt.Y("Признак:N", sort=list(features.columns)),
        color=alt.Color("Значение:O", scale=alt.Scale(domain=[-1, 0, 1],
                                                       range=["#3666a6", "#eeeeee", "#ab3c47"])),
        tooltip=["Кадр", "Признак", "Значение"])
    st.altair_chart(heatmap, width="stretch")
    st.line_chart(matches, x="Время, с", y=["Расстояние", "Число кандидатов"])
    frame = int(st.number_input("Кадр", 0, len(matches) - 1, 0, key=f"feature_frame_{len(matches)}"))
    st.write(matches.loc[frame, ["IPA", "Расстояние", "Кандидаты"]].to_dict())
    st.dataframe(features.iloc[frame].rename("Предсказано").to_frame()
                 .join(lookup.iloc[frame].rename("Для поиска"), how="outer"))
    with st.expander("Признаки и кандидаты по всем кадрам"):
        st.dataframe(matches.join(features))
    st.download_button("Скачать кадры (CSV)", matches.join(features).to_csv(index=False),
                       "phonetic_features.csv", "text/csv")
