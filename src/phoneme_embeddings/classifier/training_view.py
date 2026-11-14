"""Streamlit controls for the classifier training pipelines."""

from pathlib import Path
import pickle
import zipfile

import streamlit as st


def render(local_path):
    st.title("Обучение модели")
    try:
        from .pipelines import CLASS_MAP, TrainingConfig, run_training
    except ImportError as error:
        st.error(f"Для обучения нужен PyTorch. {error}")
        return

    names = {"triplet": "MLP · начало, центр и конец кадра",
             "center": "MLP · центральная фонема",
             "ctc": "BiLSTM + CTC · последовательность фонем"}
    pipeline = st.selectbox("Пайплайн", names, format_func=names.get)
    data = Path(__file__).resolve().parent / "data"
    sequence = pipeline == "ctc"
    if sequence:
        st.caption("Укажи две отдельные папки с парами *_no_av_embs.npy и *_no_av_ph.txt.")
    else:
        st.caption("Для валидации выбери отдельный набор кадров. Локальный test можно использовать для пробного запуска.")
    with st.form("classifier_training"):
        vocabulary = st.selectbox("Словарь фонем", list(CLASS_MAP),
                                  format_func=lambda key: f"{key} · {len(CLASS_MAP[key])} классов")
        train = st.text_input("Обучающие данные", "" if sequence else str(data / "train_embeddings.pt"),
                              key=f"training_train_{sequence}")
        validation = st.text_input("Данные валидации", "" if sequence else str(data / "test_embeddings.pt"),
                                   key=f"training_validation_{sequence}")
        output = st.text_input("Папка результатов", "reports/classifier_training")
        first, second = st.columns(2)
        epochs = first.number_input("Эпохи", 1, 200, 1)
        batch = second.number_input("Размер батча", 1, 1024, 32)
        optimizer = first.selectbox("Оптимизатор", ["AdamW", "Adam"])
        schedule = second.selectbox("Расписание скорости обучения", ["cosine", "step", "constant"])
        learning_rate = first.number_input("Скорость обучения", 0.000001, 1.0, 0.001, format="%.6f")
        dropout = second.number_input("Dropout", 0.0, 0.9, 0.1)
        limit = st.number_input("Лимит примеров на набор (0 — весь набор)", min_value=0, value=0)
        submitted = st.form_submit_button("Запустить обучение", type="primary")

    if submitted:
        st.session_state.pop("classifier_training_result", None)
        if not all(value.strip() for value in (train, validation, output)):
            st.error("Укажи источники данных и папку результатов.")
            return
        config = TrainingConfig(pipeline, vocabulary, epochs=epochs, batch_size=batch,
                                learning_rate=learning_rate, dropout=dropout, optimizer=optimizer,
                                schedule=schedule, limit=limit)
        progress = st.progress(0, text="Загружаю данные")
        try:
            directory, history = run_training(
                config, local_path(train), local_path(validation), local_path(output),
                on_epoch=lambda rows: progress.progress(len(rows) / epochs,
                                                       text=f"Эпоха {len(rows)} из {epochs}"))
        except (OSError, ValueError, RuntimeError, EOFError, TypeError, KeyError,
                pickle.UnpicklingError, zipfile.BadZipFile) as error:
            st.error(f"Обучение остановлено. {error}")
            return
        st.session_state["classifier_training_result"] = directory, history

    result = st.session_state.get("classifier_training_result")
    if result:
        directory, history = result
        st.success("Обучение завершено.")
        st.code(str(directory / "best.pth"), language=None)
        st.line_chart(history, x="epoch", y=["train_loss", "validation_loss"])
        st.dataframe(history, hide_index=True)
