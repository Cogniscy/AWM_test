"""Streamlit section for testing the thesis phoneme classifier."""

import csv
import io
import json
from pathlib import Path
import pickle
import zipfile

import pandas as pd
import streamlit as st


def render(local_path):
    st.title("Тестирование модели")
    try:
        from . import CLASS_MAP, evaluate
    except ImportError as error:
        st.error(f"Для тестирования нужен PyTorch. {error}")
        return

    st.caption("Покадровая оценка центральной фонемы на размеченных эмбеддингах HuBERT-Soft.")
    model_id = st.selectbox("Модель", list(CLASS_MAP),
                            format_func=lambda uid: f"{uid} · {len(CLASS_MAP[uid])} классов")
    directory = Path(__file__).resolve().parent
    default_checkpoint = str(directory / "weights" / f"{model_id}_model.pth")
    local_test = directory / "data" / "test_embeddings.pt"
    with st.form("classifier_test"):
        checkpoint_text = st.text_input("Путь к checkpoint", default_checkpoint,
                                        key=f"classifier_checkpoint_{model_id}")
        test_text = st.text_input("Путь к тестовым эмбеддингам (.pt)",
                                  str(local_test) if local_test.is_file() else "")
        submitted = st.form_submit_button("Запустить тестирование", type="primary")

    if submitted:
        st.session_state.pop("classifier_result", None)
        checkpoint, test_path = local_path(checkpoint_text), local_path(test_text)
        if not checkpoint.is_file() or not test_path.is_file():
            st.error("Укажи существующие файлы checkpoint и тестовых эмбеддингов.")
            return
        try:
            with st.spinner("Загружаю эмбеддинги и считаю метрики…"):
                st.session_state["classifier_result"] = evaluate(model_id, checkpoint, test_path)
        except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError,
                zipfile.BadZipFile) as error:
            st.error(f"Не удалось выполнить оценку. {error}")
            return

    result = st.session_state.get("classifier_result")
    if result is None or result["model"] != model_id:
        return
    st.caption(f"Тестовый файл · {result['test_embeddings']}")
    for column, key, title in zip(st.columns(3), ("accuracy", "balanced_accuracy", "macro_f1"),
                                  ("Accuracy", "Balanced accuracy", "Macro-F1")):
        column.metric(title, f"{result[key]:.2%}")
    st.write(f"Оценено кадров — {result['frames']:,}. "
             f"Метки вне словаря — {result['excluded_frames']:,}.")
    labels, matrix = result["labels"], result["confusion_matrix"]
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Истинная / предсказанная", *labels])
    writer.writerows([label, *counts] for label, counts in zip(labels, matrix))
    with st.expander("Матрица ошибок"):
        st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels))
    first, second = st.columns(2)
    first.download_button("Скачать метрики (JSON)", json.dumps(result, ensure_ascii=False, indent=2),
                          f"{model_id}_metrics.json", "application/json")
    second.download_button("Скачать матрицу (CSV)", csv_buffer.getvalue(),
                           f"{model_id}_confusion.csv", "text/csv")
