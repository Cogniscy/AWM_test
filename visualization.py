"""Browse saved embeddings together with their source audio and annotation."""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from phoneme_embeddings.audio import read_wav
from phoneme_embeddings.segmentation import read_seg_b2


@st.cache_data(ttl=30, show_spinner="Ищу обработанные записи")
def result_index(root: str):
    path = Path(root)
    return sorted(str(p.parent.relative_to(path)) for p in path.rglob("run.json"))


def timeline(audio, segments, selected):
    # Preserve extrema when reducing the waveform for display.
    signal = audio.samples[:, 0]
    block = max(1, int(np.ceil(len(signal) / 2000)))
    usable = len(signal) // block * block
    bins = signal[:usable].reshape(-1, block)
    times = np.arange(len(bins)) * block / audio.sample_rate
    wave = pd.DataFrame({"time": times, "low": bins.min(axis=1), "high": bins.max(axis=1)})
    waveform = alt.Chart(wave).mark_rule(color="#4b83c4").encode(
        x=alt.X("time:Q", title="Время, с"), y=alt.Y("low:Q", title="Амплитуда"),
        y2="high:Q",
    ).properties(height=130, title="Осциллограмма")

    nfft = min(512, len(signal))
    if nfft < 2:
        raise ValueError("Аудио слишком короткое для спектрограммы")
    hop = max(nfft // 4, int(np.ceil(len(signal) / 500)))
    windows = np.lib.stride_tricks.sliding_window_view(signal, nfft)[::hop]
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(nfft), axis=1)) ** 2
    db = 10 * np.log10(np.maximum(spectrum, 1e-12))
    db = np.maximum(db, db.max() - 80)
    hz = np.fft.rfftfreq(nfft, 1 / audio.sample_rate)
    # Plot two-bin frequency groups to keep long utterances responsive.
    indices = np.arange(0, len(hz) - 1, 2)
    centers = (np.arange(len(windows)) * hop + nfft / 2) / audio.sample_rate
    spec = pd.DataFrame({
        "time": np.repeat(centers, len(indices)),
        "end": np.repeat(centers + hop / audio.sample_rate, len(indices)),
        "hz": np.tile(hz[indices], len(windows)),
        "hz_end": np.tile(hz[np.minimum(indices + 2, len(hz) - 1)], len(windows)),
        "db": db[:, indices].reshape(-1),
    })
    spectrogram = alt.Chart(spec).mark_rect().encode(
        x=alt.X("time:Q", title="Время, с"), x2="end:Q",
        y=alt.Y("hz:Q", title="Частота, Гц"), y2="hz_end:Q",
        color=alt.Color("db:Q", title="дБ", scale=alt.Scale(scheme="magma")),
        tooltip=[alt.Tooltip("time:Q", format=".3f"), "hz:Q", "db:Q"],
    ).properties(height=240, title="Спектрограмма")

    labels = pd.DataFrame([{"start": s.start, "end": s.end, "label": s.label,
                            "center": (s.start + s.end) / 2, "lane": "Фонемы"}
                           for s in segments])
    labels_chart = alt.Chart(labels).mark_rect(opacity=.35).encode(
        x=alt.X("start:Q", title="Время, с"), x2="end:Q",
        y=alt.Y("lane:N", title=None),
        color=alt.Color("label:N", legend=None),
        tooltip=["label:N", alt.Tooltip("start:Q", format=".3f"),
                 alt.Tooltip("end:Q", format=".3f")],
    )
    text = alt.Chart(labels).mark_text(fontSize=11).encode(
        x="center:Q", y="lane:N", text="label:N",
    )
    annotation = (labels_chart + text).properties(height=50, title="Фонемная разметка")
    selection = pd.DataFrame([{"start": selected["start_sec"], "end": selected["end_sec"]}])
    layers = []
    for chart in (waveform, spectrogram, annotation):
        highlight = alt.Chart(selection).mark_rect(color="#00c9b1", opacity=.18).encode(
            x="start:Q", x2="end:Q",
        )
        layers.append(chart + highlight)
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    return alt.vconcat(layers[0].add_params(zoom), *layers[1:]).resolve_scale(x="shared", color="independent")


def vector_distances(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("Для сравнения нужны векторы одинаковой размерности.")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Векторы содержат нечисловые значения.")
    norm = np.linalg.norm(first) * np.linalg.norm(second)
    return {
        "Косинусное сходство": float(np.clip(np.dot(first, second) / norm, -1, 1)) if norm else None,
        "L2-расстояние": float(np.linalg.norm(first - second)),
        "MSE": float(np.mean((first - second) ** 2)),
    }


def compare_vector(output, index, model, first_run, first, first_description, chosen):
    st.subheader("Сравнение двух векторов")
    if not st.checkbox("Выбрать второй вектор"):
        return
    st.caption("Вектор A — выбранный выше интервал. Вектор B выбирается ниже.")
    query = st.text_input("Поиск записи для вектора B", key="compare_query")
    matching = [name for name in index if Path(name).parts[0] == model
                and query.casefold() in name.casefold()]
    if not matching:
        st.info("Нет подходящих записей.")
        return
    if chosen in matching:
        matching.remove(chosen)
        matching.insert(0, chosen)
    if len(matching) > 500:
        st.caption("Первые 500 записей. Для других записей уточни поиск.")
    second_path = st.selectbox("Запись для вектора B", matching[:500])
    folder = output / second_path
    second_run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    # Equal dimensions alone do not establish a common embedding space.
    encoder_a = {k: v for k, v in first_run["encoder"].items() if k != "device"}
    encoder_b = {k: v for k, v in second_run["encoder"].items() if k != "device"}
    if encoder_a != encoder_b:
        st.warning("У записей разные веса или параметры энкодера. Выбери запись с тем же энкодером.")
        return
    modes = [m for m in ("frame", "segment_mean") if (folder / f"{m}.npy").is_file()]
    mode = st.selectbox("Представление вектора B", modes,
                        format_func=lambda m: "Кадры" if m == "frame" else "Фонемные сегменты")
    matrix = np.load(folder / f"{mode}.npy", mmap_mode="r", allow_pickle=False)
    rows = [json.loads(line) for line in (folder / f"{mode}.jsonl").read_text(encoding="utf-8").splitlines()]
    available = [i for i, row in enumerate(rows) if row["row"] is not None]
    if not available:
        st.info("В этой записи нет векторов выбранного типа.")
        return

    def describe(i):
        item = rows[i]
        label = item.get("label") or " / ".join(item.get("labels", []))
        return f"{i} · {label} · {item['start_sec']:.3f}–{item['end_sec']:.3f} с"

    selected = rows[st.selectbox("Кадр или сегмент B", available, format_func=describe)]
    second = np.asarray(matrix[selected["row"]])
    metrics = vector_distances(first, second)
    st.caption(f"A · {first_description}")
    st.caption(f"B · {second_path} · {selected['start_sec']:.3f}–{selected['end_sec']:.3f} с")
    for column, (name, value) in zip(st.columns(3), metrics.items()):
        column.metric(name, "не определено" if value is None else f"{value:.6g}")
    values = pd.DataFrame({"Координата": np.arange(len(first)),
                           "A": first, "B": second, "A − B": first - second})
    st.line_chart(values, x="Координата", y=["A", "B"])
    st.line_chart(values, x="Координата", y="A − B")
    with st.expander("Значения обоих векторов и разность"):
        st.dataframe(values, hide_index=True)
    st.download_button("Скачать сравнение CSV", values.to_csv(index=False).encode("utf-8"),
                       "vector_comparison.csv", "text/csv")


def render(root: Path):
    st.title("Визуализация")
    output = Path(st.text_input("Каталог результатов", str(root / "data/extracted"))).expanduser()
    if not output.is_absolute():
        output = root / output
    if st.button("Обновить список записей"):
        result_index.clear()
    if not output.is_dir():
        st.info("Укажи существующий каталог с результатами извлечения.")
        return
    index = result_index(str(output))
    if not index:
        st.info("Пока нет завершённых результатов. Сначала запусти обработку.")
        return
    models = sorted({Path(name).parts[0] for name in index})
    model = st.selectbox("Модель", models)
    query = st.text_input("Поиск по диктору, записи или дате прогона", placeholder="Например, Vladimir или ata1101")
    matching = [name for name in index if Path(name).parts[0] == model
                and query.casefold() in name.casefold()]
    st.caption(f"Найдено записей: {len(matching)}")
    if not matching:
        return
    if len(matching) > 500:
        st.info("Показаны первые 500 результатов. Уточни поиск для выбора других записей.")
    chosen = st.selectbox("Запись и прогон", matching[:500])
    folder = output / chosen
    try:
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        modes = [m for m in ("frame", "segment_mean") if (folder / f"{m}.npy").is_file()]
        mode = st.radio("Представление", modes, horizontal=True,
                        format_func=lambda m: {"frame": "Кадры", "segment_mean": "Фонемные сегменты"}[m])
        vectors = np.load(folder / f"{mode}.npy", mmap_mode="r", allow_pickle=False)
        metadata = [json.loads(line) for line in (folder / f"{mode}.jsonl").read_text(encoding="utf-8").splitlines()]
        st.caption(f"Аудио: {run['wav']} · размерность: {vectors.shape[1]}")
        options = list(range(len(metadata)))
        if not options:
            st.info("В этом результате нет интервалов для просмотра.")
            return

        def label(i):
            row = metadata[i]
            phoneme = row.get("label") or " / ".join(row.get("labels", []))
            suffix = " · нет целых кадров" if row["row"] is None else ""
            return f"{i} · {phoneme} · {row['start_sec']:.3f}–{row['end_sec']:.3f} с{suffix}"

        selected = metadata[st.selectbox("Кадр или сегмент", options, format_func=label)]
        wav = Path(run["wav"])
        if wav.is_file():
            audio = read_wav(wav)
            segments = read_seg_b2(Path(run["segmentation"]), audio.duration)
            st.audio(str(wav))
            chart = timeline(audio, segments, selected)
            # Large spectrograms use inline data beyond Altair's default 5000-row cap.
            with alt.data_transformers.enable("default", max_rows=None):
                st.altair_chart(chart, width="stretch")
            st.caption("Масштабируй и перемещай временную ось. Зелёным выделен выбранный кадр или сегмент.")
        else:
            st.warning(f"Исходное аудио не найдено: {wav}")

        if selected["row"] is None:
            st.info("Внутри этого сегмента нет целых кадров. Усреднённый вектор отсутствует.")
        else:
            vector = np.asarray(vectors[selected["row"]])
            values = pd.DataFrame({"Координата": np.arange(len(vector)), "Значение": vector})
            st.subheader("Вектор выбранного интервала")
            st.line_chart(values, x="Координата", y="Значение")
            st.caption(f"Размерность {len(vector)} · L2-норма {np.linalg.norm(vector):.6f}")
            with st.expander("Численные значения"):
                st.dataframe(values, hide_index=True)
            compare_vector(output, index, model, run, vector,
                           f"{chosen} · {selected['start_sec']:.3f}–{selected['end_sec']:.3f} с", chosen)
    except (OSError, ValueError, KeyError, IndexError) as error:
        st.error(f"Не удалось открыть результат: {error}")
