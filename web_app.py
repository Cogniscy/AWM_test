"""Local browser interface for phoneme embedding extraction."""

import json
import os
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import streamlit as st

from phoneme_embeddings.encoders.registry import create_encoder
from phoneme_embeddings.extract import extract
from phoneme_embeddings.resume import completed_pairs

ROOT = Path(__file__).resolve().parent
STAGES = {
    "read": "Читаю аудио и фонемную разметку",
    "load_model": "Подготавливаю модель",
    "encode": "Извлекаю эмбеддинги из аудио",
    "frame": "Сопоставляю кадры с разметкой и сохраняю",
    "segment_mean": "Усредняю кадры внутри фонемных сегментов",
    "metadata": "Сохраняю параметры прогона",
}
MODEL_LABELS = {
    "hubert_soft": "HuBERT-Soft",
    "whisper_ppg_base_en": "WhisperPPG Base.en",
    "whisper_ppg": "WhisperPPG Medium",
    "contentvec_256_l9": "ContentVec 256 L9",
    "contentvec_768_l12": "ContentVec 768 L12",
    "syllablelm": "SyllableLM",
}
MODE_LABELS = {"frame": "Покадровый", "segment_mean": "Среднее по фонемному сегменту"}


def local_path(value: str) -> Path:
    return (ROOT / Path(value.strip()).expanduser()).resolve()


def corpus_pairs(root: Path):
    wavs, annotations = {}, {}
    for file in sorted(root.rglob("*")):
        extension = file.suffix.lower()
        if extension not in (".wav", ".seg_b2") or not file.is_file():
            continue
        relative = file.relative_to(root).with_suffix("")
        files = wavs if extension == ".wav" else annotations
        if relative in files:
            raise ValueError(f"Два файла одного типа для записи {relative}")
        files[relative] = file
    pairs = [(wav, annotations[name], name) for name, wav in wavs.items() if name in annotations]
    return pairs, [wav.relative_to(root) for name, wav in wavs.items() if name not in annotations]


def run():
    st.set_page_config(page_title="Фонемные эмбеддинги", page_icon="🎙️")
    page = st.sidebar.radio("Раздел", ["Обработка", "Визуализация", "Векторы", "Тестирование модели", "Обучение модели", "Фонетические признаки", "Граф переходов", "Эксперименты MLP"])
    if page == "Эксперименты MLP":
        from mlp_experiments_view import render
        render(ROOT, local_path)
        return
    if page == "Граф переходов":
        from transition_graph_view import render
        render(ROOT)
        return
    if page == "Фонетические признаки":
        from phoneme_embeddings.classifier.feature_view import render
        render(ROOT, local_path)
        return
    if page == "Обучение модели":
        from phoneme_embeddings.classifier.training_view import render
        render(local_path)
        return
    if page == "Тестирование модели":
        from phoneme_embeddings.classifier.view import render
        render(local_path)
        return
    if page == "Векторы":
        from vector_view import render
        render(ROOT)
        return
    if page == "Визуализация":
        from visualization import render
        render(ROOT)
        return
    st.title("Извлечение фонемных эмбеддингов")
    st.write("Все выбранные модели обработают одни и те же записи и фонемную разметку.")
    configs = {path.stem: json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((ROOT / "configs").glob("*.json"))}
    configs = {name: config for name, config in configs.items()
               if isinstance(config.get("encoder"), dict) and "checkpoint" in config["encoder"]}
    checkpoints = {name: local_path(config["encoder"]["checkpoint"]) for name, config in configs.items()}
    ready = [name for name in configs if checkpoints[name].is_file()]
    unavailable = [name for name in configs if name not in ready]
    if unavailable:
        with st.expander("Модели, для которых ещё нужны веса"):
            for name in unavailable:
                st.write(MODEL_LABELS.get(name, name))
                st.code(str(checkpoints[name].relative_to(ROOT) if checkpoints[name].is_relative_to(ROOT)
                            else checkpoints[name]), language=None)

    is_corpus = st.radio("Источник", ["Одна запись", "Папка корпуса"], horizontal=True) == "Папка корпуса"
    with st.form("extraction"):
        if is_corpus:
            corpus_text = st.text_input("Путь к папке корпуса", "data/examples")
            st.caption("Для каждого WAV ищу файл .seg_B2 с тем же именем в той же подпапке.")
        else:
            wav_text = st.text_input("Путь к WAV", "data/examples/ata1101.wav")
            seg_text = st.text_input("Путь к разметке .seg_B2", "data/examples/ata1101.seg_B2")
        selected = st.multiselect("Энкодеры", ready, default=ready[:1],
                                  format_func=lambda name: MODEL_LABELS.get(name, name))
        modes = st.multiselect("Режимы извлечения", list(MODE_LABELS), default=list(MODE_LABELS),
                               format_func=MODE_LABELS.get)
        output_text = st.text_input("Каталог результатов", "data/extracted")
        resume = st.checkbox("Только необработанные", value=True)
        st.caption("Готовые записи с той же моделью и настройками будут пропущены. "
                   "Для продолжения укажи прежний каталог результатов.")
        submitted = st.form_submit_button("Запустить извлечение", type="primary", disabled=not ready)

    if not submitted:
        return
    if not selected or not modes or not output_text.strip():
        st.error("Выбери хотя бы одну модель и режим, а также укажи каталог результатов.")
        return

    if is_corpus:
        corpus = local_path(corpus_text)
        if not corpus_text.strip() or not corpus.is_dir():
            st.error("Укажи существующую папку корпуса.")
            return
        try:
            pairs, missing = corpus_pairs(corpus)
        except (OSError, ValueError) as error:
            st.error(str(error))
            return
        if missing:
            with st.expander(f"Пропущены записи без разметки: {len(missing)}"):
                st.code("\n".join(map(str, missing)), language=None)
        if not pairs:
            st.error("В папке нет пар WAV и .seg_B2.")
            return
        st.write(f"Найдено записей с разметкой: {len(pairs)}")
    else:
        wav, seg = local_path(wav_text), local_path(seg_text)
        if not wav.is_file() or wav.suffix.lower() != ".wav":
            st.error("Укажи существующий WAV-файл.")
            return
        if not seg.is_file() or seg.suffix.lower() != ".seg_b2":
            st.error("Укажи существующий файл разметки .seg_B2.")
            return
        pairs = [(wav, seg, Path(wav.stem))]

    output_root = local_path(output_text)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    phases = ["read", "load_model", "encode",
              *(mode for mode in MODE_LABELS if mode in modes), "metadata"]
    total_runs = len(selected) * len(pairs)
    total = total_runs * len(phases)
    progress = st.progress(0, text="Подготовка")
    succeeded = 0
    skipped_total = 0
    failed_total = 0
    for model_index, name in enumerate(selected):
        title = MODEL_LABELS.get(name, name)
        config = {**configs[name], "modes": modes,
                  "encoder": {**configs[name]["encoder"], "checkpoint": str(checkpoints[name])}}
        load_encoder = lru_cache(maxsize=1)(create_encoder)
        with st.status(f"{model_index + 1}/{len(selected)} · {title}", expanded=True) as status:
            current = status.empty()
            failures = []
            skipped = 0
            done = set()
            if resume:
                current.write("Проверяю ранее сохраненные результаты…")
                done = completed_pairs(
                    output_root / name, config,
                    on_scan=lambda count: current.write(f"Проверяю результаты: {count}"),
                )
            pending = sum((wav.resolve(), seg.resolve()) not in done for wav, seg, _ in pairs)
            st.write(f"Уже готовы: {len(pairs) - pending}; осталось обработать: {pending}")
            log_path = output_root / name / "logs" / f"{run_id}.jsonl"

            def log_result(wav, seg, relative, result, **details):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                record = {"time": datetime.now().astimezone().isoformat(),
                          "pair_id": str(relative), "wav": str(wav.resolve()),
                          "segmentation": str(seg.resolve()), "status": result,
                          "config": config, **details}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())

            for file_index, (wav, seg, relative) in enumerate(pairs):
                index = model_index * len(pairs) + file_index
                if (wav.resolve(), seg.resolve()) in done:
                    skipped += 1
                    skipped_total += 1
                    progress.progress((index + 1) / total_runs,
                                      text=f"{title} · уже готово: {relative}")
                    continue
                out = (output_root / name / corpus.name / run_id / relative if is_corpus
                       else output_root / name / relative / run_id)

                def show(stage: str):
                    completed = index * len(phases) + phases.index(stage)
                    message = f"{file_index + 1}/{len(pairs)} · {relative} · {STAGES[stage]}"
                    progress.progress(completed / total, text=f"{title} · {message}")
                    current.write(message)

                log_result(wav, seg, relative, "started", output=str(out))
                try:
                    extract(wav, seg, out, config, on_stage=show, encoder_factory=load_encoder)
                except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
                    failures.append(f"{relative}: {error}")
                    failed_total += 1
                    log_result(wav, seg, relative, "error", error=str(error), output=str(out))
                else:
                    succeeded += 1
                    log_result(wav, seg, relative, "complete", output=str(out))
            state = "error" if failures else "complete"
            status.update(label=f"{title} · новых: {len(pairs) - len(failures) - skipped}, "
                                f"уже готовых: {skipped}, ошибок: {len(failures)}",
                          state=state, expanded=bool(failures))
            if failures:
                status.code("\n".join(failures), language=None)
        load_encoder.cache_clear()
        if len(failures) + skipped < len(pairs):
            folder = output_root / name / corpus.name / run_id if is_corpus else out
            st.code(str(folder), language=None)

    progress.progress(1.0, text=f"Завершено · успешно: {succeeded}, уже готовых: {skipped_total}, ошибок: {failed_total}")


if __name__ == "__main__":
    run()
