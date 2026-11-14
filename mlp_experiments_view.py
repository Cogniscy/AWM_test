"""Browser interface for direct residual MLP comparisons."""
import json

import pandas as pd
import streamlit as st

from mlp_experiments import SPLITS, prepare, run_experiment
from transition_graph import initial_graph, version_id


MODELS = {
    "unconditioned": "MLP без условия",
    "conditioned": "MLP с кодом перехода",
    "identity": "Исходный вектор",
    "mean_shift": "Средний сдвиг",
    "oracle": "Целевой прототип",
}


def render(root, local_path):
    st.title("Эксперименты MLP")
    configs = root / "configs"
    versions = {path.stem: path for path in sorted((configs / "transition_graphs").glob("*.json"))}
    version = st.selectbox("Версия графа", ["pilot", *versions],
                           format_func=lambda key: "Исходный пилот" if key == "pilot" else key[:16])
    try:
        graph = initial_graph(configs) if version == "pilot" else json.loads(versions[version].read_text())
        if version != "pilot" and version_id(graph) != version:
            raise ValueError("Файл сохранённой версии графа изменён.")
    except (OSError, ValueError, KeyError) as error:
        st.error(str(error))
        return
    st.caption(f"Граф {version_id(graph)}")
    pilot = local_path(st.text_input("Папка сохранённых прототипов", "reports/corpres_mlp_pilot/results"))
    mode = st.radio("Проверка", ["Известные переходы на других дикторах", "Отложенные пары"])
    edges = [edge for edge in graph["edges"] if edge["enabled"]]
    pairs = sorted({tuple(sorted((edge["source"], edge["target"]))) for edge in edges})
    held = []
    if mode == "Отложенные пары":
        held = st.multiselect("Пары для теста", pairs, format_func=lambda pair: " ↔ ".join(pair))
        if not held:
            st.info("Выбери хотя бы одну пару. Оба направления будут исключены из train и validation.")
    counts = {"train": 0, "validation": 0, "test": 0}
    excluded = {frozenset(pair) for pair in held}
    for edge in edges:
        pair = frozenset((edge["source"], edge["target"]))
        if pair not in excluded:
            counts["train"] += 1
            counts["validation"] += 1
        if not excluded or pair in excluded:
            counts["test"] += 1
    st.dataframe([{"Часть": split, "Дикторы": ", ".join(graph["split"]["speakers"][split]),
                   "Переходы": counts[split]} for split in SPLITS], hide_index=True)
    with st.form("mlp_experiment"):
        models = st.multiselect("Модели для сравнения", ["unconditioned", "conditioned"],
                                default=["unconditioned", "conditioned"], format_func=MODELS.get)
        columns = st.columns(3)
        width = columns[0].number_input("Ширина", 1, 512, 128)
        depth = columns[1].number_input("Скрытых слоёв", 1, 5, 1)
        seed = columns[2].number_input("Seed", 0, 999999, 42)
        epochs = st.number_input("Эпохи", 1, 3000, 400)
        lr = st.number_input("Learning rate", 0.000001, 0.1, 0.003, format="%.6f")
        checkpoint = local_path(st.text_input(
            "Замороженный классификатор",
            f"src/phoneme_embeddings/classifier/weights/{graph['classifier']}_model.pth"))
        clicked = st.form_submit_button("Запустить эксперимент", type="primary",
                                        disabled=mode == "Отложенные пары" and not held)
    if clicked:
        try:
            if not models:
                raise ValueError("Выбери хотя бы одну модель.")
            batches, operations = prepare(graph, pilot, held)
            settings = {"width": int(width), "depth": int(depth), "seed": int(seed),
                        "epochs": int(epochs), "lr": lr, "models": models,
                        "holdout_pairs": held, "optimizer": "Adam", "selection": "validation_mse"}
            progress = st.progress(0, text="Подготовка эксперимента")

            def notify(model, epoch, loss):
                fraction = (models.index(model) + epoch / epochs) / len(models)
                progress.progress(fraction, text=f"{MODELS[model]} · эпоха {epoch}/{epochs} · val MSE {loss:.6g}")

            folder, result = run_experiment(graph, batches, operations, settings, checkpoint, pilot,
                                            root / "reports/mlp_experiments", notify)
            progress.progress(1.0, text="Эксперимент завершён")
            st.session_state.mlp_experiment_result = (str(folder), result)
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
            st.error(f"Эксперимент не выполнен: {error}")
    if "mlp_experiment_result" not in st.session_state:
        return
    folder, result = st.session_state.mlp_experiment_result
    st.subheader("Последний завершённый эксперимент")
    st.caption(f"Граф {result['graph_version']} · seed {result['settings']['seed']}")
    st.code(folder, language=None)
    rows = [{"Модель": MODELS[name], "MSE": m["mse"], "Точность, %": 100 * m["accuracy"],
             "Верно": f"{m['correct']}/{m['total']}", "Параметры": m.get("parameters"),
             "Лучшая эпоха": m.get("epoch")}
            for name, m in result["metrics"].items()]
    st.dataframe(rows, hide_index=True)
    name = st.selectbox("Предсказания по переходам", list(result["metrics"]), format_func=MODELS.get)
    predictions = [{**row, "from": str(row["from"]), "to": str(row["to"])}
                   for row in result["metrics"][name]["predictions"]]
    st.dataframe(predictions, hide_index=True)
    if "history" in result["metrics"][name]:
        st.line_chart(pd.DataFrame(result["metrics"][name]["history"]), x="epoch",
                      y=["train_mse", "validation_mse"])
    st.download_button("Скачать метрики JSON", json.dumps(result, ensure_ascii=False, indent=2),
                       "mlp_experiment_metrics.json", "application/json")
