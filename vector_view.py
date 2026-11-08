"""Sample saved embeddings and inspect reproducible two-dimensional projections."""

import colorsys
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from visualization import result_index


def phoneme_group(label):
    # Working CORPRES convention: vowel digit variants share a base.
    return re.sub(r"^([aeiouy])[0-9]+$", r"\1", label)


def sample_vectors(root, paths, mode, limit, seed, pure):
    rng = np.random.default_rng(seed)
    vectors, records, priorities = [], [], np.empty(0)
    dimension = None
    for relative in paths:
        folder = root / relative
        if not (folder / f"{mode}.npy").is_file():
            continue
        matrix = np.load(folder / f"{mode}.npy", mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2:
            raise ValueError(f"Неверная форма массива: {folder}")
        if dimension is None:
            dimension = matrix.shape[1]
        if dimension != matrix.shape[1]:
            raise ValueError("В выборке разные размерности эмбеддингов")
        metadata = [json.loads(line) for line in (folder / f"{mode}.jsonl").read_text(encoding="utf-8").splitlines()]
        local_records, rows = [], []
        for item in metadata:
            row = item["row"]
            if row is None:
                continue
            labels = item.get("labels", [])
            if labels and pure and len(set(labels)) > 1:
                continue
            label = item.get("label") or labels[1]
            if label == "sil":
                continue
            if not isinstance(row, int) or not 0 <= row < len(matrix):
                raise ValueError(f"Неверный индекс вектора: {folder}")
            local_records.append({"Аллофон": label, "Запись": relative,
                                  "Начало, с": item["start_sec"], "Конец, с": item["end_sec"]})
            rows.append(row)
        if not rows:
            continue
        batch = np.asarray(matrix[rows])
        finite = np.isfinite(batch).all(axis=1)
        batch = batch[finite]
        local_records = [r for r, ok in zip(local_records, finite) if ok]
        if not len(batch):
            continue
        # Random priorities give a uniform bounded sample across all candidate rows.
        priorities = np.concatenate((priorities, rng.random(len(batch))))
        records.extend(local_records)
        vectors = np.vstack((vectors, batch)) if len(vectors) else batch
        keep = np.argsort(priorities)[:limit]
        priorities = priorities[keep]
        vectors = vectors[keep]
        records = [records[i] for i in keep]
    if len(records) < 3:
        raise ValueError("Нужно хотя бы три подходящих вектора. Расширь выборку.")
    return np.asarray(vectors), pd.DataFrame(records)


def project(vectors, method, seed, params=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    params = dict(params or {})
    scale = params.pop("standardize", False)
    source = StandardScaler().fit_transform(vectors) if scale else vectors
    if method == "PCA":
        effective = {"n_components": 2, "whiten": params.get("whiten", False),
                     "svd_solver": params.get("svd_solver", "auto"), "random_state": seed}
        pca = PCA(**effective)
        coords = pca.fit_transform(source)
        note = f"Две компоненты объясняют {pca.explained_variance_ratio_.sum():.1%} дисперсии."
    else:
        pre = int(params.pop("pre_pca", 50))
        actual_pre = min(pre, len(source) - 1, source.shape[1])
        if pre:
            source = PCA(n_components=actual_pre, random_state=seed).fit_transform(source)
        effective = {
            "n_components": 2, "perplexity": params.get("perplexity", min(30, (len(source) - 1) / 3)),
            "learning_rate": params.get("learning_rate", "auto"),
            "early_exaggeration": params.get("early_exaggeration", 12.0),
            "max_iter": params.get("max_iter", 1000), "metric": params.get("metric", "euclidean"),
            "init": params.get("init", "pca"), "angle": params.get("angle", .5),
            "random_state": seed,
        }
        if effective["perplexity"] >= len(source):
            raise ValueError(f"Perplexity должна быть меньше числа точек ({len(source)}).")
        coords = TSNE(**effective).fit_transform(source)
        effective["pre_pca"] = actual_pre if pre else 0
        note = "t-SNE показывает локальное соседство. Расстояния между далёкими группами условны."
    effective["standardize"] = scale
    return coords, note, effective


def save_projection(chart, directory, metadata, formats):
    import vl_convert as vl

    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    params = metadata["parameters"]
    short = {"std": params["standardize"], "seed": params["random_state"]}
    if metadata["method"] == "PCA":
        short.update(whiten=params["whiten"], solver=params["svd_solver"])
    else:
        short.update(pp=params["perplexity"], lr=params["learning_rate"],
                     iter=params["max_iter"], ee=params["early_exaggeration"],
                     metric=params["metric"], init=params["init"], angle=params["angle"],
                     pre=params["pre_pca"])
    def safe(value):
        return re.sub(r"[^\w.-]+", "-", str(value)).strip("-") or "dataset"
    parameters = "_".join(f"{k}-{safe(v)}" for k, v in short.items())
    stem = (f"vector-visualization_{safe(metadata['method'])}_{safe(metadata['dataset'])[:25]}_"
            f"{safe(metadata['model'])[:25]}_{metadata['mode']}_n-{metadata['visible_points']}_{parameters}")
    # Full settings, paths and grouping are also stored next to the image.
    stem = stem.encode("utf-8")[:210].decode("utf-8", errors="ignore") + "_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with alt.data_transformers.enable("default", max_rows=None):
        spec = chart.to_dict()
    outputs = []
    for extension in formats:
        payload = vl.vegalite_to_png(spec, scale=2) if extension == "png" else vl.vegalite_to_svg(spec).encode("utf-8")
        path = directory / f"{stem}.{extension}"
        with path.open("xb") as handle:
            handle.write(payload)
        outputs.append(path)
    sidecar = directory / f"{stem}.json"
    with sidecar.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return outputs, sidecar


def label_colors(labels, mapping):
    families = {}
    for label in sorted(labels):
        families.setdefault(mapping.get(label, label), []).append(label)
    colors = {}
    for family, members in families.items():
        digest = hashlib.sha256(family.encode()).digest()
        hue = int.from_bytes(digest[:4], "big") / 2**32
        for i, label in enumerate(members):
            light = .38 + .30 * i / max(1, len(members) - 1)
            rgb = colorsys.hls_to_rgb(hue, light, .7)
            colors[label] = "#" + "".join(f"{round(c * 255):02x}" for c in rgb)
    return colors


def render(root):
    st.title("Визуализация векторов")
    st.write("Каждая точка — эмбеддинг кадра или отдельного произнесения фонемы.")
    output = Path(st.text_input("Каталог результатов", str(root / "data/extracted"))).expanduser()
    if not output.is_absolute():
        output = root / output
    index = result_index(str(output))
    if not index:
        st.info("Сначала извлеки эмбеддинги.")
        return
    method = st.selectbox("Уменьшение размерности", ["PCA", "t-SNE"])
    with st.form("projection"):
        model = st.selectbox("Модель", sorted({Path(p).parts[0] for p in index}))
        mode = st.selectbox("Векторы", ["segment_mean", "frame"],
                            format_func=lambda m: "Средние по фонемным сегментам" if m == "segment_mean" else "Кадры")
        query = st.text_input("Фильтр записей по пути", placeholder="Диктор, дата или имя записи")
        count = st.number_input("Максимум записей", 1, 1000, 100)
        limit = st.number_input("Максимум точек", 100, 10000, 3000, step=100)
        standardize = st.checkbox("Стандартизировать координаты перед проекцией", False)
        if method == "PCA":
            whiten = st.checkbox("PCA · whitening", False)
            solver = st.selectbox("PCA · solver", ["auto", "full", "randomized"])
            params = {"standardize": standardize, "whiten": whiten, "svd_solver": solver}
        else:
            perplexity = st.number_input("t-SNE · perplexity", 1.0, 500.0, 30.0)
            auto_lr = st.checkbox("t-SNE · learning rate auto", True)
            learning_rate = st.number_input("t-SNE · learning rate", 1.0, 10000.0, 200.0, disabled=auto_lr)
            early = st.number_input("t-SNE · early exaggeration", 1.0, 100.0, 12.0)
            iterations = st.number_input("t-SNE · максимум итераций", 250, 10000, 1000, step=250)
            metric = st.selectbox("t-SNE · метрика", ["euclidean", "cosine"])
            init = st.selectbox("t-SNE · инициализация", ["pca", "random"])
            angle = st.slider("t-SNE · angle", 0.0, 1.0, .5, .05)
            pre = st.number_input("t-SNE · PCA перед проекцией (0 — отключить)", 0, 256, 50)
            params = {"standardize": standardize, "perplexity": perplexity,
                      "learning_rate": "auto" if auto_lr else learning_rate,
                      "early_exaggeration": early, "max_iter": int(iterations),
                      "metric": metric, "init": init, "angle": angle, "pre_pca": int(pre)}
        seed = st.number_input("Seed выборки и проекции", 0, 999999, 42)
        pure = st.checkbox("Для кадров брать только одинаковые метки во всех трёх частях", True)
        clicked = st.form_submit_button("Построить проекцию", type="primary")
    if clicked:
        try:
            paths = [p for p in index if Path(p).parts[0] == model and query.casefold() in p.casefold()]
            rng = np.random.default_rng(seed)
            if len(paths) > count:
                paths = sorted(rng.choice(paths, int(count), replace=False).tolist())
            with st.spinner("Собираю выборку и рассчитываю проекцию"):
                vectors, data = sample_vectors(output, paths, mode, int(limit), int(seed), pure)
                coords, note, effective = project(vectors, method, int(seed), params)
                data["X"], data["Y"] = coords.T
                first = json.loads((output / paths[0] / "run.json").read_text())
                source = Path(first["wav"])
                dataset = "CORPRES" if "CORPRES" in source.parts else source.parent.name
                metadata = {"dataset": dataset, "results_root": str(output.resolve()), "model": model,
                            "method": method, "mode": mode, "parameters": effective, "seed": int(seed),
                            "query": query, "recording_limit": int(count), "point_limit": int(limit),
                            "pure_frames": pure, "source_results": paths, "total_points": len(data)}
                st.session_state["vector_projection"] = (data, note, model, method, len(paths), metadata)
        except (OSError, ValueError, KeyError, IndexError, ImportError) as error:
            st.error(str(error))
    if "vector_projection" not in st.session_state:
        return
    saved = st.session_state["vector_projection"]
    if len(saved) != 6:
        st.info("Перестрой проекцию, чтобы сохранить её с параметрами.")
        return
    data, note, model, method, count, metadata = saved
    st.caption(f"Построенный график · {model} · {method} · {len(data)} точек · {count} записей")
    st.caption(note)
    labels = sorted(data["Аллофон"].unique())
    defaults = {label: phoneme_group(label) for label in labels}
    with st.expander("Группировка аллофонов по фонемам"):
        st.write("По умолчанию a0, a1, a2 и другие цифровые варианты гласной имеют общий цвет a. "
                 "Мягкие согласные остаются отдельными группами. Это рабочая группировка по обозначениям CORPRES. "
                 "При необходимости измени соответствия в таблице.")
        mapping_table = st.data_editor(
            pd.DataFrame({"Аллофон": labels, "Фонема": [defaults[l] for l in labels]}),
            disabled=["Аллофон"], hide_index=True,
            key="groups_" + hashlib.sha256("\n".join(labels).encode()).hexdigest()[:12],
        )
    mapping = {label: (str(group).strip() if pd.notna(group) and str(group).strip() else label)
               for label, group in zip(mapping_table["Аллофон"], mapping_table["Фонема"])}
    display = data.copy()
    display["Фонема"] = display["Аллофон"].map(mapping)
    selected = st.multiselect("Показать фонемы", sorted(display["Фонема"].unique()),
                              default=sorted(display["Фонема"].unique()),
                              key="families_" + hashlib.sha256(str(mapping).encode()).hexdigest()[:12])
    display = display[display["Фонема"].isin(selected)]
    palette = label_colors(labels, mapping)
    chart = alt.Chart(display).mark_circle(size=35, opacity=.65).encode(
        x=alt.X("X:Q", title=f"{method} · ось 1"),
        y=alt.Y("Y:Q", title=f"{method} · ось 2"),
        color=alt.Color("Аллофон:N", scale=alt.Scale(domain=labels, range=[palette[l] for l in labels]),
                        legend=alt.Legend(title="Аллофон")),
        tooltip=["Фонема:N", "Аллофон:N", "Запись:N",
                 alt.Tooltip("Начало, с:Q", format=".3f"),
                 alt.Tooltip("Конец, с:Q", format=".3f")],
    ).properties(height=580).interactive()
    with alt.data_transformers.enable("default", max_rows=None):
        st.altair_chart(chart, width="stretch")
    st.download_button("Скачать координаты и метки CSV", display.to_csv(index=False).encode("utf-8"),
                       "embedding_projection.csv", "text/csv")

    with st.expander("Параметры построенного графика"):
        st.json(metadata["parameters"])
    st.subheader("Сохранить изображение")
    dataset_name = st.text_input("Название датасета для имени файла", metadata["dataset"])
    target = st.text_input("Папка для изображений", str(Path.home() / "Downloads"))
    formats = st.multiselect("Форматы изображения", ["png", "svg"], default=["png"])
    st.caption("Сохраняется весь график выбранных фонем. Параметры и группировка записываются рядом в JSON.")
    if st.button("Сохранить картинку в папку"):
        if not formats or not target.strip() or not dataset_name.strip():
            st.error("Укажи название датасета, папку и хотя бы один формат.")
        else:
            try:
                info = {**metadata, "dataset": dataset_name, "visible_points": len(display),
                        "selected_phonemes": selected, "allophone_mapping": mapping, "colors": palette}
                outputs, sidecar = save_projection(chart, target, info, formats)
                st.success("Изображения сохранены")
                for path in [*outputs, sidecar]:
                    st.code(str(path), language=None)
            except (OSError, ValueError, ImportError, RuntimeError) as error:
                st.error(f"Не удалось сохранить изображение: {error}")
