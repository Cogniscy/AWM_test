"""Streamlit editor for the CORPRES transition graph."""
import json
import math

import altair as alt
import pandas as pd
import streamlit as st

from transition_graph import differences, edit, initial_graph, save_version, two_step_paths, version_id


def network(labels, edges):
    coords = {label: (math.cos(angle), math.sin(angle))
              for i, label in enumerate(labels)
              for angle in [2 * math.pi * i / len(labels)]}
    nodes = pd.DataFrame([{"label": label, "x": x, "y": y}
                          for label, (x, y) in coords.items()])
    chooser = alt.selection_point(name="vertex", fields=["label"], on="click", toggle=False)
    base = alt.Chart(nodes)
    points = base.mark_circle(size=450, opacity=1).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.16, 1.16])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.16, 1.16])),
        color=alt.condition(chooser, alt.value("#3878a9"), alt.value("#a1b5c6")),
        tooltip=[alt.Tooltip("label:N", title="Метка")],
    ).add_params(chooser)
    captions = base.mark_text(color="white").encode(x="x:Q", y="y:Q", text="label:N")
    if not edges:
        return (points + captions).properties(height=580)
    lines = []
    for edge in edges:
        x, y = coords[edge["source"]]
        end_x, end_y = coords[edge["target"]]
        dx, dy = end_x - x, end_y - y
        length = math.hypot(dx, dy)
        offset_x, offset_y = -.014 * dy / length, .014 * dx / length
        x, y = x + offset_x, y + offset_y
        end_x, end_y = end_x + offset_x, end_y + offset_y
        lines.append({**edge, "x": x, "y": y, "end_x": end_x, "end_y": end_y,
                      "arrow_x": end_x - .06 * dx / length,
                      "arrow_y": end_y - .06 * dy / length,
                      "angle": math.degrees(math.atan2(dx, dy)) % 360})
    line = alt.Chart(pd.DataFrame(lines))
    tooltip = ["source:N", "target:N", "feature:N", "from:N", "to:N"]
    rules = line.mark_rule(opacity=.35).encode(
        x="x:Q", y="y:Q", x2="end_x:Q", y2="end_y:Q", color="feature:N", tooltip=tooltip,
    )
    arrows = line.mark_point(shape="triangle-up", filled=True, size=42).encode(
        x="arrow_x:Q", y="arrow_y:Q", angle=alt.Angle("angle:Q", scale=None),
        color="feature:N", tooltip=tooltip,
    )
    return (rules + arrows + points + captions).properties(height=580)


def render(root):
    st.title("Граф переходов")
    directory = root / "configs/transition_graphs"
    versions = {p.stem: p for p in sorted(directory.glob("*.json"))}
    base = st.selectbox("Версия графа", ["pilot", *versions],
                        format_func=lambda v: "Исходный пилот" if v == "pilot" else v[:16])
    try:
        if st.session_state.get("transition_base") != base:
            graph = initial_graph(root / "configs") if base == "pilot" else json.loads(versions[base].read_text())
            if base != "pilot" and version_id(graph) != base:
                raise ValueError("Файл сохранённой версии изменён.")
            st.session_state.transition_draft = graph
            st.session_state.transition_base = base
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Не удалось открыть граф: {error}")
        return

    graph = st.session_state.transition_draft
    labels = sorted(graph["labels"])
    enabled = [edge for edge in graph["edges"] if edge["enabled"]]
    st.caption(f"{len(labels)} метка · {len(enabled)} включённых переходов · "
               f"{len(graph['edges']) - len(enabled)} отключено")
    view, editor, routes = st.tabs(["Просмотр", "Редактирование", "Двухшаговые пути"])
    with view:
        focus = st.selectbox("Фильтр по метке", [None, *labels], format_func=lambda v: v or "Все")
        features = st.multiselect("Фильтр по признаку", sorted({e["feature"] for e in enabled}))
        shown = [e for e in enabled if (focus is None or focus in (e["source"], e["target"]))
                 and (not features or e["feature"] in features)]
        shown_labels = labels if focus is None and not features else sorted(
            {label for edge in shown for label in (edge["source"], edge["target"])} | ({focus} if focus else set()))
        selected = st.altair_chart(network(shown_labels, shown), width="stretch",
                                  on_select="rerun", selection_mode="vertex", key="transition_network")
        selected_nodes = selected.selection.get("vertex", [])
        label = selected_nodes[0]["label"] if selected_nodes else focus
        label = st.selectbox("Метка для просмотра", labels,
                             index=labels.index(label) if label in labels else 0,
                             key=f"transition_inspect_{label}")
        info = graph["labels"][label]
        st.write(f"**{label} · {info['ipa']}**")
        st.write(info["features"])
        st.dataframe([info["counts"]], hide_index=True)
        st.dataframe([{**e, "from": str(e["from"]), "to": str(e["to"])}
                      for e in enabled if label in (e["source"], e["target"])], hide_index=True)
    with editor:
        source = st.selectbox("Из", labels, index=labels.index("b") if "b" in labels else 0)
        target = st.selectbox("В", labels, index=labels.index("p") if "p" in labels else 0)
        active = {(e["source"], e["target"]) for e in enabled}
        forward = "включён" if (source, target) in active else "отключён или отсутствует"
        backward = "включён" if (target, source) in active else "отключён или отсутствует"
        st.caption(f"{source} → {target} · {forward} | {target} → {source} · {backward}")
        diff = differences(graph["labels"], source, target)
        st.dataframe([{"Признак": k, "До": str(v[0]), "После": str(v[1])}
                      for k, v in diff.items()], hide_index=True)
        actions = {"enable": "Добавить или включить", "disable": "Отключить",
                   "reverse": "Развернуть", "both": "Включить оба направления"}
        action = st.selectbox("Действие", list(actions), format_func=actions.get)
        if st.button("Применить переход"):
            try:
                st.session_state.transition_draft = edit(graph, source, target, action)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    with routes:
        paths = two_step_paths(enabled)
        multiple = st.checkbox("Только пары с несколькими порядками признаков")
        candidates = sorted(key for key, values in paths.items()
                            if not multiple or len({tuple(edge["feature"] for edge in path)
                                                    for path in values}) > 1)
        st.caption(f"Доступно пар с двухшаговым маршрутом: {len(candidates)}")
        if candidates:
            starts = sorted({a for a, _ in candidates})
            begin = st.selectbox("Начало маршрута", starts, index=starts.index("b") if "b" in starts else 0)
            ends = [b for a, b in candidates if a == begin]
            end = st.selectbox("Цель маршрута", ends, index=ends.index("p'") if "p'" in ends else 0)
            st.dataframe([{"Маршрут": f"{begin} → {a['target']} → {end}",
                           "Первый шаг": a["feature"], "Второй шаг": b["feature"]}
                          for a, b in paths[begin, end]], hide_index=True)
    st.caption(f"Идентификатор текущего графа · {version_id(graph)}")
    if st.button("Сохранить версию графа", type="primary"):
        try:
            saved = save_version(directory, graph)
            st.success(f"Версия сохранена: {saved.relative_to(root)}")
        except (OSError, ValueError) as error:
            st.error(f"Не удалось сохранить версию: {error}")
