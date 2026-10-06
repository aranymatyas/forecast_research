"""Plotting utilities for the :mod:`python.full_view` EventSamples framework.

The main entry point is :func:`mosaic`, which renders a Marimekko / mosaic
diagram directly from :class:`~python.full_view.EventSamples` objects.

Design notes
------------
A mosaic (Marimekko) diagram encodes a bivariate categorical distribution:

* The plot width is split into **columns** whose widths are proportional to the
  number of units falling into each X-partition.
* Each column is split into **segments** whose heights are proportional to the
  *conditional* number of units of each Y-partition *within that column*.

The X and Y partitions are each supplied as a list of ``EventSamples``. Every
``EventSamples`` only exposes a boolean ``pandas.Series`` (``.data``) and a
human-readable ``.desc``. Because several events may be ``True`` for the same
unit (and the author explicitly wants overlaps to be visible), we do **not**
assume the supplied events are mutually exclusive. Instead, for each axis we
build the partition from the *combinations of membership* across the supplied
events -- i.e. the truth table of which events hold for each unit:

* a unit that matches exactly one event lands in that event's cell;
* a unit that matches several events lands in an explicit "A & B" overlap cell;
* a unit that matches none lands in a "none" cell.

This mirrors the ``' + '.join(parts)`` categorisation used in
``nb/m_datasets/all_per_series_tests.ipynb`` and guarantees the resulting cells
are mutually exclusive and exhaustive, which is what a mosaic diagram requires.
"""

from __future__ import annotations

from functools import reduce
from operator import and_
from typing import Sequence, Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from python.full_view import Event, CombineMethod, CombinePValues
import python.full_view as _fv


__all__ = [
    "mosaic",
    "bar",
    "association_matrix",
    "upset",
    "prob_vs_alpha",
    "corr_vs_alpha",
    "combine_significance",
]


def _compose_label(events: Sequence[Event], indices: Sequence[int]) -> str:
    """Compose a cell label from the matched events using ``EventSamples.&``.

    Reuses :meth:`EventSamples.__and__` so the description formatting (the
    ``(A) ∩ (B)`` composition) lives in one place and stays consistent with the
    rest of the event algebra. An empty selection yields ``'none'``.
    """
    if not indices:
        return "none"
    combined = reduce(and_, (events[i] for i in indices))
    return combined.desc


def _partition(
    events: Sequence[Event], exclusive: bool = False
) -> tuple[pd.DataFrame, dict[str, tuple[int, ...]]]:
    """Turn a list of events into exclusive cells.

    Returns a ``DataFrame`` indexed like the event data, with a single
    ``'cell'`` column giving the exclusive partition label of every unit.

    Cell labels are composed through :func:`_compose_label`, which chains the
    matched events with :meth:`EventSamples.__and__`, so the ``(A) ∩ (B)``
    description formatting is reused rather than re-implemented here.

    When ``exclusive`` is ``False`` (default) the events may overlap: the label
    is built from *every* event that is ``True`` for that unit, or ``'none'``
    when no event matches. Only the combinations that actually occur are
    materialised, so there is no ``2**len(events)`` blow-up.

    When ``exclusive`` is ``True`` the caller guarantees the events are mutually
    exclusive (no unit is ``True`` for more than one). The cross-section logic
    is skipped: each unit is labelled by the single event that matches (first by
    input order if the promise is violated), which is cheapest for long lists.
    """
    if not events:
        raise ValueError("at least one EventSamples must be supplied per axis")

    # Align all boolean series on their shared index and coerce to bool.
    frame = pd.concat(
        [ev.data.rename(i) for i, ev in enumerate(events)], axis=1
    ).fillna(False).astype(bool)

    if exclusive:
        # Fast path: pick the single matching event per unit (first True wins
        # if the "mutually exclusive" promise is violated). Vectorised, so it
        # stays fast for very long event lists.
        cell = pd.Series("none", index=frame.index, dtype=object)
        assigned = pd.Series(False, index=frame.index)
        label_to_indices: dict[str, tuple[int, ...]] = {"none": ()}
        for i, ev in enumerate(events):
            col = frame[i] & ~assigned
            lbl = _compose_label(events, (i,))
            cell[col] = lbl
            label_to_indices[lbl] = (i,)
            assigned |= frame[i]
        return pd.DataFrame({"cell": cell}, index=frame.index), label_to_indices

    # General path: each unit's membership is the tuple of matched indices.
    # We only compose a label for the *distinct* tuples that actually occur,
    # avoiding the 2**N enumeration while still reusing the ``&`` composition.
    membership = frame.apply(lambda row: tuple(np.flatnonzero(row.values)), axis=1)
    combo_to_label = {combo: _compose_label(events, combo) for combo in membership.unique()}
    label_to_indices = {lbl: combo for combo, lbl in combo_to_label.items()}
    # Map tuple-keyed dict explicitly (Series.map treats tuple keys as a
    # MultiIndex, so build the labelled series directly instead).
    cell = pd.Series(
        [combo_to_label[c] for c in membership], index=frame.index, dtype=object
    )
    return pd.DataFrame({"cell": cell}, index=frame.index), label_to_indices


def _ordered_cells(label_to_indices: dict[str, tuple[int, ...]]) -> list[str]:
    """Order the exclusive cells by lexicographic sort of their member indices.

    Each atomic event has an index equal to its position in the event list
    (``events[0]`` -> 0, ...). A cell is a combination of one or more events;
    ``label_to_indices`` maps each occurring cell label to the tuple of its
    members' indices (in increasing order), as produced by :func:`_partition`.
    Sorting these tuples lexicographically groups every cell by its lowest-index
    member, with prefixes sorting before their extensions
    (``(0,) < (0, 1) < (0, 1, 2) < (0, 2) < (1,) ...``).

    For ``[decr, stag, incr]`` this yields:
    ``decr, decr & stag, decr & stag & incr, decr & incr, stag, stag & incr,
    incr`` -- a stable, semantically sensible order derived purely from the
    argument order, with no hardcoded labels and no ties to break. The
    all-``False`` ``'none'`` cell has no members (empty tuple) and is placed
    last. Only labels that actually occur are considered, so there is no
    ``2**N`` enumeration regardless of the number of events.
    """
    def sort_key(item) -> tuple:
        _lbl, indices = item
        # Empty tuple ("none") goes last; otherwise lexicographic order.
        return (not indices, indices)

    return [lbl for lbl, _ in sorted(label_to_indices.items(), key=sort_key)]


def mosaic(
    x_events: Sequence[Event],
    y_events: Sequence[Event],
    *,
    title: str | None = None,
    x_title: str = "",
    y_title: str = "",
    color_sequence: Sequence[str] | None = None,
    gap: float = 0.004,
    normalize: bool = True,
    legend: bool = False,
    x_exclusive: bool = False,
    y_exclusive: bool = False,
    horizontal_line_at: float = 0,
    vertical_line_at: float = 0,
    label_min_height: float = 0.06,
    label_min_width: float = 0.02,
    ignore_none: bool = False,
) -> go.Figure:
    """Build a mosaic (Marimekko) diagram from ``EventSamples``.

    Parameters
    ----------
    x_events
        Events defining the **X** partitioning. Column widths are proportional
        to how many units fall in each resulting (exclusive) X cell. Overlaps
        between the supplied events become their own column.
    y_events
        Events defining the **Y** partitioning. Within every column, segment
        heights are proportional to the conditional count of each Y cell.
    title, x_title, y_title
        Figure / axis titles.
    color_sequence
        Colours for the Y cells. Defaults to ``plotly`` qualitative palette.
    gap
        Fractional gap drawn between columns (and between the stacked segments),
        purely cosmetic.
    normalize
        When ``True`` (default) widths/heights are fractions in ``[0, 1]`` and
        the full canvas is used. When ``False`` the axes carry raw counts.
    legend
        When ``False`` (default) the legend is omitted and the Y partition
        names are placed on the (left) y-axis instead, aligned to the first
        column's segments. Set ``True`` to show the legend as well.
    x_exclusive, y_exclusive
        Skip the cross-section (combination) building for that axis. Set these
        when you have curated the event list yourself and *know* the events are
        mutually exclusive (no unit matches more than one), so the overlap cells
        would be empty anyway. For long event lists (roughly > 10) this avoids
        the ``2**N`` combination enumeration and the per-row label ``apply``,
        which otherwise become very slow. Each unit is then labelled by the
        single event it matches (or ``'none'``). If the exclusivity promise is
        violated, the first matching event (by input order) wins.
    label_min_height, label_min_width
        Thresholds (fractions in ``[0, 1]``) controlling when a segment's
        ``count`` / ``%`` text is drawn. A segment is labelled only if its
        height exceeds ``label_min_height`` (as a share of its column's height)
        *and* its column's width exceeds ``label_min_width`` (as a share of the
        total width). Raise them to declutter, set both to ``0`` to always
        label.
    ignore_none
        When ``True``, units whose X *or* Y partition falls in the catch-all
        ``'none'`` cell (i.e. they match none of the supplied events on that
        axis) are dropped before plotting. Neither axis then shows a ``'none'``
        column or segment, and widths / conditional heights are recomputed over
        the remaining units only.

    Returns
    -------
    plotly.graph_objects.Figure
        A mosaic diagram. Hovering a tile reports the X cell, Y cell, count and
        both the conditional (within-column) and overall fractions.
    """

    # Align both partitions on the common index so counts are consistent.
    x_df, x_label_indices = _partition(x_events, exclusive=x_exclusive)
    y_df, y_label_indices = _partition(y_events, exclusive=y_exclusive)
    x_part = x_df["cell"]
    y_part = y_df["cell"]

    joined = pd.DataFrame({"x": x_part, "y": y_part}).dropna()

    if ignore_none:
        # Drop units whose X or Y partition is the catch-all "none" cell, so
        # neither axis shows a "none" column/segment. Widths and conditional
        # heights are then computed over the remaining units only.
        joined = joined[(joined["x"] != "none") & (joined["y"] != "none")]

    total = len(joined)
    if total == 0:
        raise ValueError("no units to plot after aligning the event series")

    # Keep only labels that actually survived the inner-join / dropna.
    present_x = set(joined["x"].unique())
    present_y = set(joined["y"].unique())
    x_cells = _ordered_cells(
        {k: v for k, v in x_label_indices.items() if k in present_x}
    )
    y_cells = _ordered_cells(
        {k: v for k, v in y_label_indices.items() if k in present_y}
    )

    if color_sequence is None:
        color_sequence = px.colors.qualitative.Plotly
    color_map = {
        yc: color_sequence[i % len(color_sequence)] for i, yc in enumerate(y_cells)
    }

    counts = (
        joined.groupby(["x", "y"]).size().rename("count").reset_index()
    )
    count_lookup = {
        (r["x"], r["y"]): int(r["count"]) for _, r in counts.iterrows()
    }
    col_totals = joined.groupby("x").size()

    grand = total if normalize else 1.0

    fig = go.Figure()

    # ------------------------------------------------------------------ layout
    # Walk the columns left to right. Each column width is proportional to its
    # share of all units; within it, stack the Y segments bottom to top.
    x_cursor = 0.0
    x_tick_pos: list[float] = []
    x_tick_txt: list[str] = []
    # Y-axis category labels are taken from the first (leftmost) column's
    # segment layout, so the Y partition names replace the legend.
    y_tick_pos: list[float] = []
    y_tick_txt: list[str] = []
    first_col = True
    legend_seen: set[str] = set()

    for xc in x_cells:
        col_n = int(col_totals.get(xc, 0))
        if col_n == 0:
            continue
        width = col_n / grand
        x0 = x_cursor
        x1 = x_cursor + width
        x_tick_pos.append((x0 + x1) / 2)
        # X label: cell name, overall share (column width) and count.
        x_tick_txt.append(f"{xc}<br>{col_n / total:.1%}<br>(n={col_n})")

        y_cursor = 0.0
        for yc in y_cells:
            n = count_lookup.get((xc, yc), 0)
            if n == 0:
                continue
            seg_h = n / col_n  # conditional height within the column (0..1)
            y0 = y_cursor
            y1 = y_cursor + seg_h
            y_cursor = y1

            if first_col:
                y_tick_pos.append((y0 + y1) / 2)
                y_tick_txt.append(yc)

            # Apply cosmetic gaps inside the tile (never changes the data).
            gx0, gx1 = x0 + gap / 2, x1 - gap / 2
            gy0, gy1 = y0 + gap / 2, y1 - gap / 2

            cond_frac = n / col_n
            overall_frac = n / total
            show_legend = legend and yc not in legend_seen
            legend_seen.add(yc)

            fig.add_trace(
                go.Scatter(
                    x=[gx0, gx1, gx1, gx0, gx0],
                    y=[gy0, gy0, gy1, gy1, gy0],
                    fill="toself",
                    mode="lines",
                    line=dict(width=0),
                    fillcolor=color_map[yc],
                    name=yc,
                    legendgroup=yc,
                    showlegend=show_legend,
                    hoveron="fills",
                    text=(
                        f"X: {xc}<br>Y: {yc}<br>"
                        f"count: {n}<br>"
                        f"within column: {cond_frac:.1%}<br>"
                        f"of all units: {overall_frac:.1%}"
                    ),
                    hoverinfo="text",
                )
            )

            # Label the segment when it is tall and wide enough to read.
            if seg_h > label_min_height and width > label_min_width:
                fig.add_annotation(
                    x=(gx0 + gx1) / 2,
                    y=(gy0 + gy1) / 2,
                    text=f"{n}<br>{cond_frac:.0%}",
                    showarrow=False,
                    font=dict(size=10, color="white"),
                )

        first_col = False
        x_cursor = x1

    x_max = x_cursor if x_cursor > 0 else 1.0

    fig.update_layout(
        title=title or "Mosaic diagram",
        xaxis=dict(
            title=x_title,
            range=[0, x_max],
            tickmode="array",
            tickvals=x_tick_pos,
            ticktext=x_tick_txt,
            showgrid=False,
            zeroline=False,
        ),
        # Secondary top axis: cumulative percentage markers across the width.
        xaxis2=dict(
            overlaying="x",
            side="top",
            range=[0, 100],
            tickmode="array",
            tickvals=list(range(0, 101, 10)),
            ticktext=[f"{v}%" for v in range(0, 101, 10)],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            range=[0, 1],
            tickmode="array",
            tickvals=y_tick_pos,
            ticktext=y_tick_txt,
            showgrid=False,
            zeroline=False,
        ),
        # Secondary right axis: percentage markers for the (conditional) height.
        yaxis2=dict(
            overlaying="y",
            side="right",
            range=[0, 100],
            tickmode="array",
            tickvals=list(range(0, 101, 10)),
            ticktext=[f"{v}%" for v in range(0, 101, 10)],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
        showlegend=legend,
        legend_title_text="Y partition",
        plot_bgcolor="white",
        bargap=0,
        height=1000,
    )

    if horizontal_line_at: fig.add_hline(y=horizontal_line_at, line_dash='dash', line_color='black')
    if vertical_line_at:   fig.add_vline(x=vertical_line_at,   line_dash='dash', line_color='black')

    # Anchor invisible traces to the secondary axes so the 0-100% percentage
    # markers on the top (x2) and right (y2) actually render.
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 0],
            xaxis="x2",
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, 100],
            yaxis="y2",
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    return fig


def bar(
    y_event: Event,
    x_events: Sequence[Event],
    *,
    title: str | None = None,
    x_title: str = "",
    y_title: str = "",
    color_sequence: Sequence[str] | None = None,
    x_exclusive: bool = False,
    horizontal_line_at: float = 0,
    label_bars: bool = True,
    ignore_none: bool = False,
) -> go.Figure:
    """Bar chart of ``P(y_event | x_cell)`` for every X partition cell.

    This is the one-dimensional companion to :func:`mosaic`. Instead of
    splitting each column into a full Y partition, every bar simply reports the
    *conditional share* of a single ``y_event`` within that X cell. The y-axis
    is a percentage (``0``--``100%``), exactly like the within-column heights in
    :func:`mosaic`.

    Parameters
    ----------
    y_event
        The single event whose conditional frequency is plotted. Each bar's
        height is ``count(y_event & x_cell) / count(x_cell)``.
    x_events
        Events defining the **X** partitioning, turned into mutually exclusive
        cells with the same :func:`_partition` logic as :func:`mosaic`. Overlaps
        between the supplied events become their own bar (unless
        ``x_exclusive``).
    title, x_title, y_title
        Figure / axis titles. ``y_title`` defaults to the ``y_event``
        description when left empty.
    color_sequence
        Colours cycled across the bars. Defaults to the ``plotly`` qualitative
        palette.
    x_exclusive
        Skip the cross-section (combination) building for the X axis. Set when
        the supplied events are known to be mutually exclusive, avoiding the
        ``2**N`` combination enumeration. See :func:`mosaic` for details.
    horizontal_line_at
        Draw a dashed horizontal reference line at this fraction (e.g. ``0.5``).
    label_bars
        When ``True`` (default) annotate each bar with its percentage and the
        underlying counts.
    ignore_none
        When ``True``, units whose X partition falls in the catch-all
        ``'none'`` cell are dropped before plotting, so no ``'none'`` bar is
        shown and the conditional shares are computed over the remaining units.

    Returns
    -------
    plotly.graph_objects.Figure
        A bar chart. Hovering a bar reports the X cell, the conditional
        percentage and the ``n_y / n_cell`` counts.
    """
    # Build the exclusive X partition exactly like the mosaic does.
    x_df, x_label_indices = _partition(x_events, exclusive=x_exclusive)
    x_part = x_df["cell"]

    # Align the single y_event onto the same index as a boolean series.
    y_bool = y_event.data.reindex(x_part.index).fillna(False).astype(bool)

    joined = pd.DataFrame({"x": x_part, "y": y_bool}).dropna()

    if ignore_none:
        joined = joined[joined["x"] != "none"]

    total = len(joined)
    if total == 0:
        raise ValueError("no units to plot after aligning the event series")

    present_x = set(joined["x"].unique())
    x_cells = _ordered_cells(
        {k: v for k, v in x_label_indices.items() if k in present_x}
    )

    if color_sequence is None:
        color_sequence = px.colors.qualitative.Plotly

    col_totals = joined.groupby("x").size()
    y_counts = joined[joined["y"]].groupby("x").size()

    xs: list[str] = []
    heights: list[float] = []
    colors: list[str] = []
    texts: list[str] = []
    hovertexts: list[str] = []

    for i, xc in enumerate(x_cells):
        n_cell = int(col_totals.get(xc, 0))
        if n_cell == 0:
            continue
        n_y = int(y_counts.get(xc, 0))
        frac = n_y / n_cell  # conditional share within the X cell (0..1)

        xs.append(xc)
        heights.append(frac)
        colors.append(color_sequence[i % len(color_sequence)])
        texts.append(f"{frac:.0%}<br>({n_y}/{n_cell})" if label_bars else "")
        hovertexts.append(
            f"X: {xc}<br>"
            f"{y_event.desc}: {frac:.1%}<br>"
            f"count: {n_y}/{n_cell}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=xs,
            y=heights,
            marker_color=colors,
            text=texts,
            textposition="outside",
            hovertext=hovertexts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title or f"Share of [{y_event.desc}] per X cell",
        xaxis=dict(
            title=x_title,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title or y_event.desc,
            range=[0, 1],
            tickmode="array",
            tickvals=[v / 100 for v in range(0, 101, 10)],
            ticktext=[f"{v}%" for v in range(0, 101, 10)],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
        plot_bgcolor="white",
        bargap=0.2,
        height=1000,
    )

    if horizontal_line_at:
        fig.add_hline(y=horizontal_line_at, line_dash="dash", line_color="black")

    return fig


# --------------------------------------------------------------------------- #
# Pairwise association matrix
# --------------------------------------------------------------------------- #

def _align_event_matrix(events: Sequence[Event]) -> tuple[np.ndarray, list[str]]:
    """Stack a list of events into an aligned boolean ``(n_units, n_events)`` array.

    Returns the array and the list of event descriptions used as axis labels.
    Raises if fewer than two events are supplied or no units remain.

    Alignment strategy
    ------------------
    Many events here are views over the same ``COMPLETE_DF`` (e.g.
    ``DECREASING @ INSERTED_ROW @ BOTTOM``). Their ``.data`` series therefore
    share an index that is **not unique** (``COMPLETE_DF`` has many rows per
    unit). Label-based alignment (``pd.concat(join=...)``) raises
    ``cannot reindex on an axis with duplicate labels`` on such indices.

    So we align **positionally** whenever every event exposes the same number of
    rows *and* an identical index object/content -- which is exactly the common
    ``COMPLETE_DF`` case, and keeps every row. Only when the indices genuinely
    differ (different lengths or contents, i.e. events from unrelated series) do
    we fall back to a label-based inner join, which both aligns the units and
    requires a unique index to be meaningful.
    """
    if len(events) < 2:
        raise ValueError("at least two events are required for a pairwise matrix")

    labels = [ev.desc for ev in events]
    series = [ev.data for ev in events]

    same_length = len({len(s) for s in series}) == 1
    first_index = series[0].index
    same_index = same_length and all(s.index.equals(first_index) for s in series[1:])

    if same_index:
        # Positional stack: already row-aligned, works with duplicate indices.
        mat = np.column_stack([s.to_numpy() for s in series]).astype(bool)
    else:
        # Fall back to label-based inner join (requires a unique index).
        frame = pd.concat(
            [s.rename(i) for i, s in enumerate(series)], axis=1, join="inner"
        )
        if frame.empty:
            raise ValueError("no overlapping units across the supplied events")
        mat = frame.astype(bool).to_numpy()

    if mat.shape[0] == 0:
        raise ValueError("no units to associate across the supplied events")
    return mat, labels


def _pairwise_phi(mat: np.ndarray) -> np.ndarray:
    """Mean-square-contingency (phi) coefficient between every pair of columns.

    Phi is the Pearson correlation of two binary variables and lies in
    ``[-1, 1]``: ``+1`` perfect co-occurrence, ``-1`` perfect mutual exclusion,
    ``0`` independence. Computed as the column-correlation of the 0/1 matrix.
    Columns with zero variance (always-True / always-False events) yield ``NaN``
    for every pair they take part in, since phi is undefined there.
    """
    x = mat.astype(float)
    n = x.shape[0]
    means = x.mean(axis=0)
    centered = x - means
    cov = centered.T @ centered / n
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.where(denom > 0, cov / denom, np.nan)
    return phi


def _pairwise_jaccard(mat: np.ndarray) -> np.ndarray:
    """Jaccard index ``|A & B| / |A U B|`` between every pair of columns.

    Lies in ``[0, 1]``: ``1`` identical membership, ``0`` disjoint. A pair where
    both events are empty (never True) has an undefined union and yields
    ``NaN``. The diagonal is ``1`` by definition.
    """
    x = mat.astype(float)
    inter = x.T @ x  # |A & B|
    col_sums = x.sum(axis=0)
    union = col_sums[:, None] + col_sums[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, inter / union, np.nan)
    return jac


def _pairwise_mutual_information(mat: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Mutual information (in bits) between every pair of binary columns.

    Uses the 2x2 contingency table of each pair. When ``normalized`` is ``True``
    (default) the MI is divided by the joint entropy, giving a value in
    ``[0, 1]`` that is comparable across pairs (0 independent, 1 perfectly
    co-determined). When ``False`` the raw MI in bits is returned.

    Note MI is non-negative and symmetric, so it captures *strength* of
    association but not its *direction* (unlike phi).
    """
    x = mat.astype(float)
    n = x.shape[0]
    k = x.shape[1]
    out = np.zeros((k, k), dtype=float)

    def _xlogx(p: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = p * np.log2(p)
        return np.where(p > 0, r, 0.0)

    for i in range(k):
        for j in range(i, k):
            a = x[:, i]
            b = x[:, j]
            # Joint counts of the 2x2 table.
            n11 = np.sum((a == 1) & (b == 1))
            n10 = np.sum((a == 1) & (b == 0))
            n01 = np.sum((a == 0) & (b == 1))
            n00 = np.sum((a == 0) & (b == 0))
            joint = np.array([n11, n10, n01, n00], dtype=float) / n
            pa = np.array([n11 + n10, n01 + n00], dtype=float) / n  # P(A=1), P(A=0)
            pb = np.array([n11 + n01, n10 + n00], dtype=float) / n  # P(B=1), P(B=0)

            # H(A) + H(B) - H(A,B)
            h_a = -np.sum(_xlogx(pa))
            h_b = -np.sum(_xlogx(pb))
            h_ab = -np.sum(_xlogx(joint))
            mi = h_a + h_b - h_ab
            mi = max(mi, 0.0)  # clamp tiny negative float noise

            if normalized:
                val = mi / h_ab if h_ab > 0 else (1.0 if mi > 0 else 0.0)
            else:
                val = mi

            out[i, j] = val
            out[j, i] = val
    return out


_ASSOCIATION_METHODS = {
    "phi": (_pairwise_phi, -1.0, 1.0, "RdBu_r"),
    "jaccard": (_pairwise_jaccard, 0.0, 1.0, "Blues"),
    "mutual_information": (_pairwise_mutual_information, 0.0, 1.0, "Viridis"),
}


def association_matrix(
    events: Sequence[Event],
    *,
    method: Literal['phi', 'jaccard', 'mutual_information'] = "phi",
    title: str | None = None,
    labels: Sequence[str] | None = None,
    color_scale: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    show_values: bool = True,
    normalized_mi: bool = True,
) -> go.Figure:
    """Pairwise association heatmap across a list of events.

    Computes a symmetric ``n_events x n_events`` association matrix and renders
    it as a heatmap, so you can see which events "travel together" -- e.g.
    whether ``INCREASING@TOP`` co-occurs with ``INCREASING@BOTTOM``.

    All events are aligned on their shared index first, so only units present in
    every event are counted.

    Parameters
    ----------
    events
        The events to cross-associate (tendency / transition / granularity
        events, etc.). At least two are required.
    method
        Association measure. One of:

        * ``'phi'`` (default) -- the mean-square-contingency (phi) coefficient,
          i.e. Pearson correlation of the two binary indicators, in ``[-1, 1]``.
          Signed: positive = co-occur, negative = mutually exclusive.
        * ``'jaccard'`` -- Jaccard index ``|A & B| / |A U B|`` in ``[0, 1]``.
          Ignores the jointly-absent cell, so it focuses on co-presence.
        * ``'mutual_information'`` -- MI from the 2x2 table (normalized to
          ``[0, 1]`` by joint entropy when ``normalized_mi`` is ``True``, else
          raw bits). Unsigned strength of association.
    title
        Figure title. Defaults to the method name.
    labels
        Axis tick labels. Defaults to each event's ``.desc``.
    color_scale
        Plotly colour scale name. Defaults to a per-method sensible choice
        (diverging ``RdBu_r`` for the signed phi, sequential otherwise).
    zmin, zmax
        Colour-scale limits. Default to the method's natural range.
    show_values
        Annotate each cell with its rounded value.
    normalized_mi
        Only used when ``method='mutual_information'``; see above.

    Returns
    -------
    plotly.graph_objects.Figure
        A symmetric heatmap of the pairwise association values.
    """
    if method not in _ASSOCIATION_METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_ASSOCIATION_METHODS)}"
        )

    mat, auto_labels = _align_event_matrix(events)
    tick_labels = list(labels) if labels is not None else auto_labels

    func, default_min, default_max, default_scale = _ASSOCIATION_METHODS[method]
    if method == "mutual_information":
        values = func(mat, normalized=normalized_mi)
        if not normalized_mi:
            default_min, default_max = None, None  # raw bits: let plotly autoscale
    else:
        values = func(mat)

    z = values
    text = (
        [[("" if np.isnan(v) else f"{v:.2f}") for v in row] for row in z]
        if show_values
        else None
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=tick_labels,
            y=tick_labels,
            colorscale=color_scale or default_scale,
            zmin=zmin if zmin is not None else default_min,
            zmax=zmax if zmax is not None else default_max,
            text=text,
            texttemplate="%{text}" if show_values else None,
            hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.3f}<extra></extra>",
            colorbar=dict(title=method),
        )
    )
    fig.update_layout(
        title=title or f"Pairwise association ({method})",
        xaxis=dict(tickangle=45, constrain="domain"),
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        plot_bgcolor="white",
        height=900,
    )
    return fig


# --------------------------------------------------------------------------- #
# UpSet plot
# --------------------------------------------------------------------------- #

def upset(
    events: Sequence[Event],
    *,
    exclusive: bool = False,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    sort_by: Literal["size", "degree"] = "size",
    sort_ascending: bool = False,
    min_size: int = 1,
    include_none: bool = False,
    show_set_sizes: bool = True,
    show_percentages: bool = True,
    bar_color: str = "#3b4cc0",
    dot_color: str = "#3b4cc0",
    dot_empty_color: str = "rgba(0,0,0,0.12)",
    height: int = 800,
) -> go.Figure:
    """UpSet plot of the exclusive intersections across a list of events.

    An UpSet plot is the scalable alternative to a Venn diagram for more than
    three overlapping sets. It pairs a **bar chart of intersection sizes** (how
    many units fall in each exclusive combination of events) with a **dot
    matrix** underneath showing *which* events each bar combines -- a filled dot
    means the event is part of that intersection, an empty dot means it is not.

    This consumes exactly the ``label_to_indices`` combination structure built
    by :func:`_partition`: every occurring combination of ``True`` events
    becomes one bar + one dot-matrix column, so the ``((A) ∩ (B)) ∩ (C)`` labels
    that clutter a mosaic are replaced by readable dot columns.

    Parameters
    ----------
    events
        The events to intersect. Their exclusive combinations (via
        :func:`_partition`) form the bars.
    exclusive
        Passed through to :func:`_partition`. Leave ``False`` (default) to let
        events overlap and form combination columns; set ``True`` only when the
        events are known to be mutually exclusive (then every bar has degree 1).
    title
        Figure title.
    labels
        Row labels for the dot matrix (one per event). Defaults to each event's
        ``.desc``.
    sort_by
        ``'size'`` (default) orders bars by intersection size; ``'degree'``
        orders by the number of events in the combination (then by size).
    sort_ascending
        Reverse the sort. Default ``False`` (largest / highest-degree first).
    min_size
        Drop combinations with fewer than this many units. Default ``1`` hides
        empty combinations only; set ``0`` to keep everything that occurs.
    include_none
        Keep the catch-all ``'none'`` combination (units matching no event).
        Default ``False`` drops it, since it has degree 0 and no dots.
    show_set_sizes
        Draw the per-event total-size bars on the left. Default ``True``.
    show_percentages
        Annotate the intersection bars with their share of all plotted units.
    bar_color, dot_color, dot_empty_color
        Cosmetic colours for the intersection bars, filled dots and empty dots.
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        A composed UpSet figure (intersection bars + dot matrix, plus optional
        set-size bars).
    """
    # Reuse the mosaic's exclusive-combination partition machinery.
    part_df, label_to_indices = _partition(events, exclusive=exclusive)
    cell = part_df["cell"]

    n_events = len(events)
    row_labels = list(labels) if labels is not None else [ev.desc for ev in events]
    if len(row_labels) != n_events:
        raise ValueError("labels must have exactly one entry per event")

    # Count units per exclusive combination.
    counts = cell.value_counts()

    # Build the list of combinations to draw: (label, member-index tuple, size).
    combos: list[tuple[str, tuple[int, ...], int]] = []
    for lbl, indices in label_to_indices.items():
        if not include_none and len(indices) == 0:
            continue
        size = int(counts.get(lbl, 0))
        if size < min_size:
            continue
        combos.append((lbl, indices, size))

    if not combos:
        raise ValueError("no combinations to plot (check min_size / events)")

    # Sort the columns.
    if sort_by == "size":
        key = lambda c: (c[2], len(c[1]))
    elif sort_by == "degree":
        key = lambda c: (len(c[1]), c[2])
    else:
        raise ValueError("sort_by must be 'size' or 'degree'")
    combos.sort(key=key, reverse=not sort_ascending)

    combo_labels = [c[0] for c in combos]
    combo_indices = [c[1] for c in combos]
    combo_sizes = [c[2] for c in combos]
    n_cols = len(combos)
    total_plotted = sum(combo_sizes)

    # Per-event (set) sizes over the same partitioned units.
    set_sizes = [int(ev.data.astype(bool).sum()) for ev in events]

    # ------------------------------------------------------------------ layout
    # Two columns: left = set-size bars (optional), right = the main UpSet.
    # Two rows: top = intersection-size bars, bottom = dot matrix.
    if show_set_sizes:
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.22, 0.78],
            row_heights=[0.6, 0.4],
            horizontal_spacing=0.04,
            vertical_spacing=0.04,
            shared_xaxes=False,
            specs=[
                [None, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )
        main_bar_rc = (1, 2)
        matrix_rc = (2, 2)
        setbar_rc = (2, 1)
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.04,
            specs=[[{"type": "bar"}], [{"type": "scatter"}]],
        )
        main_bar_rc = (1, 1)
        matrix_rc = (2, 1)
        setbar_rc = None

    x_pos = list(range(n_cols))

    # --- intersection-size bars (top) ---
    bar_text = None
    if show_percentages and total_plotted > 0:
        bar_text = [f"{s}<br>{s / total_plotted:.1%}" for s in combo_sizes]
    else:
        bar_text = [str(s) for s in combo_sizes]

    fig.add_trace(
        go.Bar(
            x=x_pos,
            y=combo_sizes,
            marker_color=bar_color,
            text=bar_text,
            textposition="outside",
            hovertext=[
                f"{lbl}<br>size: {s}" for lbl, s in zip(combo_labels, combo_sizes)
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=main_bar_rc[0],
        col=main_bar_rc[1],
    )

    # --- dot matrix (bottom) ---
    # Background: every (event-row, column) gets an "empty" grey dot; filled
    # dots and the connecting line are drawn on top for member events.
    bg_x, bg_y = [], []
    for col in range(n_cols):
        for row in range(n_events):
            bg_x.append(col)
            bg_y.append(row)
    fig.add_trace(
        go.Scatter(
            x=bg_x,
            y=bg_y,
            mode="markers",
            marker=dict(size=14, color=dot_empty_color),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=matrix_rc[0],
        col=matrix_rc[1],
    )

    for col, (lbl, indices) in enumerate(zip(combo_labels, combo_indices)):
        if not indices:
            continue
        ys = sorted(indices)
        # Connecting vertical line between the lowest and highest member dots.
        if len(ys) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[col, col],
                    y=[ys[0], ys[-1]],
                    mode="lines",
                    line=dict(color=dot_color, width=3),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=matrix_rc[0],
                col=matrix_rc[1],
            )
        fig.add_trace(
            go.Scatter(
                x=[col] * len(ys),
                y=ys,
                mode="markers",
                marker=dict(size=14, color=dot_color),
                hovertext=[f"{lbl}<br>{row_labels[i]}" for i in ys],
                hoverinfo="text",
                showlegend=False,
            ),
            row=matrix_rc[0],
            col=matrix_rc[1],
        )

    # --- set-size bars (left, horizontal) ---
    if setbar_rc is not None:
        fig.add_trace(
            go.Bar(
                x=set_sizes,
                y=list(range(n_events)),
                orientation="h",
                marker_color="rgba(0,0,0,0.45)",
                text=[str(s) for s in set_sizes],
                textposition="auto",
                hovertext=[
                    f"{row_labels[i]}<br>total: {set_sizes[i]}"
                    for i in range(n_events)
                ],
                hoverinfo="text",
                showlegend=False,
            ),
            row=setbar_rc[0],
            col=setbar_rc[1],
        )

    # ------------------------------------------------------------------ axes
    # Top intersection-size bars: hide x ticks (aligned with matrix below).
    fig.update_xaxes(
        showticklabels=False,
        range=[-0.5, n_cols - 0.5],
        row=main_bar_rc[0],
        col=main_bar_rc[1],
    )
    fig.update_yaxes(
        title_text="Intersection size",
        rangemode="tozero",
        row=main_bar_rc[0],
        col=main_bar_rc[1],
    )

    # Dot matrix axes: category labels on y (events), no x labels.
    fig.update_xaxes(
        showticklabels=False,
        range=[-0.5, n_cols - 0.5],
        showgrid=False,
        zeroline=False,
        row=matrix_rc[0],
        col=matrix_rc[1],
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_events)),
        ticktext=row_labels,
        range=[-0.5, n_events - 0.5],
        showgrid=False,
        zeroline=False,
        row=matrix_rc[0],
        col=matrix_rc[1],
    )

    if setbar_rc is not None:
        # Set-size bars share the event rows; reverse x so bars grow leftward.
        fig.update_xaxes(
            title_text="Set size",
            autorange="reversed",
            row=setbar_rc[0],
            col=setbar_rc[1],
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(n_events)),
            ticktext=["" for _ in range(n_events)],  # labels live on the matrix
            range=[-0.5, n_events - 0.5],
            showgrid=False,
            zeroline=False,
            row=setbar_rc[0],
            col=setbar_rc[1],
        )

    fig.update_layout(
        title=title or "UpSet plot",
        plot_bgcolor="white",
        height=height,
        bargap=0.2,
    )
    return fig


# --------------------------------------------------------------------------- #
# P(event) vs alpha threshold curve
# --------------------------------------------------------------------------- #

def prob_vs_alpha(
    events: Event | Sequence[Event],
    *,
    alphas: Sequence[float] | None = None,
    max_alpha: float = 0.25,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    x_title: str = "significance threshold α",
    y_title: str = "P(event | α)",
    color_sequence: Sequence[str] | None = None,
    mark_default_alpha: bool = True,
    log_x: bool = False,
    height: int = 600,
) -> go.Figure:
    """Sweep the significance threshold α and plot ``P(event | α)`` for each event.

    For a single significant-direction leaf (``p < α``), ``P(event | α)`` is the
    fraction of units with ``p < α`` -- i.e. **exactly the ECDF of that event's
    p-value vector**. Reading the curve against the arbitrary ``α = 0.05`` shows
    how robust a "fraction significant" conclusion is to the chosen threshold:
    a curve that is flat around 0.05 means the conclusion barely depends on it,
    a steep curve means it is threshold-sensitive.

    Because the value is obtained through :meth:`Event.decision_at`, this works
    unchanged for composite events too (``A & B``, ``A | B``, ``~A``,
    ``A.given(B)``) -- the decision is recomputed at each α and averaged, which
    correctly handles negation / mixed-direction (even non-monotone) curves that
    no single p-value vector could represent.

    Parameters
    ----------
    events
        A single :class:`~python.full_view.Event` or a sequence of them. Each
        becomes one curve.
    alphas
        The α grid to sweep. Defaults to 200 points in ``[1e-4, max_alpha]``
        (log-spaced when ``log_x`` is set, else linear). When given explicitly,
        values above ``max_alpha`` are dropped.
    max_alpha
        Upper bound of the swept α range (default ``0.25``). Larger values make
        no sense as a significance level, so the curve is not drawn past it.
    labels
        Legend labels, one per event. Defaults to each event's ``.desc``.
    title, x_title, y_title
        Figure / axis titles.
    color_sequence
        Line colours. Defaults to the plotly qualitative palette.
    mark_default_alpha
        Draw a dashed vertical line at the module default
        :data:`python.full_view.ALPHA` (the 0.05 you are testing robustness to).
    log_x
        Use a log-scaled α axis, handy when most p-values are tiny.
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        One ``P(event | α)`` curve per event over the α grid.
    """
    if isinstance(events, Event):
        events = [events]
    events = list(events)
    if not events:
        raise ValueError("at least one event is required")

    if labels is None:
        labels = [ev.desc for ev in events]
    elif len(labels) != len(events):
        raise ValueError("labels must have one entry per event")

    if max_alpha <= 0:
        raise ValueError("max_alpha must be positive")

    if alphas is None:
        if log_x:
            grid = np.logspace(-4, np.log10(max_alpha), 200)
        else:
            grid = np.linspace(1e-4, max_alpha, 200)
    else:
        grid = np.asarray(sorted(a for a in alphas if a <= max_alpha), dtype=float)
    if grid.size == 0:
        raise ValueError("alphas must be non-empty (and contain values <= max_alpha)")

    if color_sequence is None:
        color_sequence = px.colors.qualitative.Plotly

    fig = go.Figure()
    for i, (ev, lbl) in enumerate(zip(events, labels)):
        # P(event | α) = mean of the per-unit decision at that threshold.
        # NaN-safe mean: units that can't be decided (NaN p-value -> False in
        # both comparisons) simply never count as the event, matching .data.
        ys = [float(ev.decision_at(a).mean()) for a in grid]
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=ys,
                mode="lines",
                name=lbl,
                line=dict(color=color_sequence[i % len(color_sequence)], width=2),
                hovertemplate="α=%{x:.4g}<br>P=%{y:.3f}<extra>" + lbl + "</extra>",
            )
        )

    if mark_default_alpha:
        default_alpha = getattr(_fv, "ALPHA", 0.05)
        fig.add_vline(
            x=default_alpha,
            line_dash="dash",
            line_color="black",
            annotation_text=f"α = {default_alpha:g}",
            annotation_position="top",
        )

    fig.update_layout(
        title=title or "P(event) vs significance threshold",
        xaxis=dict(
            title=x_title,
            type="log" if log_x else "linear",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            title=y_title,
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
        ),
        plot_bgcolor="white",
        height=height,
        legend_title_text="event",
    )
    return fig


# --------------------------------------------------------------------------- #
# Corr(event, event) vs alpha threshold curve
# --------------------------------------------------------------------------- #

def corr_vs_alpha(
    pairs: tuple[Event, Event] | Sequence[tuple[Event, Event]],
    *,
    alphas: Sequence[float] | None = None,
    max_alpha: float = 0.25,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    x_title: str = "significance threshold α",
    y_title: str = "corr(A, B | α)  (phi)",
    color_sequence: Sequence[str] | None = None,
    mark_default_alpha: bool = True,
    log_x: bool = False,
    height: int = 600,
) -> go.Figure:
    """Sweep α and plot the phi correlation between two events for each pair.

    This is the :class:`~python.full_view.Corr` analogue of
    :func:`prob_vs_alpha`. :class:`Corr` reports the phi coefficient between two
    events' decisions *at the default* :data:`python.full_view.ALPHA`; this
    function recomputes that same phi at every α on a grid, so you can see how
    the association between two events (e.g. ``incr@top`` and ``incr@bottom``)
    depends on the arbitrary 0.05 threshold.

    phi is the Pearson correlation of the two binary decision vectors (identical
    to what :class:`Corr` computes), in ``[-1, 1]``. It is computed positionally
    on the aligned boolean arrays, so a non-unique ``COMPLETE_DF`` index does not
    distort it.

    Parameters
    ----------
    pairs
        A single ``(A, B)`` pair of :class:`Event` objects, or a sequence of
        such pairs. Each pair becomes one curve.
    alphas, max_alpha, labels, title, x_title, y_title, color_sequence,
    mark_default_alpha, log_x, height
        As in :func:`prob_vs_alpha`. ``labels`` default to
        ``corr(A.desc, B.desc)`` for each pair, and values above ``max_alpha``
        are dropped from an explicit ``alphas``.

    Returns
    -------
    plotly.graph_objects.Figure
        One phi-vs-α curve per pair. ``NaN`` phi (a constant decision at some α,
        where correlation is undefined) leaves a gap in the line.
    """
    # Normalise to a list of pairs. A bare 2-tuple of Events is one pair.
    if (
        isinstance(pairs, tuple)
        and len(pairs) == 2
        and all(isinstance(e, Event) for e in pairs)
    ):
        pairs = [pairs]
    pairs = [tuple(p) for p in pairs]
    if not pairs:
        raise ValueError("at least one (A, B) pair is required")
    for p in pairs:
        if len(p) != 2 or not all(isinstance(e, Event) for e in p):
            raise ValueError("each pair must be a 2-tuple of Event objects")

    if labels is None:
        labels = [f"corr({a.desc} , {b.desc})" for a, b in pairs]
    elif len(labels) != len(pairs):
        raise ValueError("labels must have one entry per pair")

    if max_alpha <= 0:
        raise ValueError("max_alpha must be positive")

    if alphas is None:
        if log_x:
            grid = np.logspace(-4, np.log10(max_alpha), 200)
        else:
            grid = np.linspace(1e-4, max_alpha, 200)
    else:
        grid = np.asarray(sorted(a for a in alphas if a <= max_alpha), dtype=float)
    if grid.size == 0:
        raise ValueError("alphas must be non-empty (and contain values <= max_alpha)")

    if color_sequence is None:
        color_sequence = px.colors.qualitative.Plotly

    def _phi(a_bool: np.ndarray, b_bool: np.ndarray) -> float:
        # Pearson correlation of two binary vectors == phi coefficient.
        # Undefined (NaN) when either vector is constant.
        a = a_bool.astype(float)
        b = b_bool.astype(float)
        sa = a.std()
        sb = b.std()
        if sa == 0 or sb == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    fig = go.Figure()
    for i, ((a_ev, b_ev), lbl) in enumerate(zip(pairs, labels)):
        ys = []
        for alpha in grid:
            a_bool = np.asarray(a_ev.decision_at(alpha))
            b_bool = np.asarray(b_ev.decision_at(alpha))
            ys.append(_phi(a_bool, b_bool))
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=ys,
                mode="lines",
                name=lbl,
                connectgaps=False,
                line=dict(color=color_sequence[i % len(color_sequence)], width=2),
                hovertemplate="α=%{x:.4g}<br>phi=%{y:.3f}<extra>" + lbl + "</extra>",
            )
        )

    if mark_default_alpha:
        default_alpha = getattr(_fv, "ALPHA", 0.05)
        fig.add_vline(
            x=default_alpha,
            line_dash="dash",
            line_color="black",
            annotation_text=f"α = {default_alpha:g}",
            annotation_position="top",
        )
    # Zero reference: no association.
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.3)")

    fig.update_layout(
        title=title or "Event correlation (phi) vs significance threshold",
        xaxis=dict(
            title=x_title,
            type="log" if log_x else "linear",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            title=y_title,
            range=[-1, 1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
        ),
        plot_bgcolor="white",
        height=height,
        legend_title_text="pair",
    )
    return fig


# --------------------------------------------------------------------------- #
# Combined-p-value significance grid
# --------------------------------------------------------------------------- #

def _combine_grid(
    events: Sequence[Event], methods: Sequence[CombineMethod], alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Combine every event under every method, returning (combined_p, significant).

    Both outputs are ``(n_events, n_methods)`` arrays: the raw combined p-value
    (via :class:`~python.full_view.CombinePValues`) and the boolean
    ``combined_p < alpha`` significance decision.
    """
    combined = np.empty((len(events), len(methods)), dtype=float)
    for i, ev in enumerate(events):
        for j, m in enumerate(methods):
            combined[i, j] = float(CombinePValues(m, ev))
    return combined, combined < alpha


def combine_significance(
    events: Sequence[Event],
    right_events: Sequence[Event] | None = None,
    *,
    alpha: float = 0.05,
    methods: Sequence[CombineMethod] | None = None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    right_labels: Sequence[str] | None = None,
    show_values: bool = False,
    sig_color: str = "#111111",
    nonsig_color: str = "#f0f0f0",
    quad_colors: Sequence[str] | None = None,
    height: int | None = None,
) -> go.Figure:
    """Significance grid of combined p-values per event (one or two sequences).

    This is the visual replacement for the long 120-row table in
    ``nb/m_datasets/full_table.ipynb``. That cell iterated granularity ×
    tendency × transition × dataset internally and reported, per row, the
    combined p-value under all five :class:`~python.full_view.CombineMethod`
    strategies. Here the **event iteration is pulled out** of the function: the
    caller passes exactly the list(s) of events it wants plotted, and this
    function only does the "combine under every method, then threshold at
    ``alpha``" part.

    **Single-sequence mode** (``right_events=None``, the original behaviour):
    for every event (one row) and every combine method (one column) the
    per-unit p-values are pooled and compared against ``alpha``. The cell is
    drawn **``sig_color`` (black) when significant** and **``nonsig_color``
    (near-white) otherwise**, so the grid reads at a glance as a significance
    mask. Row labels sit on the left y-axis.

    **Paired mode** (``right_events`` supplied): the two sequences are zipped
    **row-wise** -- ``events[i]`` is paired with ``right_events[i]`` -- and both
    must have the same length (validated). Each cell now encodes the **four
    significance combinations** of the pair under that method with four colours:

    ======================  ====================================
    (left, right) decision  default colour
    ======================  ====================================
    both significant        green  (``#2ca02c``)
    only left significant   blue   (``#1f77b4``)
    only right significant  orange (``#ff7f0e``)
    neither significant     near-white (``#f0f0f0``)
    ======================  ====================================

    The **left y-axis** carries the ``events`` labels and the **right y-axis**
    the ``right_events`` labels, so each row shows which two events are being
    compared. A legend maps the four colours to the four combinations.

    Parameters
    ----------
    events
        The (left) events to combine and threshold, one per row. Build these
        outside (e.g. ``[(T @ Tr @ G).given(D) for ...]``) — no iteration
        happens here.
    right_events
        Optional second sequence, paired row-wise with ``events``. When given it
        **must have the same length** as ``events`` (validated); each row then
        shows the four-way significance combination of the pair. Leave ``None``
        for the original single-sequence black/white grid.
    alpha
        Significance threshold applied to each combined p-value. Default
        ``0.05``.
    methods
        Which combine methods to use as columns. Defaults to every
        :class:`CombineMethod` in enum order (fisher, stouffer, pearson,
        tippett, mudholkar_george).
    title
        Figure title. Defaults to a description including ``alpha``.
    labels
        Left-axis row labels, one per ``events`` entry. Defaults to each event's
        ``.desc``.
    right_labels
        Right-axis row labels, one per ``right_events`` entry. Only used in
        paired mode; defaults to each right event's ``.desc``.
    show_values
        When ``True`` also annotate each cell with the combined p-value(s) in
        scientific notation (left value in single mode; ``left / right`` in
        paired mode). Default ``False`` keeps the grid purely colour-coded.
    sig_color, nonsig_color
        Fill colours for the significant / non-significant cells in
        **single-sequence mode**. ``nonsig_color`` also doubles as the
        "neither significant" colour in paired mode.
    quad_colors
        Four colours for paired mode, in the order
        ``[both, only-left, only-right, neither]``. Defaults to
        ``["#2ca02c", "#1f77b4", "#ff7f0e", nonsig_color]``.
    height
        Figure height in pixels. Defaults to a value scaled to the number of
        rows so dense grids stay readable.

    Returns
    -------
    plotly.graph_objects.Figure
        A heatmap with events on the y-axis (left, plus right in paired mode)
        and combine methods on the x-axis. Hovering a cell reports the
        event(s), method, combined p-value(s) and the decision(s).
    """
    events = list(events)
    if not events:
        raise ValueError("at least one event is required")

    paired = right_events is not None
    if paired:
        right_events = list(right_events)
        if len(right_events) != len(events):
            raise ValueError(
                "events and right_events must have the same number of rows "
                f"(got {len(events)} and {len(right_events)})"
            )

    if methods is None:
        methods = list(CombineMethod)
    methods = list(methods)
    if not methods:
        raise ValueError("at least one combine method is required")
    method_labels = [str(m) for m in methods]

    row_labels = list(labels) if labels is not None else [ev.desc for ev in events]
    if len(row_labels) != len(events):
        raise ValueError("labels must have one entry per event")

    n_rows = len(events)
    n_cols = len(methods)
    # Numeric y positions let us put independent tick labels on both the left
    # (yaxis) and right (yaxis2) axes, aligned to the same rows.
    y_pos = list(range(n_rows))

    left_p, left_sig = _combine_grid(events, methods, alpha)

    if not paired:
        # ---------------------------- single-sequence (original) behaviour ---
        z = left_sig.astype(float)
        colorscale = [[0.0, nonsig_color], [1.0, sig_color]]

        hover = np.empty_like(left_p, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                decision = "significant" if left_sig[i, j] else "not significant"
                hover[i, j] = (
                    f"event: {row_labels[i]}<br>"
                    f"method: {method_labels[j]}<br>"
                    f"combined p: {left_p[i, j]:.3g}<br>"
                    f"{decision} (α={alpha:g})"
                )

        text = None
        texttemplate = None
        if show_values:
            text = [[f"{v:.1e}" for v in row] for row in left_p]
            texttemplate = "%{text}"

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=method_labels,
                y=y_pos,
                colorscale=colorscale,
                zmin=0.0,
                zmax=1.0,
                showscale=False,
                xgap=1,
                ygap=1,
                text=text,
                texttemplate=texttemplate,
                textfont=dict(size=9, color="#d33"),
                hovertext=hover,
                hoverinfo="text",
            )
        )

        n_sig = int(left_sig.sum())
        n_total = left_sig.size
        fig.update_layout(
            title=(
                title
                or f"Combined-p significance (α = {alpha:g}) — "
                   f"{n_sig}/{n_total} cells significant"
            ),
            xaxis=dict(title="combine method", side="top", tickangle=0),
            yaxis=dict(
                autorange="reversed",  # first event on top
                tickmode="array",
                tickvals=y_pos,
                ticktext=row_labels,
            ),
            plot_bgcolor="white",
            height=height if height is not None else max(400, 22 * n_rows + 160),
        )
        return fig

    # -------------------------------------------------- paired (four-colour) ---
    right_row_labels = (
        list(right_labels) if right_labels is not None
        else [ev.desc for ev in right_events]
    )
    if len(right_row_labels) != n_rows:
        raise ValueError("right_labels must have one entry per right event")

    right_p, right_sig = _combine_grid(right_events, methods, alpha)

    if quad_colors is None:
        quad_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", nonsig_color]
    quad_colors = list(quad_colors)
    if len(quad_colors) != 4:
        raise ValueError("quad_colors must have exactly four entries")

    # Encode the four combinations as 0..3:
    #   0 neither, 1 only-left, 2 only-right, 3 both.
    code = (left_sig.astype(int) * 1) + (right_sig.astype(int) * 2)
    # Map code -> palette index in [both, only-left, only-right, neither] order.
    code_to_color = {3: quad_colors[0], 1: quad_colors[1],
                     2: quad_colors[2], 0: quad_colors[3]}
    code_to_label = {3: "both significant", 1: "only left significant",
                     2: "only right significant", 0: "neither significant"}

    # Build a discrete 4-level colourscale over z in {0,1,2,3}. Each level is a
    # flat band so no interpolation colour is ever shown.
    z = code.astype(float)
    bands = [
        (0.0, 0.25, code_to_color[0]),
        (0.25, 0.50, code_to_color[1]),
        (0.50, 0.75, code_to_color[2]),
        (0.75, 1.0, code_to_color[3]),
    ]
    colorscale = []
    for lo, hi, c in bands:
        colorscale.append([lo, c])
        colorscale.append([hi, c])

    hover = np.empty_like(left_p, dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            hover[i, j] = (
                f"method: {method_labels[j]}<br>"
                f"left:  {row_labels[i]}<br>"
                f"  combined p: {left_p[i, j]:.3g}"
                f" ({'sig' if left_sig[i, j] else 'ns'})<br>"
                f"right: {right_row_labels[i]}<br>"
                f"  combined p: {right_p[i, j]:.3g}"
                f" ({'sig' if right_sig[i, j] else 'ns'})<br>"
                f"→ {code_to_label[int(code[i, j])]} (α={alpha:g})"
            )

    text = None
    texttemplate = None
    if show_values:
        text = [
            [f"{left_p[i, j]:.0e}/{right_p[i, j]:.0e}" for j in range(n_cols)]
            for i in range(n_rows)
        ]
        texttemplate = "%{text}"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=method_labels,
            y=y_pos,
            colorscale=colorscale,
            zmin=0.0,
            zmax=3.0,
            showscale=False,
            xgap=1,
            ygap=1,
            text=text,
            texttemplate=texttemplate,
            textfont=dict(size=8, color="#000"),
            hovertext=hover,
            hoverinfo="text",
        )
    )

    # Dummy legend traces so the four combinations are explained.
    for code_val in (3, 1, 2, 0):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=code_to_color[code_val], symbol="square"),
                name=code_to_label[code_val],
                showlegend=True,
                hoverinfo="skip",
            )
        )

    both = int((code == 3).sum())
    left_only = int((code == 1).sum())
    right_only = int((code == 2).sum())
    neither = int((code == 0).sum())
    fig.update_layout(
        title=(
            title
            or f"Paired combined-p significance (α = {alpha:g}) — "
               f"both {both}, left {left_only}, right {right_only}, "
               f"neither {neither}"
        ),
        xaxis=dict(title="combine method", side="top", tickangle=0),
        yaxis=dict(
            # Explicit reversed range (first pair on top). Must be explicit, not
            # autorange, so the overlaying right axis can share identical limits.
            range=[n_rows - 0.5, -0.5],
            tickmode="array",
            tickvals=y_pos,
            ticktext=row_labels,
            title="left event",
        ),
        # Right axis mirrors the same rows with the right-event labels. It uses
        # the SAME explicit range as the primary y-axis (autorange on an
        # overlaying axis would recompute from the single anchor point and drop
        # most ticks).
        yaxis2=dict(
            overlaying="y",
            side="right",
            range=[n_rows - 0.5, -0.5],
            tickmode="array",
            tickvals=y_pos,
            ticktext=right_row_labels,
            title="right event",
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="white",
        height=height if height is not None else max(400, 22 * n_rows + 160),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08,
                    xanchor="center", x=0.5),
    )

    # Anchor an invisible trace to yaxis2 so its tick labels render, aligned to
    # the same reversed row range as the primary y-axis.
    fig.add_trace(
        go.Scatter(
            x=[method_labels[0]],
            y=[0],
            yaxis="y2",
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    return fig
