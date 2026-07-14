import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Literal
import plotly.express as px
import plotly.graph_objects as go

from python.utils import is_long, long_to_wide
from python.temporal_hierarchy import TemporalHierarchy
from python.aggregation import Aggregation, DAYS, MONTHS
from python.time_series import BASE_FORECASTS, VALIDATION_SET, RECONCILED_FORECASTS, TRAIN_SET
from python.error_metrics import ErrorMetrics, MetricsIndex, MetricsFilter
from python.sampled_hierarchies import SampledHierarchies, MetricAggregateStore, MetricAggregate, StatsMatrix, IndexMatrix

def plot_nans(df: pd.DataFrame):
    '''' Optimal for wide format. GenAI code '''
    if is_long(df):
        # Long Format
        df = long_to_wide(df)

    mask = df.isna().astype(int)
    # Wide Format
    fig = px.imshow(mask, aspect='auto', title=f"Missing Values Heatmap of {len(df.columns) - 1} Series", y=df.index, color_continuous_scale=["#02FC0F", "#8D3939"])
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        coloraxis_colorbar=dict( title="Data Status", tickvals=[0, 1], ticktext=["Present", "Missing"], lenmode="pixels", len=150)
    )
    fig.update_traces(
        hovertemplate="<b>Date:</b> %{y}<br><b>Series:</b> %{x}<br><b>Status:</b> %{customdata}<extra></extra>",
        customdata=mask.replace({0: "Present", 1: "Missing"})
    )
    return fig


def plot_heatmap(df: pd.DataFrame, log = False, last_n_days: int | None = None):
    ''' Plots the heatmap of values in the time series. Optimal for wide format. Highlights NaNs.
        last_n_days: if set, only plot the last N days (rows) of the DataFrame. '''
    if is_long(df):
        df = long_to_wide(df)

    if last_n_days is not None:
        df = df.iloc[-last_n_days:]

    if log:
        df_vals = np.log(df.where(df > 0))
    else:
        df_vals = df

    fig = px.imshow(df_vals, aspect='auto', title=f"Time Series Value Distribution ({len(df.columns)} Series)", color_continuous_scale="Viridis",
                    labels=dict(x="Series", y="Date", color="Value")
    )

    # Set the background of the actual plotting area.
    fig.update_layout( height=800, plot_bgcolor='red', xaxis_showticklabels=False, yaxis=dict(showgrid=False))

    return fig


def plot_summing_matrix(S: pd.DataFrame, height: int=700):
    ''' Plots a summing matrix of a temporal hierarchy. GenAI code '''
    # 1. Clean data and extract labels

    df_numeric = S.drop(columns=['hierarchy'], errors='ignore')
    matrix = df_numeric.values
    x_labels = df_numeric.columns.astype(str).tolist()
    y_labels = df_numeric.index.astype(str).tolist()

    # 2. Configure colors and custom data for hovering
    hover_data = np.empty(matrix.shape, dtype=object)
    hover_data[:] = ""

    if 'hierarchy' in S.columns:
        hierarchies = S['hierarchy'].values
        unique_hierarchies = list(reversed(pd.unique(hierarchies)))
        N = len(unique_hierarchies)

        for i, h in enumerate(hierarchies):
            # Swap the 1s for distinct category integers (1, 2, 3...)
            idx = unique_hierarchies.index(h) + 1
            matrix[i] = np.where(matrix[i] == 1, idx, 0)
            hover_data[i] = np.where(matrix[i] > 0, h, "")

        # Get standard plotly colors
        palette = px.colors.qualitative.Plotly
        colors = ['#edf2f9'] + [palette[i % len(palette)] for i in range(N)]

        # Build the discrete colorscale blocks
        colorscale = []
        for i in range(N + 1):
            v0 = i / (N + 1)
            v1 = (i + 1) / (N + 1)
            colorscale.append([v0, colors[i]])
            colorscale.append([v1, colors[i]])

        zmin, zmax = -0.5, N + 0.5

        colorbar_dict = dict(
            title="Aggregation Level",
            tickmode='array',
            tickvals=list(range(1, N + 1)),
            ticktext=unique_hierarchies,
            len=0.8
        )
        showscale = True
        hovertemplate = "Col: %{x}<br>Row: %{y}<br>Hierarchy: %{customdata}<extra></extra>"

    else:
        # Fallback to monochrome if the hierarchy column is missing
        colorscale = [[0.0, '#edf2f9'], [1.0, 'black']]
        zmin, zmax = 0, 1
        colorbar_dict = None
        showscale = False
        hovertemplate = "Col: %{x}<br>Row: %{y}<extra></extra>"

    # 3. Create Heatmap
    fig = go.Figure(data=go.Heatmap(z=matrix, customdata=hover_data, hovertemplate=hovertemplate, colorscale=colorscale,
                                    zmin=zmin, zmax=zmax, showscale=showscale, colorbar=colorbar_dict, xgap=1, ygap=1))
    # 4. Apply labels purely as cosmetic text over the integer coordinates
    fig.update_layout(
        yaxis=dict( autorange='reversed', tickmode='array', tickvals=list(range(len(y_labels))), ticktext=y_labels, showticklabels=False),
        xaxis=dict( side='top', tickangle=-45, tickmode='array', tickvals=list(range(len(x_labels))), ticktext=x_labels),
        margin=dict(t=80),
        height=height
    )

    return fig

def plot_forecast(ths: list[TemporalHierarchy], lvls: list[Aggregation] = [DAYS, MONTHS], col: int | str = 1, history_days: int = 1):
    ''' Plots the actual, base and reconciled forecasts for the specified levels of the specified time series. GenAI code '''
    if isinstance(col, int):
        col = ths[0][lvls[0]][VALIDATION_SET].wide.columns[col]

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for th in ths:
        for i, lvl in enumerate(lvls):
            train = th[lvl][TRAIN_SET].wide[col]
            validation = th[lvl][VALIDATION_SET].wide[col]
            base = th[lvl][BASE_FORECASTS].wide[col]
            reconciled = th[lvl][RECONCILED_FORECASTS].wide[col]

            color = colors[i % len(colors)]

            if history_days > 0:
                history = train.loc[train.index.max() - timedelta(days=history_days):train.index.max()]
                actual = pd.concat([history, validation])
            else:
                actual = validation

            x_actual = actual.index
            x_forecast = validation.index

            fig.add_trace(go.Scatter(x=x_actual, y=actual, mode='lines', name=f'{th} - {lvl} - Actual', line=dict(color=color, width=2)))
            fig.add_trace(go.Scatter(x=x_forecast, y=base, mode='lines', name=f'{th} - {lvl} - Base', line=dict(color=color, dash='dot')))
            fig.add_trace(go.Scatter(x=x_forecast, y=reconciled, mode='lines', name=f'{lvl} - Reconciled', line=dict(color=color, dash='dash')))

    fig.update_layout(title=f'Forecast Comparison for Series: {col}', xaxis_title='Date', yaxis_title='Value')
    return fig

def plot_error_distribution(em: ErrorMetrics, indeces: list[MetricsIndex], title_info: str = '', outlier_threshold = 0.0, bin_size = 0.15):
    ''' Plot overlapping histograms of the selected error metrics. GenAI code. '''
    fig = go.Figure()

    for index in indeces:
        errors = index.values(em).flatten()
        if outlier_threshold:
            errors = errors[errors < np.quantile(errors, outlier_threshold)]
        fig.add_trace(go.Histogram(x=errors, name=str(index), opacity=0.6, histnorm='probability'))

    fig.update_traces(xbins_size=bin_size)
    title = 'Error Distribution' + f' - {title_info}' if title_info else ''
    fig.update_layout(barmode='overlay', title=title, xaxis_title='Error Value', yaxis_title='Frequency')
    return fig

def plot_error_qq(em: ErrorMetrics, index_x: MetricsIndex, index_y: MetricsIndex, outlier_threshold = 0.0):
    ''' Plots a QQ diagram of the selected two error metrics '''
    x = index_x.values(em).flatten()
    y = index_y.values(em).flatten()
    x.sort()
    y.sort()

    if outlier_threshold:
        x = x[x < np.quantile(x, outlier_threshold)]
        y = y[y < np.quantile(y, outlier_threshold)]

    x = np.quantile(x, np.linspace(0, 1, 100))
    y = np.quantile(y, np.linspace(0, 1, 100))

    color = np.where(y > x, 'Above', 'Below')
    fig = px.scatter(x=x, y=y, color=color, title=f'QQ-Plot ({index_x}) vs ({index_y})', labels={'x': f'{index_x}', 'y': f'{index_y}'},
                     color_discrete_map={'Above': 'Red', 'Below': 'Blue'})
    fig.add_shape(type="line", x0=min(x), y0=min(x), x1=max(x), y1=max(x), line=dict(dash='dash'))
    return fig

def plot_error_diff(em: ErrorMetrics, index_base: MetricsIndex, index_comp: MetricsIndex, outlier_threshold = 0.0):
    ''' Plots the difference of the two selected error metrics. GenAI code. '''
    base = index_base.values(em).flatten()
    comp = index_comp.values(em).flatten()

    if len(base) != len(comp):
        raise ValueError("Indeces are not comparable, they must index the same amount of rows!")

    df = pd.DataFrame({'base': base, 'comp': comp})
    if outlier_threshold:
        df = df[(df['base'] < np.quantile(df['base'], outlier_threshold)) & (df['comp'] < np.quantile(df['comp'], outlier_threshold))]
    df['diff'] = df['comp'] - df['base']
    df.sort_values('base', inplace=True)
    df['error_status'] = df['diff'] > 0

    fig = px.scatter(df, x='base', y='diff', title=f'Difference in Error: {index_comp} vs {index_base}',
                     labels={'base': f'Base Error - {index_base}', 'diff': f'Difference to {index_comp}'},
                     color='error_status',
                     color_discrete_map={True: '#EF553B', False: '#00CC96'})

    x_lines = []
    y_lines = []
    for x, y in zip(df['base'], df['diff']):
        x_lines.extend([x, x, None])    # X coordinate stays the same for a vertical line
        y_lines.extend([0, y, None])    # Y coordinates go from 0 up/down to the diff value

    fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode='lines', line=dict(color='gray', width=1.5), showlegend=False, hoverinfo='skip'))
    fig.data = fig.data[::-1]
    fig.add_hline(y=0, line_dash="dot", line_color='red')

    return fig

def plot_error_func(em: ErrorMetrics, index_x: MetricsIndex, index_y: MetricsIndex, outlier_threshold = 0.0):
    ''' Plots one error metrics in the function of another one '''
    x = index_x.values(em).flatten()
    y = index_y.values(em).flatten()

    df = pd.DataFrame({'x': x, 'y': y})
    if outlier_threshold:
        df = df[(df['x'] < np.quantile(df['x'], outlier_threshold)) & (df['y'] < np.quantile(df['y'], outlier_threshold))]
    df.sort_values('x', inplace=True)


    fig = px.scatter(df, x='x', y='y', title=f'Scatter Plot: {index_x} vs {index_y}',
                     labels={'x': f'{index_x}', 'y': f'{index_y}'},
                     trendline='ols', trendline_color_override='red')

    fig.add_shape(type="line", x0=df['x'].min(), y0=df['x'].min(), x1=df['x'].max(), y1=df['x'].max(),
                  line=dict(dash='dash'))


    return fig

def plot_error_strips(em: ErrorMetrics, index: MetricsIndex, symlog = False):
    df = index.rows(em)

    index_labels = [
        " | ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        for idx in df.index
    ]

    original_values = df.values.flatten()

    if symlog:
        plot_values = np.sign(original_values) * np.log1p(np.abs(original_values))
        y_label = 'Error Value (SymLog)'
    else:
        plot_values = original_values
        y_label = 'Error Value'

    plot_df = pd.DataFrame({
        y_label: plot_values,
        'Original Error': original_values,
        'Time Series ID': np.tile(df.columns, len(df)),
        'Metric': np.repeat(index_labels, len(df.columns))
    })

    fig = px.strip(plot_df, x='Metric', y=y_label, color='Metric', title=f'Strip Plot: {index}',
                   hover_data={ 'Time Series ID': True, 'Original Error': ':.4f', y_label: False,})

    # Optional: Clean up the layout (hiding the x-axis title since the ticks/legend explain it)
    fig.update_layout(xaxis_title=None)
    fig.update_xaxes(showticklabels=False)

    return fig


def plot_error_boxes(em: ErrorMetrics, index: MetricsIndex, symlog: bool = False):
    df = index.rows(em)

    index_labels = [
        " | ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        for idx in df.index
    ]

    original_values = df.values.flatten()

    if symlog:
        plot_values = np.sign(original_values) * np.log1p(np.abs(original_values))
        y_label = 'Error Value (SymLog)'
    else:
        plot_values = original_values
        y_label = 'Error Value'

    plot_df = pd.DataFrame({
        y_label: plot_values,
        'Original Error': original_values,
        'Time Series ID': np.tile(df.columns, len(df)),
        'Metric': np.repeat(index_labels, len(df.columns))
    })

    fig = px.box(plot_df, x='Metric', y=y_label, color='Metric', title=f'Box Plot: {index}',
                 hover_data={ 'Time Series ID': True, 'Original Error': ':.4f', y_label: False,})

    # 5. Clean up the x-axis
    fig.update_layout(xaxis_title=None)
    fig.update_xaxes(showticklabels=False)

    return fig

def plot_stats(stats_df: pd.DataFrame, stat: str, labels: tuple = (0,)):
    x = [str([i[j] for j in labels]) for i in stats_df.index]
    fig = px.bar(stats_df, x=x, y=stat, title=f'{stat} statistics')
    fig.update_layout(xaxis_title='Method', yaxis_title=stat)
    return fig

def plot_error_heatmap(em: ErrorMetrics, x_index: MetricsIndex, y_index: MetricsIndex, bins=15, outlier_threshold=0.0, log_scale=False):
    x = x_index.rows(em).values.flatten()
    y = y_index.rows(em).values.flatten()

    if outlier_threshold:
        mask = (x < np.quantile(x, outlier_threshold)) & (y < np.quantile(y, outlier_threshold))
        x = x[mask]
        y = y[mask]


    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)

    if log_scale:
        heatmap = np.log1p(heatmap)

    title_prefix = "Log scale " if log_scale else ""

    fig = go.Figure(go.Heatmap(z=heatmap.T, x=xedges, y=yedges, colorscale='Viridis'))
    fig.update_layout(title=title_prefix + f'Error Heatmap of ({x_index}) and ({y_index})',
                      xaxis_title=f'{x_index}', yaxis_title=f'{y_index}')
    return fig

def plot_error_signs(em: ErrorMetrics, x_index: MetricsIndex, y_index: MetricsIndex, percentage=False):
    x = x_index.rows(em).values.flatten()
    y = y_index.rows(em).values.flatten()

    df = pd.DataFrame({'x': x, 'y': y})
    df = df[(df['x'] != 0) & (df['y'] != 0)]
    df['sign_x'] = np.sign(df['x'])
    df['sign_y'] = np.sign(df['y'])

    counts = df.groupby(['sign_x', 'sign_y']).size().unstack(fill_value=0).astype(float)
    if percentage:
        counts.loc[:, :] = np.round(counts / len(x) * 100, decimals=2)

    fig = go.Figure(go.Heatmap(z=counts.values, x=['-', '+'], y=['-', '+'],
                               text=counts.values, texttemplate='%{text}', colorscale='Viridis'))
    fig.update_layout(xaxis_title=f'{y_index} sign', yaxis_title=f'{x_index} sign', title=f'Signs of {x_index} and {y_index}')
    return fig

def plot_error_over_inconsistency(em: ErrorMetrics, mi: MetricsIndex, inc: pd.DataFrame, outlier_threshold = 0.0):
    errors = mi.rows(em).values.flatten()
    inc_vals = inc.values.flatten()

    if outlier_threshold:
        mask = (errors < np.quantile(errors, outlier_threshold)) & (np.abs(inc_vals) < np.quantile(np.abs(inc_vals), outlier_threshold))
        errors = errors[mask]
        inc_vals = inc_vals[mask]

    fig = px.scatter(x=inc_vals, y=errors,
               title=f'{mi} Error vs inconsistency', labels={'x': f'{inc.index.tolist()} Inconsistency', 'y': f'{mi} Error'})
    return fig

def plot_sampled_error_matrix(sh: SampledHierarchies, value: str, *,
                              em: ErrorMetrics | None=None, index_matrix: IndexMatrix | None=None, stats_matrix: StatsMatrix | None=None,
                              df_filter: MetricsFilter=None,
                              accuracy=2):
    if stats_matrix is None and (em is None or index_matrix is None):
        raise ValueError('Either stats_matrix or both em and indeces need to be specified')

    all_stats = sh.error_matrix(em, index_matrix, df_filter) if stats_matrix is None else stats_matrix
    stats = np.vectorize(lambda s: s[value].item())(all_stats)
    fig = px.imshow(stats, labels={"x": "Heights", "y": "Sequence IDs", "color": "Error"},
                x=sh.heights, y=sh.group_ids,
                text_auto=f'.{accuracy}f')
    fig.update_xaxes(tickmode='array', tickvals=sh.heights)
    fig.update_yaxes(tickmode='array', tickvals=sh.group_ids)

    return fig

def plot_sampled_error_bars(em: ErrorMetrics, indeces: list[MetricsIndex], value: str, df_filter: MetricsFilter=None):
    stats = {str(index): em.get_error_stats(index, df_filter)[value].item() for index in indeces}
    fig = px.bar(x=list(stats.keys()), y=list(stats.values()), title=f'{value} statistics',
                 labels={'x': 'Index', 'y': value})
    return fig

def plot_aggregate_statistics(aggregates: MetricAggregate, title_suffix=''):
    fig = go.Figure(data=go.Bar(y=aggregates.values))
    fig.update_layout(title=f'{aggregates} {title_suffix}', xaxis_title=aggregates.group_by, yaxis_title=aggregates.stat)
    fig.update_yaxes(range=[aggregates.values.min() * 0.8, aggregates.values.max() * 1.05])
    return fig


def latex_sampled_error_table(sh: SampledHierarchies, value: str, *,
                              em: ErrorMetrics | None=None, index_matrix: IndexMatrix | None=None, stats_matrix: StatsMatrix | None=None,
                              df_filter: MetricsFilter=None,
                              caption: str = '', label: str = '', precision: int = 4) -> str:
    """Generate a copy-pasteable LaTeX table from a sampled hierarchy error matrix.
    
    Same interface as plot_sampled_error_matrix, but produces LaTeX output.
    
    Args:
        sh: SampledHierarchies object
        value: stat to extract (e.g. 'mean', 'median', 'winsorized')
        em: ErrorMetrics object (optional if stats_matrix provided)
        index_matrix: IndexMatrix (optional if stats_matrix provided)
        stats_matrix: pre-computed stats matrix (optional)
        df_filter: filter to apply to error dataframe
        caption: LaTeX table caption
        label: LaTeX table label for \\ref{}
        precision: decimal places
    
    Returns:
        Complete LaTeX table string ready to paste into paper.
    """
    if stats_matrix is None and (em is None or index_matrix is None):
        raise ValueError('Either stats_matrix or both em and index_matrix need to be specified')

    all_stats = sh.error_matrix(em, index_matrix, df_filter) if stats_matrix is None else stats_matrix
    matrix = np.vectorize(lambda s: s[value].item())(all_stats)

    n_rows, n_cols = matrix.shape
    fmt = f'.{precision}f'
    heights = sh.heights

    lines = []
    lines.append(r'\begin{table*}[h]')
    lines.append(f'\\caption{{{caption}}}\\label{{{label}}}')
    lines.append(r'\begin{tabular*}{\textwidth}{@{\extracolsep\fill}l' + 'c' * n_cols + '}')
    lines.append(r'\toprule')
    lines.append('Heights & ' + ' & '.join(str(h) for h in heights) + r' \\')
    lines.append(r'\midrule')

    for i in range(n_rows):
        vals = ' & '.join(f'{v:{fmt}}' for v in matrix[i])
        lines.append(f'Sequence {sh.group_ids[i]} & {vals} \\\\')

    lines.append(r'\midrule')
    mean_vals = ' & '.join(f'{v:{fmt}}' for v in matrix.mean(axis=0))
    std_vals = ' & '.join(f'{v:{fmt}}' for v in matrix.std(axis=0))
    lines.append(f'Mean & {mean_vals} \\\\')
    lines.append(f'Std & {std_vals} \\\\')

    lines.append(r'\botrule')
    lines.append(r'\end{tabular*}')
    lines.append(r'\end{table*}')

    result = '\n'.join(lines)
    print(result)
    return result

def latex_conformism_table(sh: SampledHierarchies, rows: dict[str, StatsMatrix], value: str,
                           trend: Literal['increase', 'decrease'], *,
                           caption: str = '', label: str = '', precision: int = 1) -> str:
    """Generate a LaTeX table showing theorem conformism percentages per group.

    Args:
        sh: SampledHierarchies object
        rows: dict mapping row labels to their StatsMatrix, e.g.
              {"Monthly (\\%)": month_stats, "Daily (\\%)": day_stats}
        value: stat to extract (e.g. 'mean', 'winsorized')
        trend: 'increase' or 'decrease'
        caption: LaTeX table caption
        label: LaTeX table label for \\ref{}
        precision: decimal places for percentages

    Returns:
        Complete LaTeX table string ready to paste into paper.
    """
    n_groups = len(sh.group_ids)
    fmt = f'.{precision}f'

    lines = []
    lines.append(r'\begin{table*}[h]')
    lines.append(f'\\caption{{{caption}}}\\label{{{label}}}')
    lines.append(r'\begin{tabular*}{\textwidth}{@{\extracolsep\fill}l' + 'c' * n_groups + '}')
    lines.append(r'\toprule')
    lines.append('Sequence & ' + ' & '.join(str(gid) for gid in sh.group_ids) + r' \\')
    lines.append(r'\midrule')

    for row_label, stats_matrix in rows.items():
        values = sh.trend_check_per_group(stats_matrix, value, trend) * 100
        vals = ' & '.join(f'{v:{fmt}}' for v in values)
        lines.append(f'{row_label} & {vals} \\\\')

    lines.append(r'\botrule')
    lines.append(r'\end{tabular*}')
    lines.append(r'\end{table*}')

    result = '\n'.join(lines)
    print(result)
    return result
