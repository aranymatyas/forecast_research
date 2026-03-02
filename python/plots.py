import pandas as pd
import numpy as np
import kagglehub
from datetime import datetime, timedelta, date
from typing import Literal
import zipfile
import plotly.express as px
import plotly.graph_objects as go

from python.utils import is_long, long_to_wide

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
    return fig.show()


def plot_heatmap(df: pd.DataFrame):
    ''' Plots the heatmap of values in the time series. Optimal for wide format. Highlights NaNs '''
    if is_long(df):
        df = long_to_wide(df)

    fig = px.imshow(df, aspect='auto', title=f"Time Series Value Distribution ({len(df.columns)} Series)", color_continuous_scale="Viridis",
                    labels=dict(x="Series", y="Date", color="Value")
    )

    # Set the background of the actual plotting area.
    fig.update_layout( height=800, plot_bgcolor='red', xaxis_showticklabels=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))

    return fig.show()


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
        unique_hierarchies = list(pd.unique(hierarchies))
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

    return fig.show()
