import streamlit as st
import grimoire as g
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

data = {'done': 77, 'today': 3, 'rest': 20}


def progress_bar(data):
    fig = go.Figure(
    )
    colors = iter(['#7bff77', '#ffff00', '#5a7758'])

    for i in data.items():
        fig.add_trace(
            go.Bar(y=['time'],
                   x=[i[1]],
                   name=i[0],
                   orientation='h',
                   marker=dict(color=next(colors), )))

    fig.update_layout(
        barmode='stack',
        height=200,
        margin=dict(r=0, pad=0),
    )
    return fig


st.plotly_chart(progress_bar(data))
