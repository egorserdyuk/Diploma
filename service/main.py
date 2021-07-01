import json
import base64
import datetime
import io

import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)

app.title = "Интерактивная статистическая карта Алтайского края"
server = app.server

DEFAULT_OPACITY = 0.8

mapbox_access_token = "pk.eyJ1IjoidGhlY29vbGR1bXAiLCJhIjoiY2txZjBieGl6MTlhNDJwbm1jeWd2bXJkeiJ9.gHVFvp1gZDOkVd6NZTkMow"

world_path = 'data/' + 'map.geojson'
with open(world_path) as f:
    geojson = json.load(f)

df = pd.read_csv("data/test.csv", dtype={"County": str})
fig = go.Figure(px.choropleth_mapbox(df, geojson=geojson, color='Data',
                                     locations='County', featureidkey="properties.County",
                                     color_continuous_scale="Viridis",
                                     range_color=(0, max(df.Data)),
                                     center={"lat": 52.78, "lon": 83.22},
                                     mapbox_style="carto-positron", opacity=0.5, zoom=5))
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})


def read_file(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

            YEARS = []
            dates = df.Date.values.tolist()
            YEARS += dates

            if min(YEARS) < max(YEARS):
                fig = go.Figure(px.choropleth_mapbox(df, geojson=geojson, color='Data',
                                                     locations='County', featureidkey="properties.County",
                                                     color_continuous_scale="Viridis",
                                                     range_color=(0, df.Data.max()),
                                                     center={"lat": 52.78, "lon": 83.22},
                                                     mapbox_style="carto-positron", opacity=0.5, zoom=5,
                                                     animation_frame='Date'))
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            else:
                fig = go.Figure(px.choropleth_mapbox(df, geojson=geojson, color='Data',
                                                     locations='County', featureidkey="properties.County",
                                                     color_continuous_scale="Viridis",
                                                     range_color=(0, df.Data.max()),
                                                     center={"lat": 52.78, "lon": 83.22},
                                                     mapbox_style="carto-positron", opacity=0.5, zoom=5))
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        elif 'xls' or 'xlsx' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return fig, html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),
    ])


app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                html.H4(children="Интерактивная статистическая карта Алтайского края"),
                html.P(
                    id="description",
                    children="Данный ресурс поможет вам наглядно визуализировать любые имеющиеся статистические данные \
                    Алтайского края по районам",
                ),
            ],
            style={
                'textAlign': 'center',
                'margin': '10px'
            }
        ),
        html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="heatmap-container",
                    children=[
                        dcc.Graph(
                            figure=fig
                        ),
                    ],
                ),
            ],
        ),
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Перетащите или ',
                    html.A('Выберите файл')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
            html.Div(id='output-data-upload'),
        ])
    ],
)


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            read_file(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=True)
