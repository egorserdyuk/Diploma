import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

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

geodata_path = 'data/' + 'map_points.csv'
geodata = pd.read_csv(geodata_path, dtype={"County": str})

df = pd.read_csv("data/HIV.csv", dtype={"County": str})

df = df.set_index('County')
df = df.reindex(index=geodata['County'])
df = df.reset_index()

figScatter = go.Scattermapbox(
    lat=[geodata.lat[i] for i in range(geodata.shape[0])],
    lon=[geodata.lon[i] for i in range(geodata.shape[0])],
    marker=dict(size=5, color='black'),
    mode='markers+text',
    text=[str(geodata['County'][i]) + '<br>' + str(df['Data'][i]) for i in range(geodata.shape[0])],  # .astype(str)
    textfont=dict(size=16, color='black'),
    textposition="bottom center"
)

layout = dict(margin=dict(l=0, t=0, r=0, b=0, pad=0),
              mapbox=dict(accesstoken=mapbox_access_token,
                          center=dict(lat=52.78, lon=83.22),
                          style='light',
                          zoom=5))

fig = go.Figure(data=figScatter, layout=layout)

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
    ],
)

if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=True)
