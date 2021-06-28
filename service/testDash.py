import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import date as dt, timedelta
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Dropdown(
        id='timeframe_dropdown',
        multi=False,
        options=[
            {'label': 'Today', 'value': 'Today'},
            {'label': 'Yesterday', 'value': 'Yesterday'},
            {'label': 'Last 7 days', 'value': 'Last 7 days'}
        ],
        value='Today',
        clearable=False,
    ),
    dcc.DatePickerRange(
        id='datepicker',
        display_format='DD-MM-YYYY',
        first_day_of_week=1,
        max_date_allowed=dt.today(),
    ),
])


@app.callback(
    [Output('datepicker', 'start_date'),  # This updates the field start_date in the DatePicker
     Output('datepicker', 'end_date')],  # This updates the field end_date in the DatePicker
    [Input('timeframe_dropdown', 'value')],
)
def updateDataPicker(dropdown_value):
    if dropdown_value == 'Today':
        return dt.today(), dt.today()
    elif dropdown_value == 'Yesterday':
        return dt.today() - timedelta(1), dt.today() - timedelta(1)
    else:
        return dt.today() - timedelta(6), dt.today()


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
