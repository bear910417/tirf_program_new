from dash import dcc, html
import dash_daq as daq
from dash_extensions.enrich import dash_table

def hmm_tab():
    return dcc.Tab(
        label='HMM',
        value='HMM',
        children=[
            html.Div(
                children=[
                    html.Div('Fit:'),
                    daq.ToggleSwitch(
                        id='hmm_fit', value=True, color='green',
                        style={"margin-left": "10px"}
                    ),
                    html.Div('Fix means:', style={"margin-left": "10px"}),
                    daq.ToggleSwitch(
                        id='hmm_fix', value=0, color='green',
                        style={"margin-left": "10px"}
                    ),
                    html.Div('Plot:', style={"margin-left": "10px"}),
                    daq.ToggleSwitch(
                        id='hmm_plot', value=0, color='green',
                        style={"margin-left": "10px"}
                    ),
                    html.Div('n_iter:', style={"margin-left": "10px"}),
                    dcc.Input(
                        20, id="hmm_niter", type="number", placeholder="",
                        style={'textAlign': 'left', 'width': '50px'},
                        persistence='True'
                    )
                ],
                style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}
            ),
            html.Div(
                children=[
                    html.Div('Cov Type: '),
                    dcc.RadioItems(
                        ['spherical', 'diag', 'full', "tied"],
                        'spherical',
                        id='hmm_cov_type',
                        labelStyle={'display': 'inline-block', "margin-left": "10px"}
                    )
                ],
                style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}
            ),
            html.Div('Init Means: ', style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}),
            html.Div(
                dash_table.DataTable(
                    id='hmm_means',
                    columns=[{'id': str(p), 'name': str(p)} for p in range(0, 10)],
                    data=[{str(param): -1 for param in range(0, 10)}],
                    style_cell={
                        'minWidth': '20px', 'width': '20px', 'maxWidth': '20px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'textAlign': 'center'
                    },
                    editable=True,
                    persistence=True,
                    persisted_props=['data']
                ),
                style={'width': '50%', 'height': '50%', 'padding': 5, "margin-left": "60px"}
            ),
            html.Div(
                children=[
                    html.Div('Epoch:'),
                    dcc.Input(
                        id="hmm_epoch", value=10, type="number", placeholder="",
                        style={"margin-left": "10px", "textAlign": "center", "width": "60px"}
                    ),
                    html.Button('Start', id='hmm_start')
                ],
                style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}
            )
        ],
        style={'width': '90%', 'height': '20%'}
    )
