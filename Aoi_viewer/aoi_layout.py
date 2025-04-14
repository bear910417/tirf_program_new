from dash import dcc, html
import dash_bootstrap_components as dbc

from aoi_utils import load_config

from layout.blob_tab import get_blob_tab
from layout.fret_tab import get_fret_tab

config = load_config(1)
    

def make_layout(fig):
    layout = html.Div([
        # Hidden store to hold all extra state; initial state can be adjusted as needed.
        dcc.Store(
            id="state-store",
            storage_type="memory",
            data={
                "coord_list": [],
                "org_size": 1,
            }
        ),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id="graph",
                    figure = fig,
                    style={'width': 1024, 'height': 1024},
                    config={'scrollZoom': True, 'modebar_remove': ['box select', 'lasso select']}
                ),
                html.Div([
                    html.Div(
                        dcc.Slider(
                            0, 0, 1,
                            value=0,
                            updatemode='drag',
                            tooltip={"placement": "bottom", "always_visible": False},
                            marks=None,
                            id='frame_slider'
                        ),
                        style={'width': 900}
                    ),
                    dcc.Input(
                        value=0, id="anchor", type="text",
                        placeholder="",
                        style={'textAlign': 'center'},
                        size='3', debounce=True
                    )
                ], style={"padding": 5, 'display': 'flex', 'flex-direction': 'row'}),
            ]),
            dbc.Col([
                dcc.Tabs(
                    id='tabs-example-1',
                    value='tab-1',
                    children=[
                        dcc.Tab(label='Blob', children=get_blob_tab(config)),
                        dcc.Tab(label='FRET', children=get_fret_tab(config))
                    ],
                    style={'width': '600px', 'padding': 5}
                ),
                html.Div([
                    dbc.RadioItems(
                        id="configs",
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary",
                        labelCheckedClassName="active",
                        options=[
                            {"label": "Config 1", "value": 1},
                            {"label": "Config 2", "value": 2},
                            {"label": "Config 3", "value": 3},
                            {"label": "Config 4", "value": 4},
                        ],
                        value=1,
                        style={'width': 500},
                        labelStyle={'width': '100%'}
                    ),
                    html.Button('Save Config', id='savec', className="btn btn-outline-primary")
                ], style={'padding': 20, 'display': 'flex', 'flex-direction': 'row'})
            ]),
        ], align="center")
    ])

    return layout