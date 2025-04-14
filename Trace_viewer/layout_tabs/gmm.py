from dash import dcc, html
import dash_bootstrap_components as dbc
from dash_extensions.enrich import dash_table

def gmm_tab(fig2):
    return dcc.Tab(
        label='GMM',
        value='GMM',
        children=[
            html.Br(),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            dcc.Loading(
                                id="loading",
                                type="default",
                                children=dcc.Graph(figure=fig2, id="gmm_hist")
                            )
                        ],
                        style={'width': '50%', 'float': 'middle'}
                    ),
                    html.Div(
                        children=[
                            html.Div('Fit n Gaussian Peaks: '),
                            dcc.RadioItems(
                                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                                '1',
                                id='gmm_comps',
                                labelStyle={'display': 'inline-block', 'marginTop': '5px', "margin-left": "20px"}
                            ),
                            html.Div('Cov Type: '),
                            dcc.RadioItems(
                                ['full', 'spherical', 'diag', 'tied'],
                                'full',
                                id='gmm_cov_type',
                                labelStyle={'display': 'inline-block', "margin-left": "20px"}
                            ),
                            html.Br(),
                            html.Div('Init Means: '),
                            dash_table.DataTable(
                                id='gmm_means',
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
                            html.Div('Binsize:'),
                            dcc.Slider(
                                0.01,
                                0.1,
                                step=0.01,
                                id='binsize',
                                value=0.02
                            ),
                            html.Br(),
                            html.Div(
                                children=[
                                    dcc.Dropdown(
                                        ['fret_g', 'fret_b'],
                                        'fret_g',
                                        clearable=False,
                                        searchable=False,
                                        style={'width': '100px'},
                                        id='gmm_channel'
                                    ),
                                    html.Button('Fit Histogram', id='gmm_fit'),
                                    html.Button('Save Histogram', id='gmm_save'),
                                ],
                                style={'display': 'flex'}
                            )
                        ],
                        style={'width': '30%', 'float': 'middle', 'display': 'inline-block'}
                    )
                ],
                style={'padding': 5, 'width': "100%", 'float': 'middle', 'display': 'flex'}
            )
        ],
        style={'width': '90%', 'height': '20%'}
    )
