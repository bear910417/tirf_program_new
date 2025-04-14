from dash import html
from dash_extensions.enrich import dcc
def aois_tab(fig_blob):
    return dcc.Tab(
        label='Aois',
        value='Aois',
        children=[
            html.Div(),
            html.Div(
                children=[
                    dcc.Graph(
                        id="g_blob",
                        figure=fig_blob,
                        config={'displayModeBar': False, 'staticPlot': False}
                    )
                ],
                style={
                    'padding': 5,
                    'display': 'flex',
                    'flexDirection': 'row',    # Corrected property name
                    'alignItems': 'flex-start'  # Adjusted to a valid value
                }
            ),
            html.Div(
                children=[
                    html.Div('Aoi Scale: '),
                    html.Div(
                        dcc.RangeSlider(
                            0, 10000, 100,
                            value=(0, 2000),
                            updatemode='drag',
                            tooltip={"placement": "bottom", "always_visible": False},
                            marks=None,
                            id='aoi_max'
                        ),
                        style={'width': '90%'}
                    ),
                ],
                style={
                    'padding': 5,
                    'display': 'flex',
                    'flexDirection': 'row',  # Corrected property name
                    'width': '90%'
                }
            )
        ],
        style={'width': '90%', 'height': '20%', 'padding': 5}
    )
