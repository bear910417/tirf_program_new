# aoi_callbacks/callback_load_config.py
from dash_extensions.enrich import Output, Input, State
import numpy as np
from dash import callback_context, no_update
from dash.exceptions import PreventUpdate
from aoi_utils import save_config, load_config
from global_state import global_state as gs

current_config_reference = load_config('reference')

def sync_dict(given: dict, reference: dict) -> dict:
    """
    Returns a new dictionary with exactly the keys in `reference`.
    - If a key exists in both dictionaries, the value from `given` is used.
    - If a key is missing in `given`, the reference value is used.
    - Extra keys in `given` are removed.
    """
    return { key: given.get(key, reference[key]) for key in reference }

def register_load_config(app, fsc):
    @app.callback(
                output = [
                dict(
                thres = Output('thres', 'value'),
                mpath = Output('mpath', 'value'),
                average_frame = Output('average_frame', 'value'),
                ratio_thres = Output('ratio_thres', 'value'),
                minf = Output('minf', 'value'),
                maxf = Output('maxf', 'value'),
                channel = Output('channel', 'value'),
                radius = Output('radius', 'value'),
                leakage_g = Output('leakage_g', 'value'),
                leakage_b = Output('leakage_b', 'value'),
                f_lag = Output('f_lag', 'value'),
                lag_b = Output('lag_b', 'value'),
                snap_time_g = Output('snap_time_g', 'value'),
                snap_time_b = Output('snap_time_b', 'value'),
                red_intensity = Output('red_intensity', 'value'),
                red_time = Output('red_time', 'value'),
                red = Output('red', 'value'),
                green_intensity = Output('green_intensity', 'value'),
                green_time = Output('green_time', 'value'),
                green = Output('green', 'value'),
                fit = Output('fit', 'value'),
                fit_b = Output('fit_b', 'value'),
                gfp_plot = Output('gfp_plot', 'value'),
                preserve_selected = Output('ps', 'value'),
                overwrite = Output('ow', 'value')
                )],

                inputs = dict(
                configs = Input('configs', 'value'),
                savec = Input('savec', 'n_clicks'),
                autoscale = Input('autoscale', 'n_clicks'),
                config_data = dict(
                thres = Input('thres', 'value'),
                mpath = Input('mpath', 'value'),
                average_frame = Input('average_frame', 'value'),
                ratio_thres = Input('ratio_thres', 'value'),
                minf = Input('minf', 'value'),
                maxf = Input('maxf', 'value'),
                channel = Input('channel', 'value'),
                radius = Input('radius', 'value'),
                leakage_g = Input('leakage_g', 'value'),
                leakage_b = Input('leakage_b', 'value'),
                f_lag = Input('f_lag', 'value'),
                lag_b = Input('lag_b', 'value'),
                snap_time_g = Input('snap_time_g', 'value'),
                snap_time_b = Input('snap_time_b', 'value'),
                red_intensity = Input('red_intensity', 'value'),
                red_time = Input('red_time', 'value'),
                red = Input('red', 'value'),
                green_intensity = Input('green_intensity', 'value'),
                green_time = Input('green_time', 'value'),
                green = Input('green', 'value'),
                fit = Input('fit', 'value'),
                fit_b = Input('fit_b', 'value'),
                gfp_plot = Input('gfp_plot', 'value'),
                preserve_selected = Input('ps', 'value'),
                overwrite = Input('ow', 'value'))
                )
        )
    def load_config_callback(configs, savec, autoscale, config_data):
        
        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        if "autoscale" in changed_id:
            current_fig = gs.fig
            maxf_val = float(np.round(np.max(current_fig["data"][0]["z"])))
            minf_val = float(np.round(np.min(current_fig["data"][0]["z"])))
            config_data["maxf"] = maxf_val
            config_data["minf"] = minf_val
            return config_data
            
        if "savec" in changed_id:
            save_config(configs, config_data)
            raise PreventUpdate
        if "configs" in changed_id:
            loaded = load_config(int(configs))
            if loaded is None:
                raise PreventUpdate
            loaded = sync_dict(loaded, current_config_reference)
            return loaded
        raise PreventUpdate

    return register_load_config
