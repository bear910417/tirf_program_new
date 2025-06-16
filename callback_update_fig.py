# aoi_callbacks/callback_update_fig.py
import numpy as np
import subprocess
import time
import logging
import plotly.graph_objects as go
import os
from tqdm import tqdm
from dash import callback_context, no_update
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, Input, State
from aoi_utils import draw_blobs, move_blobs, update_blobs_coords, load_path, cal_blob_intensity, save_aoi_utils, load_aoi_utils, update_fret_labels
from cal_drift import cal_drift
from global_state import global_state

def register_update_fig(app, fsc):
    @app.callback(
        [
            Output("graph", "figure"),
            Output("graph", "clickData"),
            Output("anchor", "value"),
            Output("blob", "disabled"),
            Output("cal_intensity", "disabled"),
            Output("frame_slider", "value"),
            Output("frame_slider", "max"),
            Output("snap_time_g", "max"),
            Output("red_time", "max"),
            Output("snap_time_b", "max"),
            Output("green_time", "max"),
            Output("aoi_mode", "value"),
            Output("aoi_num", "children"),
            Output("loadp", "title"),
            Output("FRET", "outline"),
            Output("auto", "n_clicks")
        ],
        [
            Input("graph", "clickData"),
            Input("graph", "relayoutData"),
            Input("blob", "n_clicks"),
            Input("up", "n_clicks"),
            Input("down", "n_clicks"),
            Input("left", "n_clicks"),
            Input("right", "n_clicks"),
            Input("fit_gauss", "n_clicks"),
            Input("frame_slider", "value"),
            Input("anchor", "value"),
            Input("average_frame", "value"),
            Input("loadp", "n_clicks"),
            Input("minf", "value"),
            Input("maxf", "value"),
            Input("reverse", "value"),
            Input("channel", "value"),
            Input("cal_drift", "n_clicks"),
            Input("load_drift", "n_clicks"),
            Input("cal_intensity", "n_clicks"),
            Input("openp", "n_clicks"),
            Input("configs", "value"),
            Input("aoi_mode", "value")
        ],
        [
            State("ratio_thres", "value"),
            State("radius", "value"),
            State("selector", "value"),
            State("move_step", "value"),
            State("path", "value"),
            State("mpath", "value"),
            State("plot_circle", "value"),
            State("thres", "value"),
            State("per_n", "value"),
            State("pairing_threshold", "value"),
            State("auto", "n_clicks")
        ],
    )
    def update_fig(clickData, relayout, blob, up, down, left, right, fit_gauss, frame, anchor,
                   average_frame, loadp, minf, maxf, reverse, channel, cal_drift_bt, load_drift, cal_intensity,
                   openp, configs, aoi_mode, ratio_thres, radius, selector,
                   move_step, path, mpath, plot, thres, per_n, pairing_threshold, auto):
        
        gs = global_state
        current_fig = gs.fig
        step_start = time.perf_counter()

        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        
        if "loadp" in changed_id:
            fsc.set("load_progress", "0")
            gs.loader, gs.image_g, gs.image_r, gs.image_b, gs.image_datas = load_path(thres, path, fsc)
            gs.blob_disable = False
            frame = 0
            fsc.set("stage", "Image Loaded")
            logging.info("Image load in %.3f sec", time.perf_counter()-step_start)
            step_start = time.perf_counter()


        channel_dict = {
            "green": gs.image_g if gs.image_g is not None else np.zeros((1,512,512)),
            "red": gs.image_r if gs.image_r is not None else np.zeros((1,512,512)),
            "blue": gs.image_b if gs.image_b is not None else np.zeros((1,512,512))
        }

        if "blob" in changed_id:
            fsc.set("progress", 0)
            gs.loader.gen_dimg(anchor = anchor, mpath = mpath, maxf = maxf, minf = minf, laser = channel, average_frame = average_frame)
            blob_list = gs.loader.det_blob(plot=plot, fsc=fsc, thres=thres, r=radius, ratio_thres=float(ratio_thres))
            gs.blob_list = blob_list
            gs.coord_list = [b.get_coord() for b in blob_list]
            coord_array = np.array(gs.coord_list) if gs.coord_list else np.empty((0,))
            current_fig = draw_blobs(current_fig, coord_array, gs.dr if gs.dr is not None else radius, reverse)
            fsc.set("stage", "Blobing Finished")
            logging.info("Blob detection in %.3f sec", time.perf_counter()-step_start)
            step_start = time.perf_counter()

        for move_button in ["up", "down", "left", "right"]:
            if move_button in changed_id:
                coord_array = np.array(gs.coord_list) if gs.coord_list else np.empty((0,))
                coord_array = move_blobs(coord_array, selector, int(move_step), changed_id)
                gs.coord_list = coord_array.tolist()
                update_blobs_coords(gs.blob_list, coord_array)

                current_fig = draw_blobs(current_fig, coord_array, gs.dr if gs.dr is not None else radius, reverse)
                logging.info("Movement %s in %.3f sec", move_button, time.perf_counter()-step_start)
                step_start = time.perf_counter()
        if 'fit_gauss' in changed_id:

            gs.loader.gen_dimg(anchor = anchor, mpath = mpath, maxf = maxf, minf = minf, laser = channel, average_frame = average_frame)
            logging.info("BM3D Image Processed in %.3f sec", time.perf_counter()-step_start)
            step_start = time.perf_counter()

            ch_dict = {
                'channel_r': 'red',
                'channel_g': 'green',
                'channel_b': 'blue'
            }
            for b in tqdm(gs.blob_list):
                b.set_image(gs.loader.dframe_r, laser = 'red')
                b.set_image(gs.loader.dframe_g, laser = 'green')
                b.set_image(gs.loader.dframe_b, laser = 'blue')
                b.gaussian_fit(ch = ch_dict[selector], laser = channel)
            gs.coord_list = [b.get_coord() for b in gs.blob_list]
            coord_array = np.array(gs.coord_list) 
            current_fig = draw_blobs(current_fig, coord_array, gs.dr, reverse)
            logging.info("Blob fitting for {ch} with {channel} laser in %.3f sec", time.perf_counter()-step_start)

        if "openp" in changed_id:
            subprocess.Popen(f'explorer "{path}"')

        if "cal_drift" in changed_id:
            if gs.loader == None:
                logging.info("Error: No image loader detected")
            else:
                cal_drift(
                    gs = gs, 
                    channel_dict = channel_dict, 
                    fsc = fsc, 
                    mpath = mpath, 
                    path = path,
                    maxf = maxf, 
                    minf = minf, 
                    average_frame = average_frame, 
                    channel = channel, 
                    per_n = per_n, 
                    pairing_threshold = pairing_threshold)
                
        if "load_drift" in changed_id:
            if gs.loader == None:
                logging.info("Error: No image loader detected")
            else:
                drifts_dir = os.path.join(path, 'drifts')
                os.makedirs(drifts_dir, exist_ok=True)
                for c1, attr in [('g', 'image_g'), ('b', 'image_b'), ('r', 'image_r')]:
                    try:
                        warped_image = np.load(os.path.join(drifts_dir, f'warped_{c1}.npy'))
                        setattr(gs, attr, warped_image)
                        setattr(gs.loader, attr, warped_image)
                        logging.info(f"Successfully loaded drift-corrected {attr} .")
                    except Exception as e:
                        logging.warning(f"Could not load drift-corrected {attr}: {e}")
            
        if "cal_intensity" in changed_id:
            fsc.set("cal_progress", 0)
            cal_blob_intensity(gs.loader, np.array(gs.coord_list), path, gs.image_datas, maxf, minf, fsc)
            fsc.set("stage", "Intensity Calculated")
            logging.info("Intensity calculation in %.3f sec", time.perf_counter()-step_start)
            step_start = time.perf_counter()

        if "graph" in changed_id:
            if isinstance(relayout, dict) and "xaxis.range[1]" in relayout:
                try:
                    size = np.round(512 / (relayout["xaxis.range[1]"] - relayout["xaxis.range[0]"]), 2)
                    if size != gs.org_size:
                        gs.org_size = size
                        gs.dr = radius * size
                        if np.any(np.array(gs.coord_list)):
                            current_fig = draw_blobs(current_fig, np.array(gs.coord_list), gs.dr, reverse)
                except Exception as e:
                    logging.exception("Error during relayout: %s", e)

            if isinstance(clickData, dict):
                if clickData["points"][0]["curveNumber"] in [1,2,3]:
                    if aoi_mode == 0:
                        remove_id = clickData["points"][0]["pointNumber"]
                        gs.rem_list.append(gs.coord_list[remove_id])
                        gs.rem_list_blob.append(gs.blob_list[remove_id])
                        gs.coord_list = np.delete(np.array(gs.coord_list), remove_id, axis=0).tolist()
                        gs.blob_list.pop(remove_id)
                        current_fig = draw_blobs(current_fig, np.array(gs.coord_list), gs.dr, reverse)

        # (Undo, Save, Load, Clear AOI logic)
        if aoi_mode == 2:
            aoi_mode = 0
            if len(gs.rem_list_blob) > 0:
                new_coord = np.array(gs.rem_list.pop())
                if np.any(np.array(gs.coord_list)):
                    combined = np.concatenate((np.array(gs.coord_list), new_coord.reshape(1,12)), axis=0)
                else:
                    combined = new_coord.reshape(1,12)
                gs.coord_list = combined.tolist()
                gs.blob_list.append(gs.rem_list_blob.pop())
                current_fig = draw_blobs(current_fig, np.array(gs.coord_list), gs.dr, reverse)
        if aoi_mode == 3:
            aoi_mode = 0
            save_aoi_utils(gs.blob_list, path + r'\\aoi.dat')
            save_aoi_utils(gs.rem_list_blob, path + r'\\bad_aoi.dat')
            logging.info("Saved AOI")
        if aoi_mode == 4:
            aoi_mode = 0
            gs.blob_list = load_aoi_utils(path + r'\\aoi.dat')
            gs.coord_list = [b.get_coord() for b in gs.blob_list]
            current_fig = draw_blobs(current_fig, np.array(gs.coord_list), gs.dr, reverse)
            logging.info("Loaded AOI")
        if aoi_mode == 5:
            aoi_mode = 0
            gs.rem_list = gs.rem_list + gs.coord_list
            gs.rem_list_blob = gs.rem_list_blob + gs.blob_list
            gs.blob_list = []
            gs.coord_list = []
            current_fig = draw_blobs(current_fig, np.empty((0,)), gs.dr, reverse)
            logging.info("Cleared AOI")
        
        if "channel" in changed_id:
            if frame > channel_dict[channel].shape[0]:
                frame = 0
        if "anchor.value" in changed_id:
            if int(anchor) < channel_dict[channel].shape[0]:
                frame = int(anchor)

        
        
        if 'reverse.value' in changed_id:
            if int(reverse) == 0:
                current_fig.update_traces(colorscale='gray', selector=dict(type='heatmap'))
                current_fig = draw_blobs(current_fig, gs.coord_list, gs.dr, reverse)
            else:
                current_fig.update_traces(colorscale='gray_r', selector=dict(type='heatmap'))
                current_fig = draw_blobs(current_fig, gs.coord_list, gs.dr, reverse)
        
        end_idx = min(channel_dict[channel].shape[0], int(frame) + int(average_frame))
        start_idx = max(0, end_idx - int(average_frame))

        smooth_image = np.average(channel_dict[channel][start_idx:end_idx], axis=0)
        current_fig.data[0].z = smooth_image

        if "autoscale" in changed_id:
            maxf = np.round(np.max(smooth_image))
            minf = np.round(np.min(smooth_image))
            
        current_fig.update_traces(zmax=maxf, zmin=minf, selector=dict(type="heatmap"))
        current_fig = update_fret_labels(current_fig, frame)


        slider_max = channel_dict[channel].shape[0]
        snap_g_max = max(channel_dict["green"].shape[0]-1, 0)
        r_max = max(channel_dict["red"].shape[0]-1, 0)
        snap_b_max = max(channel_dict["blue"].shape[0]-1, 0)
        g_max = max(channel_dict["green"].shape[0]-1, 0)
        anchor = min(int(frame), channel_dict[channel].shape[0])
        aoi_num = len(gs.coord_list)
        auto_state = no_update if fsc.get("mode") != "auto" else auto + 1

        gs.fig = current_fig

        return (current_fig, None, anchor, gs.blob_disable, gs.blob_disable,
                anchor, slider_max, snap_g_max, r_max, snap_b_max, g_max, aoi_mode,
                aoi_num, None, True, auto_state)

    return register_update_fig
