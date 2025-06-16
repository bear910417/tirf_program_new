# global_state.py
import numpy as np
from aoi_figure import create_initial_figure

class GlobalState:
    def __init__(self):

        self.fig = create_initial_figure(np.zeros((1, 512, 512)), 0, 2000, 7)
        self.loader = None
        self.image_g = np.zeros((1, 512, 512))
        self.image_r = np.zeros((1, 512, 512))
        self.image_b = np.zeros((1, 512, 512))
        self.image_datas = np.zeros((1, 512, 512))
        self.coord_list = []
        self.blob_list = []
        self.rem_list = []
        self.rem_list_blob = []
        self.dr = 1
        self.org_size = 1
        self.blob_disable = True
        self.fret_g = []

global_state = GlobalState()
