from waitress import serve 
from aoi_viewer import server 

serve(server, port = 8042, threads = 12)   