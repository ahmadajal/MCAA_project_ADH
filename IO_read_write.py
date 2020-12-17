import numpy as np
import pandas as pd

def input_read(path_to_file):
    df = pd.read_csv(path_to_file)
    inds = np.array(df["id"])
    class G(object):
        def __init__(self):
            self.x = np.array(df[["coordinates_x", "coordinates_y"]])
            self.v = np.array(df["population"])
            self.ids = inds
    g = G()
    return g

def write_output(g, selected_cites, output_file_name):
    d = {"id": g.ids,
        "include": selected_cites.astype(int)}
    df = pd.DataFrame(d)
    df.to_csv(output_file_name+".csv", index=False)

def read_output(path_to_file):
    df = pd.read_csv(path_to_file)
    return np.array(df["include"])