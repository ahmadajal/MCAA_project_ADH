import numpy as np
import pandas as pd

def input_read(path_to_file):
    df = pd.read_csv(path_to_file)
    class G(object):
        def __init__(self):
            self.x = np.array(df[["position x", "position y"]])
            self.v = np.array(df["normalized population"])
    g = G()
    return g

def write_output(selected_cites, output_file_name):
    d = {"city id": np.arange(1, len(selected_cites)+1),
        "selected": selected_cites.astype(int)}
    df = pd.DataFrame(d)
    df.to_csv(output_file_name+".csv", header=False, index=False)