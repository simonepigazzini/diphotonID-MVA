import pandas as pd


# -------------------------------------------------------------------------------------------
def read_data(fname,**kwargs):
    
    df = pd.read_hdf(fname,**kwargs)

    return df

    
    
