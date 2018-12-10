import pandas as pd


# -------------------------------------------------------------------------------------------
def read_data(fname,**kwargs):

    if fname.endswith('.hd5'):
        df = pd.read_hdf(fname,**kwargs)
    elif fname.endswith('.csv'):
        df = pd.read_csv(fname,**kwargs)

    return df

    
    
