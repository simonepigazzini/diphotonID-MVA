from keras.utils import to_categorical

import numpy as np

# -----------------------------------------------------------------------------
def profile(target,xvar,bins=10,range=None,uniform=False,moments=True,
            quantiles=np.array([0.25,0.5,0.75])):

    if range is None:
        if type(bins) is not int:
            xmin, xmax = bins.min(), bins.max()
        else:
            xmin, xmax = xvar.min(),xvar.max()
    else:
        xmin, xmax = range
    mask = ( xvar >= xmin ) & ( xvar <= xmax )
    xvar = xvar[mask]
    target = target[mask]
    if type(bins) == int:
        if uniform:
            bins = np.linspace(xmin,xmax,num=bins+1)
        else:
            ## print(xmin,xmax)
            ## xvar = np.clip( xvar, xmin, xmax )
            bins = np.percentile( xvar, np.linspace(0,100.,num=bins+1) )
            bins[0] = xmin
            bins[-1] = xmax
    print(bins.shape)
    ibins = np.digitize(xvar,bins)-1
    categories = to_categorical( ibins )

    ret = [bins]
    if moments:
        mtarget = target.reshape(-1,1) * categories
        weights = categories
        mean = np.average(mtarget,weights=categories,axis=0)
        mean2 = np.average(mtarget**2,weights=categories,axis=0)
        ret.extend( [mean, np.sqrt( mean2 - mean**2)] )
    if quantiles is not None:
        values = []
        print(categories.shape[1])
        for ibin in np.arange(categories.shape[1],dtype=int):
            values.append( np.percentile(target[categories[:,ibin].astype(np.bool)],quantiles*100.,axis=0).reshape(-1,1) )
            ## print(values)
        ret.append( np.concatenate(values,axis=-1) )
    return tuple(ret)

