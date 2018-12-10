#!/usr/bin/env python

import numpy as np

import utils.ffwd as ffwd 
import utils.pivot as pivot 
import utils.io as io

import os
import json

import argparse, argcomplete

from pprint import pprint

from sklearn.model_selection import train_test_split,KFold

##
dipho_features = ['leadptom',
                  'subleadptom',
                  'leadmva',
                  'subleadmva',
                  'leadeta',
                  'subleadeta',
                  'sigmarv',
                  'sigmawv',
                  'CosPhi',
                  'vtxprob']

## command_line options
parser = argparse.ArgumentParser(description='Run adversarial training of CMS Hgg diphoton ID MVA')
parser.add_argument('--inp-dir', type=str, dest='inp_dir', default=os.environ['SCRATCH']+'/diphotonID/samples/', help='input directory')
parser.add_argument('--out-dir', type=str, dest='out_dir', default=os.environ['SCRATCH']+'/diphotonID/AN_output/')
parser.add_argument('--inp-file', type=str, dest='inp_file', default='diphoton_id_s+b_train.hd5')
parser.add_argument('--features', type=str, dest='features', default='')
parser.add_argument('--lambda', type=float, dest='lambd', default=1.)
parser.add_argument('--valid-frac', type=float, dest='valid_frac', default=0.10)
parser.add_argument('--test-frac', type=float, dest='test_frac', default=0.10)
parser.add_argument('--batch-size', type=int, dest='batch_size', default=1024)
parser.add_argument('--epochs', type=int, dest='epochs', default=20)
parser.add_argument('--activations', type=str, dest='activations', default='relu')
parser.add_argument('--hparams', type=str, dest='hparams', default=None)
parser.add_argument('--seed', type=int, dest='seed', default=98543)
parser.add_argument('--x-val', action='store_true', dest='x_val', default=False)
parser.add_argument('--nkfolds', type=int, dest='nkfolds', default=5)
parser.add_argument('--pretrain', action='store_true', dest='pretrain', default=False)
parser.add_argument('--ext-dsc-inputs', action='store_true', dest='ext_dsc_inputs', default=False)
parser.add_argument('--clf-pretrain-weights', type=str, dest='clf_pretrain_weights', default='')
parser.add_argument('--dsc-pretrain-weights', type=str, dest='dsc_pretrain_weights', default='')

## parse options
argcomplete.autocomplete(parser)
options = parser.parse_args()

options.out_dir = options.out_dir
if options.pretrain:
    options.out_dir += '/pretrain/'
else:
    options.out_dir += '_lambda'+str(options.lambd)
os.makedirs(options.out_dir, exist_ok=True)

if options.features == '':
    features = dipho_features
    options.features = ','.join(features)
else:
    features = options.features.split(',')

if os.path.exists(options.inp_file):
    inp_file = options.inp_file
else:
    inp_file = options.inp_dir+'/'+options.inp_file

hparams = {}
if options.hparams is not None:
    for fname in  options.hparams.split(','):
       with open(fname) as hf:
          pars = json.loads(hf.read())
          hparams.update(pars)    # if inside several files we change the same parameter, it will overwrite for the one in the last file    
    
###---Read data
data = io.read_data(inp_file)

data['isSignal'] = (data['processIndex'] < 5).astype(np.float32)
data['isBkg'] = (data['processIndex'] >= 5).astype(np.float32)
X = data[features]
y = data['isSignal'].values.reshape(-1,1)
w = np.abs(data['weight'].values.ravel())
w[(y==1).ravel()] /= w[(y==1).ravel()].sum() / w[(y==1).ravel()].shape[0]
w[(y==0).ravel()] /= w[(y==0).ravel()].sum() / w[(y==1).ravel()].shape[0]

###---Sort out model parameters
def get_kwargs(fn,**kwargs):
    params = set(fn.__code__.co_varnames[:fn.__code__.co_argcount]+tuple(kwargs.keys()))
    for par in params:
        if hasattr(options,par):
            kwargs[par] = getattr(options,par)
        if par in hparams:
            kwargs[par] = hparams.pop(par)
    return kwargs

loss_params=dict()
init_kwargs = get_kwargs(pivot.PivotClassifier.__init__, monitor_dir=options.out_dir, lambd=options.lambd)
fit_kwargs = get_kwargs(pivot.PivotClassifier.fit, batch_size=options.batch_size, epochs=options.epochs, syst_shift={('leadmva','subleadmva'):[0, -0.1, 0.1]})
pretrain_clf_kwargs = { 'batch_size' : 8192, 'epochs' : options.epochs }
pretrain_dsc_kwargs = { 'batch_size' : 8192, 'epochs' : options.epochs*10 }

###---Prepare net model
clf = ffwd.FFWDRegression('clf', X.shape[1:], y.shape[1:], loss='binary_crossentropy', monitor_dir=options.out_dir)
if options.ext_dsc_inputs:
    dsc = ffwd.FFWDRegression('dsc', y.shape[1:], (1, 3), input_shape_extra=X.shape[1:], loss='categorical_crossentropy', monitor_dir=options.out_dir)
else:
    dsc = ffwd.FFWDRegression('dsc', y.shape[1:], (1, 3), loss='categorical_crossentropy', monitor_dir=options.out_dir)
net = pivot.PivotClassifier('pivot', clf, dsc, **init_kwargs)   

pprint(net.get_params())

adv_model, dsc_model = net()        

###---Load pretrained model for CLF and DSC
#if not options.pretrain:
if options.clf_pretrain_weights != "":
    clf.model.load_weights(options.clf_pretrain_weights)
if options.dsc_pretrain_weights != "":
    dsc.model.load_weights(options.dsc_pretrain_weights)

print(adv_model.summary())
print(dsc_model.summary())
print(dsc().summary())

adv_model_summary = str(adv_model.to_json()) 
dsc_model_summary = str(dsc_model.to_json()) 

if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)

with open(options.out_dir+'/adv_model_summary.txt','w+') as fo:
    fo.write(adv_model_summary)
with open(options.out_dir+'/dsc_model_summary.txt','w+') as fo:
    fo.write(dsc_model_summary)

# store = dict( model_params = net.get_params(),
#               fit_kwargs = fit_kwargs,
#               options = options.__dict__
#               )
# with open(options.out_dir+'/config.json','w+') as fo:
#     fo.write( json.dumps( store, indent=4 ) )
#     fo.close()

if options.x_val==True:
    kf = KFold(n_splits=int(1./options.valid_frac),shuffle=True,random_state=options.seed) 
    kf_idx = iter(kf.split(X)) 
    for kfold in range(options.nkfolds):
        # split data
        train_idx,valid_idx = next(kf_idx)
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        net.fit(X_train,y_train, 
                validation_data=(X_valid,y_valid),
                **fit_kwargs)
else :
    X_train,X_valid,y_train,y_valid,w_train,w_valid = train_test_split(X,y,w,test_size=options.valid_frac,random_state=options.seed)
    
    print(w_train.shape, y_train.shape)

    ### here if you want to save hdf file after each better epoch, put kfold=-1
    net.fit(X_train,y_train,w_train,
            validation_data=(X_valid,y_valid,w_valid),
            pretrain_clf_kwargs=pretrain_clf_kwargs,
            pretrain_dsc_kwargs=pretrain_dsc_kwargs,
            **fit_kwargs
    )

