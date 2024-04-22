import os,sys,glob,traceback, pickle
import pandas as pd
import xarray as xr
import numpy as np
import scipy
from scipy import stats
from scipy.special import erf
import matplotlib.pyplot as plt
from multiprocessing import Pool

fair_mesmer_run = pd.read_csv('../agriculture/fair_mesmer_run_translation.csv')
scenarios = list(fair_mesmer_run.columns)[1:]
ESMs = [f.split('/')[-1] for f in glob.glob('/mnt/PROVIDE/mesmer_emulations/mesmer-m-tp_emulations/full_emu/netcdf/*')]
gmts = np.arange(0.0, 5., 0.1).round(1)

selected_years = {}
for scenario in scenarios:
    fair2mesmer = {k:v for k,v in zip(fair_mesmer_run[scenario].values, fair_mesmer_run[scenario].index)}
    mesmer2fair = {v:k for k,v in zip(fair_mesmer_run[scenario].values, fair_mesmer_run[scenario].index)}
    relevant_runs = sorted(fair2mesmer.keys())

    gmst = pd.read_csv(f'../agriculture/tier_1/fair_temperatures/scen_{scenario}.csv', index_col=0)
    gmst.columns = [int(c) for c in gmst.columns]
    gmst = gmst.loc[:,relevant_runs]
    gmst.columns = [fair2mesmer[c] for c in gmst.columns]
    gmst = gmst.rolling(window=21, center=True).mean()
    gmst = gmst.iloc[10:-10]
    gmst = gmst.melt(value_vars=gmst.columns, var_name='run', value_name='gmst', ignore_index=False)

    selected_years[scenario] = {r:{g:list(gmst.loc[(gmst.run == r) & (np.abs(gmst.gmst - g) < 0.05)].index) \
                                    for g in gmts} for r in range(100)}

source = '/mnt/PROVIDE/mesmer-x-processed/regional_averages_realisations'
indicator = 'fwixd'
region = 'PRT'

esms = np.unique([fl.split('_')[-6] for fl in\
     glob.glob(f'{source}/{indicator}/mesmerx_*_*_{indicator}_regional_averages_realisations.nc')])


scens = np.unique([fl.split('_')[-5] for fl in\
     glob.glob(f'{source}/{indicator}/mesmerx_*_*_{indicator}_regional_averages_realisations.nc')])


scenario_dict = {
    'CurPol' : 'curpol-sap',
    'ModAct' : 'modact-sap',
    'Ren' : 'ren',
    'LD' : 'ld',
    'GS' : 'gs',
    'SP' : 'sp',
    'Neg' : 'neg-os-0',
    'ssp119' : 'ssp119-extended',
    'ssp534-over' : 'ssp534-over-extended',
    'Ref_1p5' : 'ref-1p5-extended',
}

def do_one_scenario(scenario):
    x = np.array([])
    fl = f'{source}/{indicator}/mesmerx_CanESM5_{scenario_dict[scenario]}_{indicator}_regional_averages_realisations.nc'
    y = xr.open_dataset(fl)[indicator].loc[:2100,'PRT']
    mask = y.copy() * np.nan
    for run in range(100):
        years = np.array(selected_years[scenario][run][gmt])
        years = years[years <= 2100]
        mask.loc[years,run,:] = gmt
    for esm in esms:
        fl = f'{source}/{indicator}/mesmerx_{esm}_{scenario_dict[scenario]}_{indicator}_regional_averages_realisations.nc'
        y = xr.open_dataset(fl)[indicator].loc[:2100,'PRT'].values
        x = np.append(x, y[mask == gmt].flatten())
    np.savetxt(f"../data_mesmer_x_reversal/tmp/{gmt}_{scenario}.csv", x, delimiter=",")  


for gmt in gmts:
    with Pool(10) as p:
        pool_outputs = list(p.map(do_one_scenario, scenario_dict.keys()))

percentiles = xr.DataArray(dims=['gmt','p'], coords=dict(gmt=gmts, p=np.arange(0,101,1,'int')))
for gmt in gmts:
    x = np.array([])
    for scenario in scenario_dict.keys():
        x = np.append(x, np.loadtxt(f"../data_mesmer_x_reversal/tmp/{gmt}_{scenario}.csv", delimiter=","))
    if len(x) >= 100:
        percentiles.loc[gmt][:] = np.percentile(x, range(101))


xr.Dataset({indicator:percentiles}).to_netcdf(f'../data_mesmer_x_reversal/{indicator}.nc')

