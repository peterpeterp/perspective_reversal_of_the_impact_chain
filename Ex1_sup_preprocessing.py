import sys,glob,datetime,os,cftime,importlib,gc
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import _pre_oneRun as _pre_oneRun; importlib.reload(_pre_oneRun); from _pre_oneRun import oneRun

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

import SBCK

import cartopy

import _paths as _paths; importlib.reload(_paths)

def relDiff(val,ref):
    return (val - ref) / ref * 100

def absDiff(val,ref):
    return val - ref

    
def subselect_month(x, month):
    month_mask = np.isin(x.time.dt.month, [month])
    month_mask = xr.DataArray(month_mask, coords=dict(time=x.time), dims=['time'])
    return x.where(month_mask, drop=True)

unit_d = dict(tas='K', tasmax='K', tasmin='K', tnnETCCDI='K', txxETCCDI='K', tos='K', pr='mm', rx1dayETCCDI='mm', ua='m/s', va='m/s')

class oneRun():
    def __init__(self,scenario,model,run,**kwargs):
        self._scenario = scenario
        self._model = model
        self._run = run
        self._SMR = scenario+'_'+model+'_'+run

        defaults = dict(
            level=None,
            time_format=None,
            mask=None,
            shape_for_mask=None,
            closest_grid=None,
            extent=None,
            regrid_prec=1,
            verbose=False,
            realm=None,
            months=None,
            preprocessing_function=None,
        )
        defaults.update(kwargs)
        for k,v in defaults.items():
            self.__dict__['_'+k] = v
            
        defaults.update(kwargs)
        for k,v in defaults.items():
            self.__dict__['_'+k] = v    
        
        self._history = ''
        
    def inherit_attributes_from_ensemble(self, ensembleObject):
        for k,v in ensembleObject.__dict__.items():
            if k in ['_'+w for w in \
                     ['indicator','scenario','verbose','level','time_format','mask_search_path','realm',
                      'closest_grid', 'cutoff_years', 'required_years', 'preprocessing_function',
                      'region_name','shape_for_mask','regrid_prec',
                      'extent','regrid','months','overwrite','source']]:
                self.__dict__[k] = v

        
    def get_files(self, data_paths = ['???','???']):            
        SMRs = []
        self._hist_files, self._ssp585_files = [],[]
        if self._source == 'CMIP6':
            if self._indicator in ['tas','pr','ua','va','uas','sos','tos','snc', 'tasmax', 'tasmin','abs550aer']:
                if self._realm is None:
                    if self._indicator in ['tas','pr','ua','va','uas']:
                        realm = 'Amon'
                    elif self._indicator in ['sos','tos']:
                        realm = 'Omon'
                    elif self._indicator in ['snc']:
                        realm = 'SImon'
                    elif self._indicator in ['abs550aer']:
                        realm = 'AERmon'
                    self._realm = realm

                self._scen_files = []
                for path in data_paths:
                    self._scen_files += glob.glob(path+'/*/*/%s/%s/%s/%s/%s/*/*/*'
                                             %(self._model,self._scenario,self._run,self._realm,self._indicator))

                if 'ssp' in self._scenario:
                    self._hist_files = []
                    for path in data_paths:
                        self._hist_files += glob.glob(path+'/*/*/%s/%s/%s/%s/%s/*/*/*'
                                             %(self._model,'historical',self._run,self._realm,self._indicator))
                if 'ssp534-over' == self._scenario:
                    self._ssp585_files = []
                    for path in data_paths:
                        self._ssp585_files += glob.glob(path+'/*/*/%s/%s/%s/%s/%s/*/*/*'
                                             %(self._model,'ssp585',self._run,self._realm,self._indicator))

                SMRs += ['_'.join([fl.split('/')[i] for i in [-7,-8,-6]]) for fl in self._scen_files]

        return SMRs

    def treat_data(self, var, fl):
        # this is the main preprocessing
        # it has to be in a separate function because it is applied to each subfile

        if 'lat' not in var.dims and 'latitude' in var.dims:
            var = var.rename(dict(latitude='lat', longitude='lon'))

        if 'lat' not in var.dims:
            if self._verbose: print('no lat')
            return None
        
        if self._closest_grid is not None:
            var = var.sel(self._closest_grid, method='nearest')
            var = var.squeeze()
        
        # subselect months
        if self._months is not None:
            month_mask = np.isin(var.time.dt.month, self._months)
            month_mask = xr.DataArray(month_mask, coords=dict(time=var.time), dims=['time'])
            var = var.where(month_mask, drop=True)
                
        return var

    def get_projections(self):
        self.get_files()
        self._data = None
        
        # interrupt if mask is required and missing
        if self._mask is not None:
            if os.path.isfile(self._mask['search_path'] %(self._model)) == False:
                if self._verbose: print('mask missing for %s' %(self._model))
                return 1
            
        if self._verbose:
            print(self._scen_files)
        if len(self._scen_files) > 0:
            files = sorted(self._hist_files) + sorted(self._scen_files)

            # check whether there will be enough years
            if self._required_years is not None:
                years = np.array([])
                for fl in files:
                    years = np.append(years, np.arange(int(''.join(fl.split('_')[-1][:4])),int(''.join(fl.split('_')[-1].split('-')[-1][:4]))+1))
            
                if np.sum(np.isin(years,self._required_years)) < len(self._required_years):
                    if self._verbose: print('not enough years')
            
            l,years = [],[]
            for fl in files:
                if self._verbose: print(fl)
                var = xr.open_dataset(fl)[self._indicator]
                if var is not None:
                    yrs = np.array([int(str(t)[:4]) for t in var.time.values])
                    if self._cutoff_years is not None:
                        var = var[np.isin(yrs,np.arange(self._cutoff_years[0],self._cutoff_years[1]+1,1))]
                    if var.time.shape[0] > 0:
                        try:
                            var = self.treat_data(var,fl)
                        except:
                            var = None
                        if self._preprocessing_function is not None and var is not None:
                            var = self._preprocessing_function(var)
                        if var is not None:
                            if np.sum(np.isin(yrs, np.array(years).flatten())) < len(yrs):
                                l.append(var)
                                years += [int(str(t)[:4]) for t in var.time.values]
                                
            if 'ssp' in self._scenario:
                # in ssp534-over the years 2015-2039 have to be taken from ssp585
                missing_years = np.array([yr for yr in np.arange(2015,2050,1,'int') if yr not in np.array(years)])
                if self._verbose: print('missing years', missing_years)
                if 2020 in missing_years and self._scenario == 'ssp534-over':
                    for fl in self._ssp585_files:
                        var = xr.open_dataset(fl)[self._indicator]
                        var_years = np.array([int(str(t)[:4]) for t in var.time.values])
                        var = var[np.isin(var_years,missing_years)]
                        if var.shape[0] > 0:
                            var = self.treat_data(var,fl)
                            l.append(var)
                            if self._verbose: print('adding years from ssp585',np.unique([int(str(t)[:4]) for t in var.time.values]))

                # this is needed because 2265 is not supported by np.datetime64 0_o
                time_formats = list(set([type(o.time.values[0]) for o in l]))
                if self._time_format is not None or len(time_formats) != 1:
                    if self._time_format is None:
                        self._time_format = [f for f in time_formats if f != np.datetime64][0]
                    for i,tmp in enumerate(l):
                        new_time = np.array([self._time_format(int(str(t)[:4]),int(str(t)[5:7]),int(str(t)[8:10])) for t in tmp.time.values])
                        l[i] =  tmp.assign_coords(time = new_time)
                                    

            if 'abrupt' in self._scenario or self._scenario == 'piControl':
                if len(l) > 0:
                    first_year = int(str(l[0].time.values[0])[:4])
                    for i,tmp in enumerate(l):
                        # there seem to be different standards for the time axis in this scenario
                        new_time = np.array([self._time_format(int(str(t)[:4]) -first_year, int(str(t)[5:7]), 15) 
                                             for t in tmp.time.values])
                        l[i] =  tmp.assign_coords(time = new_time)

            # subselect months
            
                        
            # concat all
            if len(l) > 0:
                out = xr.concat(l, dim='time', coords='minimal', compat='override')
                out = out.sortby('time')
                out = out.drop_duplicates('time', keep='first')
                #if self._required_years is not None:
                #    if np.sum(np.isin(np.unique(out.time.dt.year),self._required_years)) < len(self._required_years):
                #        if self._verbose: print('not enough years')
                #        return 1
                out = out.expand_dims(dim='SMR', axis=0)
                out = out.assign_coords(SMR=[self._SMR])
                self._data = out
                return 0 
        
        if self._verbose: print('something went wrong')
        return 1


        
class ensemble():
    def __init__(self,**kwargs):
        defaults = dict(
            indicator=None,
            level=None,
            realm=None,
            time_format=cftime.DatetimeNoLeap,
            mask_search_path=None,
            region_name=None,
            shape_for_mask=None,
            closest_grid=None,
            extent=None,
            cutoff_years=None,
            required_years=None,
            regrid=False,
            months=range(1,13),
            gmt=None,
            preprocessing_function=None,
            spaceAggr=None,
            regrid_prec=2,
            verbose=False,
            scenario = '',
            overwrite=False,
            source='CMIP6',
        )
        defaults.update(kwargs)
        for k,v in defaults.items():
            self.__dict__['_'+k] = v
                        
        if 'unit' not in defaults.keys() and self._indicator is not None:
            self._unit = unit_d[self._indicator]
        
        
        if self._mask_search_path is not None:
            self._regionTag = self._region_name + '-'+self._mask_search_path.split('_')[-2]
        elif self._shape_for_mask is not None:
            self._regionTag = self._region_name
        elif self._closest_grid is not None:
            self._regionTag = self._region_name
        elif self._extent is not None:
            self._regionTag = '%sto%sE%sto%sN%sgs' %(tuple(self._extent+[self._regrid_prec]))
        else:
            self._regionTag = 'global'
        
        self._tag = self._regionTag
        if self._source == 'mesmer':
            self._tag += '_source.mesmer'
        self._tag += ''.join([k+'.'+str(self.__dict__[k]) \
                            for k in ['_'+w for w in ['scenario','indicator','level']] \
                            if self.__dict__[k] is not None])
        
        if self._realm is not None:
            self._tag += '_' + self._realm
        

        self._smr_dim = 'SMR'
        self._history = ''

    def copy(self):
        class_type = type(self)
        new = class_type()
        for k,v in self.__dict__.items():
            new.__dict__[k] = v
        return new

    def convert_unit(self, shift=None, factor=None, unit=None):
        if factor is not None:
            self._data = self._data * factor
        if shift is not None:
            self._data = self._data + shift
        if unit is not None:
            self._unit = unit
        
    def get_SMRs(self, scenarios, exclude=None, onlyOneRun=False):
        SMRs = []
        for scenario in scenarios:
            dummy = oneRun(scenario,'*','*',indicator=self._indicator)
            dummy.inherit_attributes_from_ensemble(self)
            dummy._scenario = scenario
            SMRs += dummy.get_files()
        
        if exclude is not None:
            SMRs = np.array([smr for smr in SMRs if exclude not in smr])

        if onlyOneRun:
            smrs,sms = [],[]
            for smr in SMRs:
                sm = '_'.join(smr.split('_')[:2])
                if sm not in sms:
                    sms.append(sm)
                    smrs.append(smr)
            SMRs = smrs
        self._SMRs = np.unique(SMRs)


class ts_ensemble(ensemble):
    def __init__(self, **kwargs):
        ensemble.__init__(self, **kwargs)
    
    def get_projections(self, outFile=None, get_missing=False):
        if self._spaceAggr is not None:
            self._tag += '_spaceAggr.'+self._spaceAggr
            
        if outFile is None:
            outFolder = _paths.dataPath + 'regional/%s/' %(self._regionTag)
            outFile = _paths.dataPath + 'regional/%s/%s.nc' %(self._regionTag,self._tag)
        if self._verbose: print(outFile)
        
        if self._overwrite and os.path.isfile(outFile):
            os.remove(outFile)

        if os.path.isfile(outFile):
            self._data = xr.load_dataset(outFile)['data']
            loaded_SMRs = self._data.SMR.values
        else:
            loaded_SMRs = []
            
        if self._verbose: print(loaded_SMRs)
        
        for SMR in self._SMRs:
            if SMR not in loaded_SMRs and get_missing:
                if self._verbose: print(SMR)
                scenario,model,run = SMR.split('_')

                a = oneRun(scenario,model,run)
                a.inherit_attributes_from_ensemble(self)
                a._scenario = scenario
                #try:
                if True:
                    a.get_projections()
                    var = a._data
                    if var is not None:
                        print(var[0].mean())
                        print(var.lat,var.lon)
                        # area average - this could be another statistic over the region
                        if self._spaceAggr == 'wAv':
                            weights = np.cos(np.deg2rad(var.lat))
                            var = var.weighted(weights).mean(('lat','lon'))
                        else:
                            pass

                        # storing
                        if '_data' in self.__dict__.keys():
                            self._data = xr.concat([self._data.reset_coords(drop=True),var], dim='SMR', coords='minimal', compat='override')
                        else:
                            self._data = var
                            
                        print(self._data)
                        asdasd

                        if os.path.isdir(outFolder) == False:
                            os.makedirs(outFolder, exist_ok=True)
                        xr.Dataset({'data':self._data}).to_netcdf(outFile)
                #except:
                #    print('FAILURE')
        
    def bias_correction(self, obsData, months, adjPeriod = [1981, 2010], bcPeriod=[2023,2100], overwrite=False):
        outFile = _paths.dataPath + 'regional/%s/%s' %(self._regionTag,self._tag)
        outFile += '_months'+'-'.join([str(m) for m in months])
        outFile += '_adjPeriod'+'-'.join([str(m) for m in adjPeriod])
        outFile += '_bcPeriod'+'-'.join([str(m) for m in bcPeriod])
        outFile += '.nc'
        
        if os.path.isfile(outFile) and overwrite==False:
            self._dataBC = xr.load_dataset(outFile)['dataBC']
        
        else:
            month_mask = np.isin(self._data.time.dt.month, months)
            month_mask = xr.DataArray(month_mask, coords=dict(time=self._data.time), dims=['time'])
            self._dataBC = self._data.where(month_mask, drop=True).loc[:,str(bcPeriod[0]):str(bcPeriod[1])].copy() * np.nan
            # go through months
            for month in months:
                month_mask = np.isin(obsData.time.dt.month, [month])
                month_mask = xr.DataArray(month_mask, coords=dict(time=obsData.time), dims=['time'])
                # obs values over adjPeriod
                obsC = obsData.where(month_mask, drop=True).loc[str(adjPeriod[0]):str(adjPeriod[1])].values.flatten()
                for smr in self._data.id_.values:
                    simC = self._data.loc[smr, str(adjPeriod[0]):str(adjPeriod[1])]
                    month_mask = np.isin(simC.time.dt.month, [month])
                    month_mask = xr.DataArray(month_mask, coords=dict(time=simC.time), dims=['time'])
                    # sim values over adjPeriod
                    simC = simC.where(month_mask, drop=True).values.flatten()
                    if np.all(np.isfinite(simC)):
                        #go through years
                        for year in np.unique(self._dataBC.time.dt.year.values):
                            # print(month, smr, year)
                            # select adjust period
                            simP = self._data.loc[smr,str(year-15):str(year+15)]
                            month_mask = np.isin(simP.time.dt.month, [month])
                            month_mask = xr.DataArray(month_mask, coords=dict(time=simP.time), dims=['time'])
                            simP = simP.where(month_mask, drop=True)
                            bcSimP_xarray = simP.copy() * np.nan
                            simP = simP.values.flatten()
                            if np.all(np.isfinite(simP)):
                                qdm = SBCK.QDM(delta="additive")
                                qdm.fit(obsC.reshape(-1,1), simC.reshape(-1,1), simP.reshape(-1,1))
                                bcSimP, bcSimC = qdm.predict(simP.reshape(-1,1),simC.reshape(-1,1))
                                bcSimP_xarray.values = np.array(bcSimP).squeeze()
                                self._dataBC.loc[smr][(self._dataBC.time.dt.year == year) & (self._dataBC.time.dt.month == month)] = \
                                      bcSimP_xarray[(bcSimP_xarray.time.dt.year == year) & (bcSimP_xarray.time.dt.month == month)]

            xr.Dataset({'dataBC':self._dataBC}).to_netcdf(outFile)




class time_series_to_be_reversed():
    def __init__(self, x, model, scenarios, verbose=False):
        self._model = model
        self._scenarios = scenarios
        self._x = x
        self._verbose = False
                
    def load_data_peter(self, imp=None, load_tmp=True, get_missing=True, onlyOneRun=False):
            
        if self._verbose:
            print('getting x')
        self._x._tag += '_'+self._model
        self._x.get_SMRs(self._scenarios)
        SMRs = [smr for smr in self._x._SMRs if smr.split('_')[1] == self._model]
        if onlyOneRun:
            smrs,scens = [],[]
            for smr in SMRs:
                if smr.split('_')[0] not in scens:
                    scens.append(smr.split('_')[0])
                    smrs.append(smr)
            SMRs = smrs
        
        self._x._SMRs = SMRs
        self._x.get_projections(get_missing=get_missing)
        
        if '_data' not in self._x.__dict__.keys():
            if self._verbose:
                print('no data')
            return False
        
        if self._verbose:
            print('getting GMT')
        gmt = _pre_ensemble.ts_ensemble(indicator='tas', spaceAggr='wAv', source=self._x._source, required_years=self._x._required_years)
        gmt._tag += '_'+self._model
        gmt._SMRs = SMRs
        gmt.get_projections(get_missing=get_missing)
        gmt.aggregateOverYear()
        if np.isfinite(gmt._data.loc[:,1995:2014].mean('year').mean()):
            gmt._data = gmt._data.loc[:,1850:2100]
            gmt._data = gmt._data - gmt._data.loc[:,1995:2014].mean('year') + 0.85
        
        self._gmt = gmt
        if 'SMR' in self._gmt._data.dims:
            self._gmt._data = self._gmt._data.rename({'SMR':'id_'})
        if 'SMR' in self._x._data.dims:
            self._x._data = self._x._data.rename({'SMR':'id_'})
    
    
        #with open(filename, 'wb') as fl:
        #    pickle.dump({'x':self._x, 'gmt':self._gmt}, fl)
        return True
    
        
    def clean_data(self):
        useful_smrs = []
        for smr in self._x.id_.values:
            try:
                assert self._x.loc[smr].min() > -250, \
                'unrealistic values in this SMR'
                    
                useful_smrs.append(smr)
            except AssertionError as e:
                if self._verbose:
                    print(smr, '-'*5, e)
                print(smr, '-'*5, e)
        
        self._x = self._x.loc[useful_smrs]     
        self._gmt = self._gmt.loc[useful_smrs]       
            
        
    def finalize_preprocessing_daily(self):
        if '_dataBC' in self._x.__dict__.keys():
            x = self._x._dataBC
            self._xRaw = self._x._data
        else:
            x = self._x._data
            
        self._gmt_smoo = self._gmt._data.rolling(year=31, center=True).mean()
        
        ids = x.id_.values[np.isin(x.id_.values, self._gmt._data.id_.values)]
        x = x.loc[ids,:]
        
        gmt = x.copy() * np.nan
        gmt_smoo = x.copy() * np.nan
        for id_ in ids:
            for year in np.unique(x.time.dt.year.values):
                gmt.loc[id_,str(year)] = self._gmt._data.loc[id_,year]
                gmt_smoo.loc[id_,str(year)] = self._gmt_smoo.loc[id_,year]

        self._x = x
        self._gmt = gmt
        self._gmt_smoo = gmt_smoo
            
