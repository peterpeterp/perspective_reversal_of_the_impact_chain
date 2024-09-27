import sys,inspect,gc, string
import numpy as np
import xarray as xr
from scipy import stats
from scipy.optimize import curve_fit, fmin, fminbound, minimize, rosen_der
from scipy import special
from sklearn.linear_model import LinearRegression
from scipy.stats import genextreme
from sklearn.metrics import r2_score
from statsmodels.distributions.empirical_distribution import ECDF
from collections import Counter

class multi_params_wrapper():
    def __init__(self, params_multi, multi_name='esm'):
        self._params_multi = params_multi
        self._multi_axis = params_multi.index
        self._multi_name = multi_name

    def wrap(self, **kwargs):
        kwargs_for_function = {k:v for k,v in kwargs.items() if k not in ['func','dim_names']}
        out = np.array([kwargs['func'](params=self._params_multi.loc[i], **kwargs_for_function) for i in self._multi_axis])
        if 'dim_names' in kwargs.keys():
            dim_names = kwargs['dim_names']
        else:
            dim_names = [list(string.ascii_lowercase)[i] for i in range(len(out.shape)-1)]
        dims = [self._multi_name] + dim_names
        coords = {d:np.arange(out.shape[i]) for i,d in enumerate(dims)}
        coords[self._multi_name] = self._multi_axis
        return xr.DataArray(out, dims=dims, coords=coords)

class non_stat_dstr():
    def __init__(self, param_names):
        self._param_names = param_names

    def add_data(self, x, gmt, scenarioMIP=True):
        if scenarioMIP:
            xx = x.loc[:,'2015':'2100'].values.flatten()
            xx = np.append(xx, x.loc[:,:'2014'].mean('id_').values.flatten())
            gg = gmt.loc[:,'2015':'2100'].values.flatten()
            gg = np.append(gg, gmt.loc[:,:'2014'].mean('id_').values.flatten())
            valid = np.isfinite(xx) & np.isfinite(gg)
            self._x = xx[valid]
            self._gmt = gg[valid]
        
        else:
            assert False, 'at the moment this is only implemented for scenarioMIP=True'
            
        self.get_weights_for_gmt()
        
    def get_weights_for_gmt(self):
        rounded = self._gmt.round(1)
        r = rounded.copy()
        counter = Counter(r)
        weights = self._gmt.copy()
        weights[:] = 0
        for v,c in counter.items():
            if c > 1000:
                weights[rounded == v] = 1 / c

        self._weights = weights / weights.sum()

    def gmt_slice(self, g, acceptable_sample_size=30, gmt_deviation_tolerance=0.05):
        x, gmt = self._x, self._gmt
        x_slice = x[np.abs(gmt-g) < gmt_deviation_tolerance]
        gmt_slice = gmt[np.abs(gmt-g) < gmt_deviation_tolerance]
        if len(gmt_slice) > acceptable_sample_size:
            if np.abs(np.median(gmt_slice) - g) < gmt_deviation_tolerance:
                return gmt_slice,x_slice
        return None,None
    
    ######################################
    # fitting
    ######################################
    
    def neg_loglike(self, params):
        # negative log likelihood (for fit)
        return -self.loglike(params)
    
    def fit(self, p0, bounds=None, method='Powell'):
        res = minimize(self.neg_loglike, p0, bounds=bounds, method=method)
        return res.x, res    

    def BIC(self, p, k):
        return k*np.log(np.isfinite(self._gmt).sum()) -2*self.loglike(p)

    def AIC(self, p, k):
        return 2*k -2*self.loglike(p)
    
    def fit_iteratively(self, p, fitting_steps=[[0,2,3,4], [5], [1]], bound_frac=0.1, method='Powell'):
        '''
        This function allows to iteratively fit parameters of the distribution.
        
        fitting_steps: list of lists of parameter indices
        bound_frac: allowed relative deviation from previously fitted values 
        method: minimization algorithm used in scipy.optimize.minimize
        
        '''
        # set all parameter bounds to 0,0
        bounds = [(0,0) for _ in self._param_names]
        # go through the list of fitting_steps
        for variable_parameter_ids in fitting_steps:
            # set bounds for the (new) parameters to be fitted to -inf,inf
            for i in variable_parameter_ids:
                bounds[i] = (-np.inf, np.inf)
            # do a fit
            p,res = self.fit(p, bounds, method)
            # limit the bounds to the previous estimate allowing for deviations in the range of bound_frac
            bounds = [(v-np.abs(bound_frac*v),v+np.abs(bound_frac*v)) for v in p]
        return p, res  
    
    def try_different_methods_until_success(self, p, fitting_steps=[[0,2,3,4], [5], [1]], bound_frac=0.1):
        '''
        this is a wrapper for "fit_iteratively" that goes through minimization algorithms until an algorithm succeeds.
        only in few cases this is needed.
        in most cases, the "Powell" method works directly
        '''
        for method in [
            'Powell',
            'Nelder-Mead',
            'CG',
            'BFGS',
            'Newton-CG',
            'L-BFGS-B',
            'TNC',
            'COBYLA',
            'SLSQP',
            'trust-constr',
            'dogleg',
            'trust-ncg',
            'trust-exact',
            'trust-krylov']:
            popt,res = self.fit_iteratively(p, fitting_steps=fitting_steps, bound_frac=bound_frac, method=method)
            if res.success:
                return popt
        return [np.nan] * len(p)

    ######################################
    # convenience functions
    ######################################
    def gmt_for_which_event_has_prob(self, threshold, prob, params=None):
        if params is None:
            params = self._params
        for gmt in np.arange(0,10,0.01):
            if 1-self.cdf(gmt, threshold, params) > prob:
                return gmt
        return np.nan
    
    def threshold_exceed_prob_vs_gmts(self, threshold, gmt, params=None):
        if params is None:
            params = self._params
        if np.any(np.isnan(gmt)):
            print(gmt)
        cdf = self.cdf(gmt, threshold, params)
        cdf[np.isnan(cdf)] = 0
        return xr.DataArray(1 - cdf, 
                            dims=['gmt'], coords=dict(gmt=gmt))
            
    def ppf_for_gmts(self, gmt, q, params=None):
        if params is None:
            params = self._params
        ppf = self._func.ppf(q, *self.param_wrapper(params, gmt))
        return xr.DataArray(ppf, dims=['gmt'], coords=dict(gmt=gmt))

class skewN(non_stat_dstr):
    def __init__(self):
        param_names = list(np.array([(p+'_0',p+'_1') for p in ['shape','loc','scale']]).flatten())
        super().__init__(param_names)
        self._func = stats.skewnorm
    
    def param_wrapper(self, p, gmt):
        shape = p[0] + p[1]*gmt
        loc = p[2] + p[3]*gmt
        scale = p[4] + p[5]*gmt
        return shape, loc, scale
            
    def cdf(self, gmt, x, params):
        shape, loc, scale = self.param_wrapper(params, gmt)
        return stats.norm.cdf(x, loc=loc, scale=scale) - 2 * special.owens_t((x - loc) / scale, shape)
    
    def pdf(self, gmt, xaxis, params):
        shape, loc, scale = self.param_wrapper(params, gmt)
        return stats.skewnorm.pdf(xaxis, shape, loc, scale)
    
    def loglike(self, params):
        shape, loc, scale = self.param_wrapper(params, self._gmt)
        
        ll = np.array(stats.skewnorm.logpdf(self._x, 
                                 a=shape, 
                                 loc=loc, 
                                 scale=scale))
        
        return np.average(ll, weights=self._weights)
    
class normal(non_stat_dstr):
    def __init__(self):
        param_names = list(np.array([(p+'_0',p+'_1') for p in ['loc','scale']]).flatten())
        super().__init__(param_names)
        self._func = stats.norm
    
    def param_wrapper(self, p, gmt):
        loc = p[0] + p[1]*gmt
        scale = p[2] + p[3]*gmt
        return loc, scale
            
    def cdf(self, gmt, x, params):
        loc, scale = self.param_wrapper(params, gmt)
        return stats.norm.cdf(x, loc=loc, scale=scale)
    
    def pdf(self, gmt, xaxis, params):
        loc, scale = self.param_wrapper(params, gmt)
        return stats.norm.pdf(xaxis, loc, scale)
    
    def loglike(self, params):
        loc, scale = self.param_wrapper(params, self._gmt)        

        ll = np.array(stats.norm.logpdf(self._x, 
                                 loc=loc, 
                                 scale=scale))
        
        return np.average(ll, weights=self._weights)

