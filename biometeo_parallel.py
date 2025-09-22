import numpy as np
import importlib
from multiprocessing import Pool 
from tqdm import tqdm
from functools import partial
from itertools import product
import xarray as xr

# Define the wrapper function at the module level

def func_wrapper_single (func, params, result_key, ignore_errors):
    try:
        res = func (**params)
    except Exception as err:
        if not ignore_errors:
            print (err)
            print (params)
            raise err
        else:
            return np.nan
    if result_key is not None:
        return res[result_key]
    else:
        return res

def func_wrapper4ds(params, result_key, ignore_errors, static_params = None):
    ds = params['ds']
    func = params['func']

    if isinstance (func, str):
        func_name = func.split('.')[-1]
        module_name = func[0:-len(func_name)-1]
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

    if static_params is not None:
        func = partial(func, **static_params)

    all_idx = np.atleast_2d(params['idx'])

    results = [func_wrapper_single(func, {key:float(ds[key][tuple(idx)]) for key in ds.keys()}, result_key, ignore_errors) for idx in all_idx]

    return np.array(results)

def compute_parallel(params4chunks, n_jobs, result_key=None, ignore_errors=False, use_tqdm=True, tqdm_desc='', static_params = None):
    wrapped = partial(func_wrapper4ds, result_key=result_key, ignore_errors=ignore_errors, static_params=static_params)
    
    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            if use_tqdm:
                results = list(tqdm(pool.imap(wrapped, params4chunks), total=len(params4chunks), desc=tqdm_desc))
            else:
                results = list(pool.imap(wrapped, params4chunks))
    else:
        if use_tqdm:
            results = list(tqdm(map(wrapped, params4chunks), total=len(params4chunks), desc=tqdm_desc))
        else:
            results = list(map(wrapped, params4chunks))
    
    return np.concatenate(results)


def compute4xarray_ds (func, ds, params:dict, n_jobs, n_chunks=None, result_key=None, ignore_errors = False, use_tqdm=True, tqdm_desc='', min_chunk_size=2, static_params = None):
    
    first_param = ds[list(params.values())[0]]

    shape = first_param.shape

    all_indices = np.array(list(product(*[range(dim) for dim in shape])))

    ds2dict = {func_var_name: ds[param_var_name].values for func_var_name, param_var_name in params.items()}

    if n_chunks is not None and n_chunks < len(all_indices)*min_chunk_size:
        all_indices_chunked = np.array_split(all_indices, n_chunks)
    else:
        all_indices_chunked = all_indices
    
    ds2dict = [{'func': func, 'ds': ds2dict, 'idx': idx_chunk} for idx_chunk in all_indices_chunked]

    results = compute_parallel(ds2dict, n_jobs, result_key, ignore_errors, use_tqdm, tqdm_desc, static_params)

    results = results.reshape(shape)

    results = xr.DataArray(results, coords=first_param.coords, dims=first_param.dims)
    return results
    
