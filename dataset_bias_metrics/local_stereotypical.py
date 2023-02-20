__docformat__ = "numpy"

import numpy as np
import pandas as pd

def lewontinsd(df:pd.DataFrame, x:str, y:str):
    """ Lewontin's D.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the first component to analyze. Should be one of the 
        columns of df
    y: str
        Name of the second component to analyze. Should be one of the 
        columns of df
        
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame corresponding to the full combination of the groups
        in x (as rows) and y (as columns), with each cell having the value of
        the metric for that specific subgroup
    """
    
    with np.errstate(divide='ignore'):
        px = df[x].value_counts(sort=False).sort_index().to_numpy()[:, np.newaxis]
        px = px / px.sum()
        py = df[y].value_counts(sort=False).sort_index().to_numpy()[np.newaxis, :]
        py = py / py.sum()
        cross = pd.crosstab(df[x], df[y])
        real = cross.copy()
        real[:] = cross.to_numpy() / cross.to_numpy().sum()
        expected_mat = px.dot(py)
        final = real.copy()
        final[:] = real - expected_mat
        return final

def duchersz(df:pd.DataFrame, x:str, y:str):
    """ Ducher's Z.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the first component to analyze. Should be one of the 
        columns of df
    y: str
        Name of the second component to analyze. Should be one of the 
        columns of df
        
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame corresponding to the full combination of the groups
        in x (as rows) and y (as columns), with each cell having the value of
        the metric for that specific subgroup
    """
    
    with np.errstate(divide='ignore'):
        px = df[x].value_counts(sort=False).sort_index().to_numpy()[:, np.newaxis]
        px = px / px.sum()
        py = df[y].value_counts(sort=False).sort_index().to_numpy()[np.newaxis, :]
        py = py / py.sum()
        cross = pd.crosstab(df[x], df[y])
        real = cross.copy()
        real[:] = cross.to_numpy() / cross.to_numpy().sum()
        expected_mat = px.dot(py)
        final = real.copy()
        final[:] = real - expected_mat
        positivesden = real.copy()
        positivesden[:] = np.minimum(px, py) - expected_mat
        negativesden = real.copy()
        negativesden[:] = expected_mat - np.maximum(0, px + py - 1)
        final[final > 0] /= positivesden[final > 0]
        final[final < 0] /= negativesden[final < 0]
        return final
        
def npmi(df:pd.DataFrame, x:str, y:str):
    """ Normalized Pointwise Mutual Information.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the first component to analyze. Should be one of the 
        columns of df
    y: str
        Name of the second component to analyze. Should be one of the 
        columns of df
        
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame corresponding to the full combination of the groups
        in x (as rows) and y (as columns), with each cell having the value of
        the metric for that specific subgroup
    """
    
    with np.errstate(divide='ignore'):
        px = df[x].value_counts(sort=False).sort_index().to_numpy()[:, np.newaxis]
        px = px / px.sum()
        py = df[y].value_counts(sort=False).sort_index().to_numpy()[np.newaxis, :]
        py = py / py.sum()
        cross = pd.crosstab(df[x], df[y])
        real = cross.copy()
        real[:] = cross.to_numpy() / cross.to_numpy().sum()
        expected_mat = px.dot(py)
        pmi_mat = real.to_numpy() / expected_mat
        logs = -np.log(real.to_numpy())
        with np.errstate(divide='ignore',invalid='ignore'):
            pmi_mat = np.log(pmi_mat) / logs
        pmi_mat[np.logical_and((real == 0).to_numpy(), (expected_mat > 0))] = -1
        final = real.copy()
        final[:] = pmi_mat
        return final
    
metrics = {
    'NPMI': npmi,
    'Ducher\'s Z': duchersz,
    'Lewonstin\'s D': lewontinsd
}
"""
Dictionary with the full list of metrics, with the keys corresponding to the 
common symbol of each metric.
"""