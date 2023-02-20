__docformat__ = "numpy"

import pandas as pd
import numpy as np
import scipy.stats

def chisq(df:pd.DataFrame, x:str, y:str):
    """ Chi Squared metric.
    
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
    float
        Result of the metric
    """
    
    cross = pd.crosstab(df[x], df[y]).to_numpy()
    stat, p, dof, expected= scipy.stats.contingency.chi2_contingency(cross)
    return p

def cramersv(df:pd.DataFrame, x:str, y:str):
    """ Cramer's V.
    
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
    float
        Result of the metric
    """
    
    cross = pd.crosstab(df[x], df[y]).to_numpy()
    if min(cross.shape) == 1:
        return np.nan
    return scipy.stats.contingency.association(cross, method='cramer', correction=True)

def nmi(dff, x, y):
    """ Normalized Mutual Information.
    
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
    float
        Result of the metric
    """
    
    with np.errstate(divide='ignore', invalid='ignore'):
        df = dff.copy()
        px = df[x].value_counts(sort=False).sort_index().to_numpy()[:, np.newaxis]
        px = px / px.sum()
        py = df[y].value_counts(sort=False).sort_index().to_numpy()[np.newaxis, :]
        py = py / py.sum()
        real = pd.crosstab(df[x], df[y])
        real[:] = real.to_numpy() / real.to_numpy().sum()
        expected_mat = px.dot(py)
        
        pmi_mat = real.to_numpy() / expected_mat
        numerator = real.to_numpy() * np.log(pmi_mat)
        numerator[real == 0] = 0
        denominator = real.to_numpy() * np.log(real.to_numpy())
        denominator[real == 0] = 0
        final = np.sum(numerator) / (-np.sum(denominator))
        return final 

def pearson(df:pd.DataFrame, x:str, y:str):
    """ Pearson's Correlation Coefficient.
    
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
    float
        Result of the metric
    """
    
    cross = pd.crosstab(df[x], df[y]).to_numpy()
    return scipy.stats.contingency.association(cross, method='pearson', correction=True)

def theilsu(df:pd.DataFrame, x:str, y:str):
    """ Theil's U metric. Forward direction of the metric, from x to y.
    
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
    float
        Result of the metric
    """
    
    cross = pd.crosstab(df[x], df[y]).to_numpy()
    dx = cross.sum(axis=1)/cross.sum(axis=None)
    hx = -np.sum(dx * np.log(dx))
    dxy = cross/cross.sum(axis=None)
    dxgiveny = cross /cross.sum(axis=0)
    hxy = -np.sum(dxy * np.log(dxgiveny, out=np.zeros_like(dxgiveny), where=(dxgiveny!=0)))
    if (hx - hxy) == 0 and hx == 0:
        return np.nan
    return (hx - hxy) / hx

def theilsurev(df:pd.DataFrame, x:str, y:str):
    """ Theil's U metric. Wrapper for the reverse direction of the metric, from
    y to x.
    
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
    float
        Result of the metric
    """
    
    return theilsu(df, y, x)

def tschuprow(df:pd.DataFrame, x:str, y:str):
    """ Tschuprow's T.
    
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
    float
        Result of the metric
    """
    
    cross = pd.crosstab(df[x], df[y]).to_numpy()
    if min(cross.shape) == 1:
        return np.nan
    return scipy.stats.contingency.association(cross, method='tschuprow', correction=True)

metrics = {
    'ϕ_C': cramersv,
    'T': tschuprow,
    'C': pearson,
    'U→': theilsu,
    'U←': theilsurev,
    'NMI': nmi
}
"""
Dictionary with the full list of metrics, with the keys corresponding to the 
common symbol of each metric.
"""