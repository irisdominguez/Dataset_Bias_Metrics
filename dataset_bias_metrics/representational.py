__docformat__ = "numpy"

import pandas as pd
import numpy as np
from functools import partial
from typing import Callable, Union

# Auxiliar

def truediversity(df: pd.DataFrame, x: str, q: float) -> float:
    """ True Diversity. This function follows a parametric generalization of 
    several of the other metrics.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    q: float
        Degree of the diversity, i.e., 1 for ENS based on Shannon 
        entropy, 2 for Simpson's reciprocal...
    """
    
    p = df.groupby(x).size()
    p = p / np.sum(p)
    if q != 1.0:
        d = np.power(np.sum(np.power(p, q)), 1.0/(1.0-q))
    else:
        d = np.exp(-np.sum(p*np.log(p)))
    return d


# Base metrics

def bergerparker_index(df: pd.DataFrame, x:str) -> float:
    """ Berger Parker index
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    p = df.groupby(x).size()
    return np.max(p) / np.sum(p)

def ens(df: pd.DataFrame, x:str) -> float:
    """ Effective Number of Species
    
    Based on the Shannon entropy.
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    return truediversity(df, x, q=1)

def imbalanceRatio(df: pd.DataFrame, x:str) -> float:
    """ Imbalance Ratio
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    counts = df[x].value_counts()
    return counts.max() / counts.min()

def nsd(df: pd.DataFrame, x:str) -> float:
    """ Normalized Standard Deviation
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    counts = df[x].value_counts()
    if len(counts) == 1:
        return np.NaN
    dist = counts / counts.sum()
    bias = np.std(dist) * len(dist) / np.sqrt(len(dist) - 1)
    unique_values = len(counts)
    return bias

def richness(df: pd.DataFrame, x:str) -> float:
    """ Richness
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    return np.prod(df[x].nunique())

def shannon_diversity(df: pd.DataFrame, x:str) -> float:
    """ Shannon entropy (Shannon diversity index)
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    p = df.groupby(x).size()
    p = p / np.sum(p)
    if len(p) == 1:
        return 0.0
    else:
        return -np.sum(p * np.log(p))

def shannon_evenness(df: pd.DataFrame, x:str) -> float:
    """ Shannon Evenness Index
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    p = df.groupby(x).size()
    p = p / np.sum(p)
    
    n = -np.sum(p * np.log(p))
    d = np.log(p.shape[0])
    if n == 0 and d == 0:
        return np.NaN
    return -np.sum(p * np.log(p)) / np.log(p.shape[0])

def simpsons(df: pd.DataFrame, x:str) -> float:
    """ Simpon's index
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    return 1. / truediversity(df, x, q=2)

def simpsons_diversity(df: pd.DataFrame, x:str) -> float:
    """ Simpson's Diversity Index
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    return 1. - 1. / truediversity(df, x, q=2)

def simpsons_reciprocal(df: pd.DataFrame, x:str) -> float:
    """ Simpson's Reciprocal
        
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    """
    
    return truediversity(df, x, q=2)

# Corrections

def reciprocal(df:pd.DataFrame, 
               x:str, 
               metric:Callable[[pd.DataFrame, str], float]) -> float:
    """ Reciprocal of a metric
    
    This wrapper acts as the reciprocal (1/m) for a metric (m)
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df
    metric: Callable[[pd.DataFrame, str], float]
        Original metric
    
    """
    
    return 1.0 / metric(df, x)

def complementary(df: pd.DataFrame, 
                  x:str, 
                  metric:Callable[[pd.DataFrame, str], float], 
                  supLimit:Union[float,Callable[[pd.DataFrame, str], float]]=1.0) -> float:
    """ Complementary of a metric
    
    This wrapper acts as the complementary (l - m) for a metric (m), with an
    optional superior limit (l)
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to analyze, with each row corresponding to a
        single sample, and each column to a demographic component
    x: str
        Name of the component to analyze. Should be one of the 
        columns of df 
    metric: Callable[[pd.DataFrame, str], float]
        Original metric
    supLimit: Union[float,Callable[[pd.DataFrame, str], float]]
        Superior limit. Can be a float value, or a secondary metric. If it is 
        a secondary metric, it will be calculated on the fly with the same
        arguments
    """
    
    supLimit = supLimit(df, x) if callable(supLimit) else supLimit
    return supLimit - metric(df, x)

metrics = {
    'R': richness,
    'ENS': ens,
    'D': simpsons,
    '1/D': simpsons_reciprocal,
    '1-D': simpsons_diversity,
    'H': shannon_diversity,
    'SEI': shannon_evenness,
    'NSD': nsd,
    'IR': imbalanceRatio,
    'BP': bergerparker_index,
}
"""
Dictionary with the full list of metrics, with the keys corresponding to the 
common symbol of each metric
"""

metrics_as_diversity = {
    'R': richness,
    'ENS': ens,
    '1/D': simpsons_reciprocal,
    '1-D': simpsons_diversity,
    'H': shannon_diversity,
    'SEI': shannon_evenness,
    '1-NSD': partial(complementary, metric=nsd),
    '1/IR': partial(reciprocal, metric=imbalanceRatio),
    '1-BP': partial(complementary, metric=bergerparker_index),
}
"""
Dictionary with the metrics redefined in their diversity-measuring form, with 
the keys corresponding to the common symbol of each metric
"""

metrics_as_bias = {
    'R-ENS': partial(complementary, metric=ens, supLimit=richness),
    'D': simpsons,
    'R-1/D': partial(complementary, metric=simpsons_reciprocal, supLimit=richness),
    '1-SEI': partial(complementary, metric=shannon_evenness),
    'NSD': nsd,
    'IR': imbalanceRatio,
    'BP': bergerparker_index,
}
"""
Dictionary with the metrics redefined in their bias-measuring form, with 
the keys corresponding to the common symbol of each metric
"""