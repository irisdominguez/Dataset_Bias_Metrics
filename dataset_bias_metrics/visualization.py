__docformat__ = "numpy"

import matplotlib.pyplot as plt
import numpy as np

def plotTable(df, normalizeAxis=1, sort='ascending', barWHRatio=2, figsize=None):
    """ Plots a pandas DataFrame with a bar display of each value as a 
    background.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to plot.
    normalizeAxis: int
        Axis to normalize across, dividing each value by the maximum of the row
        or column.
    sort: str
        Wheter to sort the rows or columns in the non-normalized axis in 
        ascending or descending order (regarding the mean of the normalized
        values). If None, do not sort
    barWHRatio: int
        Ratio of the width to the height of the background bars
    figsize: (float, float) or None
        Figure size. If None, autocalculates considering both the number of
        rows and columns of the dataset
    """
    
    # Normalized data for visualization
    dfnorm = df.copy()
    
    if normalizeAxis == 1:
        maxs = df.max(axis=1)
        for r in df.index:
            dfnorm.loc[r] = dfnorm.loc[r].apply(lambda x: x / maxs[r])
        
        # Sort by the secondary axis over the normalized values
        indexes = range(dfnorm.shape[1])
        means = dfnorm.mean(axis=0)
        if sort=='ascending':
            indexes = np.argsort(means)
        elif sort=='descending':
            indexes = np.argsort(means)[::-1]
        dfnorm = dfnorm.iloc[:, indexes]
        df = df.iloc[:, indexes]
    elif normalizeAxis == 0:
        maxs = df.max(axis=0)
        for c in df.columns:
            dfnorm.loc[:, c] = dfnorm.loc[:, c].apply(lambda x: x / maxs[c])
        
        # Sort by the secondary axis over the normalized values
        indexes = range(dfnorm.shape[0])
        means = dfnorm.mean(axis=1)
        if sort=='ascending':
            indexes = np.argsort(means)
        elif sort=='descending':
            indexes = np.argsort(means)[::-1]
        dfnorm = dfnorm.iloc[indexes]
        df = df.iloc[indexes]


    # Automatic figsize determination
    if not figsize:
        scale = 0.4
        figsize = ((len(df.columns) +1) * barWHRatio * scale, (len(df.index) + 1) * scale)
    # Figure creation
    fig, axes = plt.subplots(1, len(df.columns), sharey=True, figsize=figsize)

    df = df.loc[::-1]
    dfnorm = dfnorm.loc[::-1]

    for i, name in enumerate(df.columns):
        ax = axes[i]
        ax.barh(dfnorm.index, dfnorm[name], color='xkcd:powder blue')
        for pos in ['top', 'right', 'bottom', 'left']: 
            ax.spines[pos].set_visible(False)
        ax.margins(y=0)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_title(name, rotation=90, pad=10)
        ax.set_xlim(0,1)
        ax.get_xaxis().set_visible(False)

        rects = axes[i].patches

        # Annotate numbers
        labels = [f"{x:.3f}".replace('nan', '-') for x in df[name]]
        for rect, label in zip(rects, labels):
            axes[i].text(
                0.5, rect.get_y() + rect.get_height() / 2, label, ha="center", va="center"
            )

    plt.show()

def plotMatrix(df, vmin=-1.0, vmax=1.0, figsize=None):
    """ Plots a pandas DataFrame with a heatmap background.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to plot.
    vmin: float
        Minimum value for the colormap
    vmax: float
        Maximum value for the colormap
    figsize: (float, float) or None
        Figure size. If None, autocalculates considering both the number of
        rows and columns of the dataset
    """
    
    # Create the figure with the proper ratio
    if not figsize:
        size = 1.0
        figsize = (size * df.shape[1], size * df.shape[0])
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Base matrix color
    im = ax.matshow(df, 
                    cmap='RdBu', 
                    vmin=vmin, 
                    vmax=vmax)
    fig.colorbar(im, shrink=0.5)

    # Add labels
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=90,
             ha="left", va="center",rotation_mode="anchor")

    # Add anotations (numbers)
    for (i,j), z in np.ndenumerate(df):
        color = 'black' if (abs(z) < 0.5) else 'white'
        ax.text(j, i, f'{z:.3f}', ha="center", va="center", color=color)

    # Aesthetics
    for pos in ['top', 'right', 'bottom', 'left']: 
        ax.spines[pos].set_visible(False)
    ax.tick_params(axis=u'both', which=u'both', bottom=False)

    plt.show()