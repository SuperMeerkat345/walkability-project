import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import requests
import zipfile
import os


data = pd.read_csv("./csvs/walkability_tracts.csv") # has data for all tracts in the U.S.

gdf = gpd.read_file('./GeospatialData/cb_2020_39_tract_500k.shp')
gdf = gdf[gdf['COUNTYFP'] == '035'] # filter only for cleveland

scores_column = "NatWalkInd"
tract_column = "TRACT"

output = "./plots/walkability_heatmap_comparison.png"
title = "Walkability Score by Census Tract\n(Cleveland)"

df = data
df[tract_column] = df[tract_column].astype(str)

# Add walkability categories based on your previous scoring system
def categorize_score(score):
    if score < 0.5:
        return "very_low"
    elif score < 2:
        return "low"
    elif score < 3:
        return "medium"
    elif score < 4:
        return "high"
    else:
        return "very_high"

df['walkability_category'] = df[scores_column].apply(categorize_score)

def create_heatmap():
    # Use your loaded shapefile
    print("Using loaded census tract boundaries...")
    
    # Create GEOID in the same format as your data
    gdf['GEOID'] = gdf['STATEFP'] + gdf['COUNTYFP'] + gdf['TRACTCE']
    
    # Merge with your walkability data
    merged = gdf.merge(df, left_on='GEOID', right_on=tract_column, how='left')
    
    # Filter to only include tracts with data
    merged_with_data = merged[merged[scores_column].notna()]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Plot the heatmap
    merged_with_data.plot(
        column=scores_column,
        cmap='RdYlGn',  # Red-Yellow-Green colormap (red=low walkability, green=high)
        legend=True,
        ax=ax,
        edgecolor='white',
        linewidth=0.5,
        legend_kwds={'label': 'Walkability Score', 'orientation': 'horizontal', 'shrink': 0.8}
    )
    
    # Customize the plot
    ax.set_title(title, 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_axis_off()
    
    # Add text showing score ranges
    textstr = '\n'.join([
        'Score Categories:',
        'Very Low: < 1.0',
        'Low: 1.0 - 2.5',
        'Medium: 2.5 - 3.5',
        'High: 3.5 - 5.0',
        'Very High: > 5.0'
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as {output}")
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total census tracts: {len(merged_with_data)}")
    print(f"Average walkability score: {merged_with_data[scores_column].mean():.2f}")
    print(f"Score range: {merged_with_data[scores_column].min():.2f} - {merged_with_data[scores_column].max():.2f}")
    
    print("\nWalkability categories:")
    print(merged_with_data['walkability_category'].value_counts().sort_index())
    
    return merged_with_data

create_heatmap()
