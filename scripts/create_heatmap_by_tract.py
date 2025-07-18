import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import requests
import zipfile
import os


data = pd.read_csv("./average_scores_by_tract.csv")
gdf = gpd.read_file('./GeospatialData/cb_2020_39_tract_500k.shp')

df = data
df['census_tract'] = df['census_tract'].astype(str)

# Add walkability categories based on your previous scoring system
def categorize_score(score):
    if score < 1:
        return "very_low"
    elif score < 2.5:
        return "low"
    elif score < 3.5:
        return "medium"
    elif score < 5:
        return "high"
    else:
        return "very_high"

df['walkability_category'] = df['avg_model_score_normalized'].apply(categorize_score)

def create_heatmap():
    # Use your loaded shapefile
    print("Using loaded census tract boundaries...")
    
    # Create GEOID in the same format as your data
    gdf['GEOID'] = gdf['STATEFP'] + gdf['COUNTYFP'] + gdf['TRACTCE']
    
    # Merge with your walkability data
    merged = gdf.merge(df, left_on='GEOID', right_on='census_tract', how='left')
    
    # Filter to only include tracts with data
    merged_with_data = merged[merged['avg_model_score_normalized'].notna()]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Plot the heatmap
    merged_with_data.plot(
        column='avg_model_score_normalized',
        cmap='RdYlGn',  # Red-Yellow-Green colormap (red=low walkability, green=high)
        legend=True,
        ax=ax,
        edgecolor='white',
        linewidth=0.5,
        legend_kwds={'label': 'Walkability Score', 'orientation': 'horizontal', 'shrink': 0.8}
    )
    
    # Customize the plot
    ax.set_title('Walkability Score by Census Tract\n(Cuyahoga County, Ohio)', 
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
    plt.savefig('walkability_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved as walkability_heatmap.png")
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total census tracts: {len(merged_with_data)}")
    print(f"Average walkability score: {merged_with_data['avg_model_score_normalized'].mean():.2f}")
    print(f"Score range: {merged_with_data['avg_model_score_normalized'].min():.2f} - {merged_with_data['avg_model_score_normalized'].max():.2f}")
    
    print("\nWalkability categories:")
    print(merged_with_data['walkability_category'].value_counts().sort_index())
    
    return merged_with_data

create_heatmap()
