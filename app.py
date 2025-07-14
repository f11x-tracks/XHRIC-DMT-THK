import os
import glob
import pandas as pd
from datetime import datetime
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

import xml.etree.ElementTree as ET

import plotly.express as px
import plotly.graph_objects as go

# Directories to search - using current directory
dirs = ['.']  # Current directory
patterns = ['*.xml']  # All XML files in the directory

# Get all files
files = []
for d in dirs:
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(d, pattern)))

print(f"Found {len(files)} XML files to process")

# Collect data
records = []
processed_files = []  # Track successfully processed files

for file in files:
    try:
        # Get file datetime (last modified)
        file_time = datetime.fromtimestamp(os.path.getmtime(file))
        
        # Extract lot number from filename (between 3rd and 4th dash)
        filename = os.path.basename(file)
        filename_parts = filename.split('-')
        lot_number = filename_parts[3] if len(filename_parts) > 3 else 'Unknown'
        
        # Determine DMT type from file path
        if 'DMT102' in file:
            dmt = 'DMT102'
        elif 'DMT103' in file:
            dmt = 'DMT103'
        else:
            dmt = 'Unknown'
            
        # Parse XML
        tree = ET.parse(file)
        root = tree.getroot()
        # Looking for DataRecord elements with Label and Datum children
        for data_record in root.findall('.//DataRecord'):
            label = data_record.findtext('Label')
            datum = data_record.findtext('Datum')
            wafer_id = data_record.findtext('WaferID')
            x_wafer_loc = data_record.findtext('XWaferLoc')
            y_wafer_loc = data_record.findtext('YWaferLoc')
            
            if label in ['Layer 1 Thickness', 'Goodness-of-Fit']:
                try:
                    datum_val = float(datum)
                    
                    # Round Layer 1 Thickness to 1 decimal place
                    if label == 'Layer 1 Thickness':
                        datum_val = round(datum_val, 1)
                    
                    # Create a unique location identifier for pairing measurements
                    location_id = f"{x_wafer_loc}_{y_wafer_loc}" if x_wafer_loc and y_wafer_loc else None
                    
                    # Calculate RADIUS
                    radius = None
                    if x_wafer_loc and y_wafer_loc:
                        try:
                            x_val = float(x_wafer_loc)
                            y_val = float(y_wafer_loc)
                            radius = np.sqrt(x_val**2 + y_val**2)
                        except (ValueError, TypeError):
                            radius = None
                    
                    records.append({
                        'datetime': file_time,
                        'Label': label,
                        'Datum': datum_val,
                        'dmt': dmt,
                        'LotNumber': lot_number,
                        'WaferID': wafer_id,
                        'XWaferLoc': x_wafer_loc,
                        'YWaferLoc': y_wafer_loc,
                        'location_id': location_id,
                        'RADIUS': radius
                    })
                except (TypeError, ValueError):
                    continue
        
        # Add to processed files list if we got here without errors
        processed_files.append({
            'filename': os.path.basename(file),
            'full_path': file,
            'dmt_type': dmt,
            'lot_number': lot_number,
            'file_datetime': file_time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        continue

# Read condition data
condition_file = r'acondition.txt'
try:
    conditions_df = pd.read_csv(condition_file, sep=' ', skipinitialspace=True)
    # Clean up column names (remove any extra whitespace)
    conditions_df.columns = conditions_df.columns.str.strip()
    conditions_df['WaferID'] = conditions_df['WaferID'].str.strip()
    conditions_df['Condition'] = conditions_df['Condition'].str.strip()
    print(f"Loaded {len(conditions_df)} condition records from {condition_file}")
except Exception as e:
    print(f"Could not read condition file {condition_file}: {e}")
    conditions_df = pd.DataFrame(columns=['WaferID', 'Condition'])

df = pd.DataFrame(records)

# Merge condition data with main dataframe
df = df.merge(conditions_df[['WaferID', 'Condition']], on='WaferID', how='left')
# Fill missing conditions with 'NA'
df['Condition'] = df['Condition'].fillna('NA')

print(f"Added condition data. Conditions found: {df['Condition'].value_counts().to_dict()}")

# Dash app
app = dash.Dash(__name__)

def make_boxplot(label, y_range=None, filtered_df=None):
    working_df = filtered_df if filtered_df is not None else df
    dff = working_df[working_df['Label'] == label]
    if dff.empty:
        return html.Div(f"No data for {label}")
    fig = px.box(
        dff,
        x=dff['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        y='Datum',
        color='dmt',
        points='all',
        title=f'Boxplot of {label} over Time',
        labels={'Datum': label, 'datetime': 'File DateTime', 'dmt': 'DMT Type'}
    )
    fig.update_layout(xaxis_title='File DateTime', yaxis_title=label)
    if y_range and label == 'Layer 1 Thickness':
        fig.update_layout(yaxis=dict(range=y_range))
    return dcc.Graph(figure=fig)

def make_wafer_plots(label):
    dff = df[df['Label'] == label]
    if dff.empty:
        return html.Div(f"No data for {label}")
    
    unique_wafers = sorted(dff['WaferID'].unique())
    plots = []
    
    for wafer_id in unique_wafers:
        wafer_data = dff[dff['WaferID'] == wafer_id]
        if not wafer_data.empty:
            fig = px.box(
                wafer_data,
                x=wafer_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                y='Datum',
                color='dmt',
                points='all',
                title=f'{label} - WaferID: {wafer_id}',
                labels={'Datum': label, 'datetime': 'File DateTime', 'dmt': 'DMT Type'}
            )
            fig.update_layout(
                xaxis_title='File DateTime', 
                yaxis_title=label,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            plots.append(dcc.Graph(figure=fig))
    
    return html.Div(plots)

def make_scatter_plot():
    # Filter data that has location information for proper pairing
    df_with_location = df[df['location_id'].notna()].copy()
    
    if df_with_location.empty:
        return html.Div("No location data available for scatter plot")
    
    # Separate GoF and Thickness data
    gof_data = df_with_location[df_with_location['Label'] == 'Goodness-of-Fit'][['datetime', 'WaferID', 'dmt', 'LotNumber', 'location_id', 'Datum']].rename(columns={'Datum': 'GoodnessOfFit'})
    thickness_data = df_with_location[df_with_location['Label'] == 'Layer 1 Thickness'][['datetime', 'WaferID', 'dmt', 'LotNumber', 'location_id', 'Datum']].rename(columns={'Datum': 'Layer1Thickness'})
    
    # Merge based on location (same measurement point)
    merged_data = pd.merge(gof_data, thickness_data, on=['datetime', 'WaferID', 'dmt', 'LotNumber', 'location_id'], how='inner')
    
    if merged_data.empty:
        return html.Div("No paired measurement data available for scatter plot")
    
    fig = px.scatter(
        merged_data,
        x='GoodnessOfFit',
        y='Layer1Thickness',
        color='dmt',
        symbol='WaferID',
        title='Layer 1 Thickness vs Goodness-of-Fit (Same Measurement Points)',
        labels={
            'GoodnessOfFit': 'Goodness-of-Fit',
            'Layer1Thickness': 'Layer 1 Thickness',
            'dmt': 'DMT Type'
        },
        hover_data=['WaferID', 'LotNumber', 'datetime']
    )
    
    fig.update_layout(
        xaxis_title='Goodness-of-Fit',
        yaxis_title='Layer 1 Thickness',
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return dcc.Graph(figure=fig)

def make_radius_thickness_plots(y_range=None, show_trend_legend=True, filtered_df=None):
    working_df = filtered_df if filtered_df is not None else df
    # Filter for Layer 1 Thickness data with valid RADIUS
    thickness_data = working_df[(working_df['Label'] == 'Layer 1 Thickness') & (working_df['RADIUS'].notna())].copy()
    
    if thickness_data.empty:
        return html.Div("No Layer 1 Thickness data with RADIUS available")
    
    # Create single scatter plot
    fig = go.Figure()
    
    # Get unique wafer IDs and assign colors/symbols
    unique_wafers = sorted(thickness_data['WaferID'].unique())
    
    # Define colors for each wafer (expanded to 12 colors)
    plotly_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78'
    ]
    
    # Add scatter points colored by WaferID
    for i, wafer_id in enumerate(unique_wafers):
        wafer_data = thickness_data[thickness_data['WaferID'] == wafer_id].copy()
        
        if wafer_data.empty:
            continue
        
        # Assign specific color for this wafer
        wafer_color = plotly_colors[i % len(plotly_colors)]
            
        # Add scatter points
        scatter_trace = go.Scatter(
            x=wafer_data['RADIUS'],
            y=wafer_data['Datum'],
            mode='markers',
            name=f'WaferID: {wafer_id}',
            marker=dict(size=8, color=wafer_color),
            text=wafer_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            customdata=wafer_data[['dmt', 'LotNumber']],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'RADIUS: %{x:.2f}<br>' +
                          'Thickness: %{y:.2f}<br>' +
                          'DMT Type: %{customdata[0]}<br>' +
                          'Lot Number: %{customdata[1]}<br>' +
                          'DateTime: %{text}<br>' +
                          '<extra></extra>'
        )
        fig.add_trace(scatter_trace)
        
        # Get the color that Plotly assigned to this trace
        # Plotly uses a default color sequence, we can predict it
        plotly_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        color = plotly_colors[i % len(plotly_colors)]
        
        # Add LOWESS trend line for each wafer if there are enough points
        if len(wafer_data) >= 5:  # Need at least 5 points for LOWESS
            try:
                # Sort by RADIUS for proper trend line
                wafer_sorted = wafer_data.sort_values('RADIUS')
                
                # Apply LOWESS smoothing
                # frac parameter controls smoothness (0.1-1.0, smaller = less smooth)
                lowess_result = lowess(wafer_sorted['Datum'], wafer_sorted['RADIUS'], 
                                     frac=0.3, it=3, return_sorted=True)
                
                # Extract smoothed values
                x_smooth = lowess_result[:, 0]
                y_smooth = lowess_result[:, 1]
                
                # Add LOWESS trend line with matching color
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name=f'LOWESS: {wafer_id}',
                    line=dict(dash='dash', width=2, color=wafer_color),
                    showlegend=show_trend_legend,   # Controlled by parameter
                    hoverinfo='skip'   # Don't show hover for trend line
                ))
                
            except Exception as e:
                # Fallback to simple linear regression if LOWESS fails
                try:
                    coeffs = np.polyfit(wafer_sorted['RADIUS'], wafer_sorted['Datum'], 1)
                    x_fit = np.linspace(wafer_sorted['RADIUS'].min(), wafer_sorted['RADIUS'].max(), 50)
                    y_fit = np.polyval(coeffs, x_fit);
                    
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode='lines',
                        name=f'Linear: {wafer_id}',
                        line=dict(dash='dot', width=2, color=wafer_color),
                        showlegend=show_trend_legend,
                        hoverinfo='skip'
                    ))
                except:
                    pass  # Skip trend line if all methods fail
    
    # Update layout
    fig.update_layout(
        title='Layer 1 Thickness vs RADIUS by WaferID (with LOWESS Trend Lines)',
        xaxis_title='RADIUS',
        yaxis_title='Layer 1 Thickness',
        xaxis=dict(range=[0, 150], dtick=15),
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True
    )
    
    # Apply y-axis range if provided
    if y_range:
        fig.update_layout(yaxis=dict(range=y_range))
    
    return dcc.Graph(figure=fig)

def make_radius_thickness_by_condition_plots(y_range=None, show_trend_legend=True, filtered_df=None):
    working_df = filtered_df if filtered_df is not None else df
    # Filter for Layer 1 Thickness data with valid RADIUS and non-NA conditions
    thickness_data = working_df[(working_df['Label'] == 'Layer 1 Thickness') & 
                       (working_df['RADIUS'].notna()) & 
                       (working_df['Condition'] != 'NA')].copy()
    
    if thickness_data.empty:
        return html.Div("No Layer 1 Thickness data with RADIUS and conditions available")
    
    # Create single scatter plot
    fig = go.Figure()
    
    # Get unique conditions
    unique_conditions = sorted(thickness_data['Condition'].unique())
    
    # Define colors for each condition (expanded to 12 colors)
    plotly_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78'
    ]
    
    # Add scatter points colored by Condition
    for i, condition in enumerate(unique_conditions):
        condition_data = thickness_data[thickness_data['Condition'] == condition].copy()
        
        if condition_data.empty:
            continue
        
        # Assign specific color for this condition
        condition_color = plotly_colors[i % len(plotly_colors)]
            
        # Add scatter points
        scatter_trace = go.Scatter(
            x=condition_data['RADIUS'],
            y=condition_data['Datum'],
            mode='markers',
            name=f'Condition: {condition}',
            marker=dict(size=8, color=condition_color),
            text=condition_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            customdata=condition_data[['dmt', 'WaferID', 'LotNumber']],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'RADIUS: %{x:.2f}<br>' +
                          'Thickness: %{y:.2f}<br>' +
                          'WaferID: %{customdata[1]}<br>' +
                          'DMT Type: %{customdata[0]}<br>' +
                          'Lot Number: %{customdata[2]}<br>' +
                          'DateTime: %{text}<br>' +
                          '<extra></extra>'
        )
        fig.add_trace(scatter_trace)
        
        # Get the actual color that Plotly assigned to the scatter trace
        # We'll extract it after the trace is added to the figure
        actual_color = None
        if len(fig.data) > 0:
            # Get the color from the most recently added trace
            last_trace = fig.data[-1]
            if hasattr(last_trace, 'marker') and hasattr(last_trace.marker, 'color'):
                actual_color = last_trace.marker.color
        
        # Fallback to default plotly colors if we can't get the actual color
        if actual_color is None:
            plotly_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            actual_color = plotly_colors[i % len(plotly_colors)]
        
        # Add LOWESS trend line for each condition if there are enough points
        if len(condition_data) >= 5:  # Need at least 5 points for LOWESS
            try:
                # Sort by RADIUS for proper trend line
                condition_sorted = condition_data.sort_values('RADIUS')
                
                # Apply LOWESS smoothing
                lowess_result = lowess(condition_sorted['Datum'], condition_sorted['RADIUS'], 
                                     frac=0.3, it=3, return_sorted=True)
                
                # Extract smoothed values
                x_smooth = lowess_result[:, 0]
                y_smooth = lowess_result[:, 1]
                
                # Add LOWESS trend line with matching color
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name=f'LOWESS: {condition}',
                    line=dict(dash='dash', width=3, color=condition_color),
                    showlegend=show_trend_legend,   # Controlled by parameter
                    hoverinfo='skip'   # Don't show hover for trend line
                ))
                
            except Exception as e:
                # Fallback to simple linear regression if LOWESS fails
                try:
                    coeffs = np.polyfit(condition_sorted['RADIUS'], condition_sorted['Datum'], 1)
                    x_fit = np.linspace(condition_sorted['RADIUS'].min(), condition_sorted['RADIUS'].max(), 50)
                    y_fit = np.polyval(coeffs, x_fit)
                    
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode='lines',
                        name=f'Linear: {condition}',
                        line=dict(dash='dot', width=3, color=condition_color),
                        showlegend=show_trend_legend,
                        hoverinfo='skip'
                    ))
                except:
                    pass  # Skip trend line if all methods fail
    
    # Update layout
    fig.update_layout(
        title='Layer 1 Thickness vs RADIUS by Condition (with LOWESS Trend Lines)',
        xaxis_title='RADIUS',
        yaxis_title='Layer 1 Thickness',
        xaxis=dict(range=[0, 150], dtick=15),
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True
    )
    
    # Apply y-axis range if provided
    if y_range:
        fig.update_layout(yaxis=dict(range=y_range))
    
    return dcc.Graph(figure=fig)

def make_files_table():
    """Create a table showing all processed XML files"""
    if not processed_files:
        return html.Div("No files were processed")
    
    # Create a DataFrame for the table
    files_df = pd.DataFrame(processed_files)
    
    # Create the table using dash_table
    table = dash_table.DataTable(
        data=files_df.to_dict('records'),
        columns=[
            {"name": "File Name", "id": "filename"},
            {"name": "DMT Type", "id": "dmt_type"},
            {"name": "Lot Number", "id": "lot_number"},
            {"name": "File Date/Time", "id": "file_datetime"},
            {"name": "Full Path", "id": "full_path"}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{dmt_type} = DMT102'},
                'backgroundColor': 'rgba(255, 182, 193, 0.3)',
            },
            {
                'if': {'filter_query': '{dmt_type} = DMT103'},
                'backgroundColor': 'rgba(173, 216, 230, 0.3)',
            }
        ],
        page_size=20,
        sort_action="native",
        filter_action="native"
    )
    
    return html.Div([
        html.H3(f"Processed XML Files ({len(processed_files)} total)"),
        table
    ])

def make_statistical_summary_table(filtered_df=None):
    """Create a statistical summary table showing mean and std dev for each WaferID with radial groups"""
    working_df = filtered_df if filtered_df is not None else df
    if working_df.empty:
        return html.Div("No data available for statistical summary")
    
    # Get unique labels and wafer IDs
    unique_labels = sorted(working_df['Label'].unique())
    unique_wafers = sorted(working_df['WaferID'].unique())
    
    summary_data = []
    
    for wafer_id in unique_wafers:
        wafer_data = working_df[working_df['WaferID'] == wafer_id]
        
        # Get condition for this wafer
        condition = wafer_data['Condition'].iloc[0] if not wafer_data.empty else 'NA'
        
        # Only process Layer 1 Thickness data
        thickness_data = wafer_data[wafer_data['Label'] == 'Layer 1 Thickness']
        label_data = thickness_data['Datum']
        
        if not label_data.empty:
            mean_val = label_data.mean()
            std_val = label_data.std()
            count_val = len(label_data)
            
            # Calculate standard deviations for radial groups
            # Center group: RADIUS 0-89
            center_data = thickness_data[(thickness_data['RADIUS'] >= 0) & (thickness_data['RADIUS'] < 89)]['Datum']
            center_std = round(center_data.std(), 4) if len(center_data) > 1 else None
            
            # Mid group: RADIUS 89-120
            mid_data = thickness_data[(thickness_data['RADIUS'] >= 89) & (thickness_data['RADIUS'] < 120)]['Datum']
            mid_std = round(mid_data.std(), 4) if len(mid_data) > 1 else None
            
            # Edge group: RADIUS 120-150
            edge_data = thickness_data[(thickness_data['RADIUS'] >= 120) & (thickness_data['RADIUS'] <= 150)]['Datum']
            edge_std = round(edge_data.std(), 4) if len(edge_data) > 1 else None
            
            summary_data.append({
                'WaferID': wafer_id,
                'Condition': condition,
                'Measurement': 'Layer 1 Thickness',
                'Mean': round(mean_val, 1),
                'Std Dev': round(std_val, 1),
                'Count': count_val,
                'Center Std (0-89)': round(center_std, 1) if center_std is not None else None,
                'Mid Std (89-120)': round(mid_std, 1) if mid_std is not None else None,
                'Edge Std (120-150)': round(edge_std, 1) if edge_std is not None else None
            })
    
    if not summary_data:
        return html.Div("No statistical data to display")
    
    # Sort summary data by Condition
    summary_data = sorted(summary_data, key=lambda x: x['Condition'])
    
    # Calculate values for color scaling
    center_std_values = [item['Center Std (0-89)'] for item in summary_data if item['Center Std (0-89)'] is not None]
    std_dev_values = [item['Std Dev'] for item in summary_data if item['Std Dev'] is not None]
    mid_std_values = [item['Mid Std (89-120)'] for item in summary_data if item['Mid Std (89-120)'] is not None]
    edge_std_values = [item['Edge Std (120-150)'] for item in summary_data if item['Edge Std (120-150)'] is not None]
    
    # Create color scale conditions for all std columns
    all_std_conditional = []
    
    # Helper function to create color scale for a column
    def create_color_scale(values, column_id, column_name):
        if not values:
            return []
        
        min_val = min(values)
        max_val = max(values)
        conditional = []
        
        # Create 10 color gradients from green to red
        for i in range(10):
            threshold = min_val + (max_val - min_val) * (i / 9)
            next_threshold = min_val + (max_val - min_val) * ((i + 1) / 9)
            
            # Color interpolation from green (0,255,0) to red (255,0,0)
            red = int(255 * (i / 9))
            green = int(255 * (1 - i / 9))
            color = f'rgba({red}, {green}, 0, 0.6)'
            
            if i == 9:  # Last condition should capture maximum
                conditional.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} >= {threshold:.4f}',
                        'column_id': column_id
                    },
                    'backgroundColor': color,
                })
            else:
                conditional.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} >= {threshold:.4f} && {{{column_name}}} < {next_threshold:.4f}',
                        'column_id': column_id
                    },
                    'backgroundColor': color,
                })
        return conditional
    
    # Create color scales for each std column
    all_std_conditional.extend(create_color_scale(center_std_values, 'Center Std (0-89)', 'Center Std (0-89)'))
    all_std_conditional.extend(create_color_scale(std_dev_values, 'Std Dev', 'Std Dev'))
    all_std_conditional.extend(create_color_scale(mid_std_values, 'Mid Std (89-120)', 'Mid Std (89-120)'))
    all_std_conditional.extend(create_color_scale(edge_std_values, 'Edge Std (120-150)', 'Edge Std (120-150)'))
    
    # Create the summary table using dash_table
    summary_table = dash_table.DataTable(
        data=summary_data,
        columns=[
            {"name": "Wafer ID", "id": "WaferID"},
            {"name": "Condition", "id": "Condition"},
            {"name": "Measurement Type", "id": "Measurement"},
            {"name": "Mean", "id": "Mean", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Std Dev", "id": "Std Dev", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Count", "id": "Count", "type": "numeric"},
            {"name": "Center Std (0-89)", "id": "Center Std (0-89)", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Mid Std (89-120)", "id": "Mid Std (89-120)", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Edge Std (120-150)", "id": "Edge Std (120-150)", "type": "numeric", "format": {"specifier": ".1f"}}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '8px',
            'fontFamily': 'Arial',
            'fontSize': '12px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgba(248, 248, 248, 0.8)',
            },
            {
                'if': {'filter_query': '{Condition} = "no-relax"'},
                'backgroundColor': 'rgba(255, 182, 193, 0.3)',
            },
            {
                'if': {'filter_query': '{Condition} contains "rpm"'},
                'backgroundColor': 'rgba(173, 216, 230, 0.3)',
            },
            {
                'if': {'filter_query': '{Condition} = "NA"'},
                'backgroundColor': 'rgba(220, 220, 220, 0.5)',
            }
        ] + all_std_conditional,  # Add the color scale conditions for all std columns
        sort_action="native",
        filter_action="native",
        page_size=30
    )
    
    return html.Div([
        html.H3(f"Statistical Summary by WaferID ({len(unique_wafers)} wafers)"),
        html.P("Mean, standard deviation, and radial group statistics for Layer 1 Thickness by wafer with condition"),
        html.P("Radial Groups: Center (0-89), Mid (89-120), Edge (120-150)", style={'fontStyle': 'italic', 'fontSize': '12px'}),
        summary_table
    ])

def make_statistical_summary_by_condition_table(filtered_df=None):
    """Create a statistical summary table showing mean and std dev grouped by Condition with radial groups"""
    working_df = filtered_df if filtered_df is not None else df
    if working_df.empty:
        return html.Div("No data available for statistical summary")
    
    # Get unique conditions
    unique_conditions = sorted(working_df['Condition'].unique())
    
    summary_data = []
    
    for condition in unique_conditions:
        condition_data = working_df[working_df['Condition'] == condition]
        
        # Only process Layer 1 Thickness data
        thickness_data = condition_data[condition_data['Label'] == 'Layer 1 Thickness']
        label_data = thickness_data['Datum']
        
        if not label_data.empty:
            mean_val = label_data.mean()
            std_val = label_data.std()
            count_val = len(label_data)
            
            # Calculate standard deviations for radial groups
            # Center group: RADIUS 0-89
            center_data = thickness_data[(thickness_data['RADIUS'] >= 0) & (thickness_data['RADIUS'] < 89)]['Datum']
            center_std = center_data.std() if len(center_data) > 1 else None
            
            # Mid group: RADIUS 89-120
            mid_data = thickness_data[(thickness_data['RADIUS'] >= 89) & (thickness_data['RADIUS'] < 120)]['Datum']
            mid_std = mid_data.std() if len(mid_data) > 1 else None
            
            # Edge group: RADIUS 120-150
            edge_data = thickness_data[(thickness_data['RADIUS'] >= 120) & (thickness_data['RADIUS'] <= 150)]['Datum']
            edge_std = edge_data.std() if len(edge_data) > 1 else None
            
            summary_data.append({
                'Condition': condition,
                'Measurement': 'Layer 1 Thickness',
                'Mean': round(mean_val, 1),
                'Std Dev': round(std_val, 1),
                'Count': count_val,
                'Center Std (0-89)': round(center_std, 1) if center_std is not None else None,
                'Mid Std (89-120)': round(mid_std, 1) if mid_std is not None else None,
                'Edge Std (120-150)': round(edge_std, 1) if edge_std is not None else None
            })
    
    if not summary_data:
        return html.Div("No statistical data to display")
    
    # Sort summary data by Condition
    summary_data = sorted(summary_data, key=lambda x: x['Condition'])
    
    # Calculate values for color scaling
    center_std_values = [item['Center Std (0-89)'] for item in summary_data if item['Center Std (0-89)'] is not None]
    std_dev_values = [item['Std Dev'] for item in summary_data if item['Std Dev'] is not None]
    mid_std_values = [item['Mid Std (89-120)'] for item in summary_data if item['Mid Std (89-120)'] is not None]
    edge_std_values = [item['Edge Std (120-150)'] for item in summary_data if item['Edge Std (120-150)'] is not None]
    
    # Create color scale conditions for all std columns
    all_std_conditional = []
    
    # Helper function to create color scale for a column
    def create_color_scale(values, column_id, column_name):
        if not values:
            return []
        
        min_val = min(values)
        max_val = max(values)
        conditional = []
        
        # Create 10 color gradients from green to red
        for i in range(10):
            threshold = min_val + (max_val - min_val) * (i / 9)
            next_threshold = min_val + (max_val - min_val) * ((i + 1) / 9)
            
            # Color interpolation from green (0,255,0) to red (255,0,0)
            red = int(255 * (i / 9))
            green = int(255 * (1 - i / 9))
            color = f'rgba({red}, {green}, 0, 0.6)'
            
            if i == 9:  # Last condition should capture maximum
                conditional.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} >= {threshold:.4f}',
                        'column_id': column_id
                    },
                    'backgroundColor': color,
                })
            else:
                conditional.append({
                    'if': {
                        'filter_query': f'{{{column_name}}} >= {threshold:.4f} && {{{column_name}}} < {next_threshold:.4f}',
                        'column_id': column_id
                    },
                    'backgroundColor': color,
                })
        return conditional
    
    # Create color scales for each std column
    all_std_conditional.extend(create_color_scale(center_std_values, 'Center Std (0-89)', 'Center Std (0-89)'))
    all_std_conditional.extend(create_color_scale(std_dev_values, 'Std Dev', 'Std Dev'))
    all_std_conditional.extend(create_color_scale(mid_std_values, 'Mid Std (89-120)', 'Mid Std (89-120)'))
    all_std_conditional.extend(create_color_scale(edge_std_values, 'Edge Std (120-150)', 'Edge Std (120-150)'))
    
    # Create the summary table using dash_table
    summary_table = dash_table.DataTable(
        data=summary_data,
        columns=[
            {"name": "Condition", "id": "Condition"},
            {"name": "Measurement Type", "id": "Measurement"},
            {"name": "Mean", "id": "Mean", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Std Dev", "id": "Std Dev", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Count", "id": "Count", "type": "numeric"},
            {"name": "Center Std (0-89)", "id": "Center Std (0-89)", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Mid Std (89-120)", "id": "Mid Std (89-120)", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Edge Std (120-150)", "id": "Edge Std (120-150)", "type": "numeric", "format": {"specifier": ".1f"}}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '8px',
            'fontFamily': 'Arial',
            'fontSize': '12px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgba(248, 248, 248, 0.8)',
            },
            {
                'if': {'filter_query': '{Condition} = "no-relax"'},
                'backgroundColor': 'rgba(255, 182, 193, 0.3)',
            },
            {
                'if': {'filter_query': '{Condition} contains "rpm"'},
                'backgroundColor': 'rgba(173, 216, 230, 0.3)',
            },
            {
                'if': {'filter_query': '{Condition} = "NA"'},
                'backgroundColor': 'rgba(220, 220, 220, 0.5)',
            }
        ] + all_std_conditional,  # Add the color scale conditions for all std columns
        sort_action="native",
        filter_action="native",
        page_size=30
    )
    
    return html.Div([
        html.H3(f"Statistical Summary by Condition ({len(unique_conditions)} conditions)"),
        html.P("Aggregated mean, standard deviation, and radial group statistics for Layer 1 Thickness by process condition"),
        html.P("Radial Groups: Center (0-89), Mid (89-120), Edge (120-150)", style={'fontStyle': 'italic', 'fontSize': '12px'}),
        summary_table
    ])

def export_summary_tables_to_excel():
    """Export both summary tables to an Excel file with multiple sheets"""
    try:
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'DMT_Summary_Tables_{timestamp}.xlsx'
        
        # Create summary data for WaferID table
        wafer_summary_data = []
        if not df.empty:
            unique_wafers = sorted(df['WaferID'].unique())
            
            for wafer_id in unique_wafers:
                wafer_data = df[df['WaferID'] == wafer_id]
                condition = wafer_data['Condition'].iloc[0] if not wafer_data.empty else 'NA'
                
                thickness_data = wafer_data[wafer_data['Label'] == 'Layer 1 Thickness']
                label_data = thickness_data['Datum']
                
                if not label_data.empty:
                    mean_val = label_data.mean()
                    std_val = label_data.std()
                    count_val = len(label_data)
                    
                    # Calculate radial group standard deviations
                    center_data = thickness_data[(thickness_data['RADIUS'] >= 0) & (thickness_data['RADIUS'] < 89)]['Datum']
                    center_std = center_data.std() if len(center_data) > 1 else None
                    
                    mid_data = thickness_data[(thickness_data['RADIUS'] >= 89) & (thickness_data['RADIUS'] < 120)]['Datum']
                    mid_std = mid_data.std() if len(mid_data) > 1 else None
                    
                    edge_data = thickness_data[(thickness_data['RADIUS'] >= 120) & (thickness_data['RADIUS'] <= 150)]['Datum']
                    edge_std = edge_data.std() if len(edge_data) > 1 else None
                    
                    wafer_summary_data.append({
                        'WaferID': wafer_id,
                        'Condition': condition,
                        'Measurement': 'Layer 1 Thickness',
                        'Mean': round(mean_val, 1),
                        'Std Dev': round(std_val, 1),
                        'Count': count_val,
                        'Center Std (0-89)': round(center_std, 1) if center_std is not None else None,
                        'Mid Std (89-120)': round(mid_std, 1) if mid_std is not None else None,
                        'Edge Std (120-150)': round(edge_std, 1) if edge_std is not None else None
                    })
        
        # Create summary data for Condition table
        condition_summary_data = []
        if not df.empty:
            unique_conditions = sorted(df['Condition'].unique())
            
            for condition in unique_conditions:
                condition_data = df[df['Condition'] == condition]
                
                thickness_data = condition_data[condition_data['Label'] == 'Layer 1 Thickness']
                label_data = thickness_data['Datum']
                
                if not label_data.empty:
                    mean_val = label_data.mean()
                    std_val = label_data.std()
                    count_val = len(label_data)
                    
                    # Calculate radial group standard deviations
                    center_data = thickness_data[(thickness_data['RADIUS'] >= 0) & (thickness_data['RADIUS'] < 89)]['Datum']
                    center_std = center_data.std() if len(center_data) > 1 else None
                    
                    mid_data = thickness_data[(thickness_data['RADIUS'] >= 89) & (thickness_data['RADIUS'] < 120)]['Datum']
                    mid_std = mid_data.std() if len(mid_data) > 1 else None
                    
                    edge_data = thickness_data[(thickness_data['RADIUS'] >= 120) & (thickness_data['RADIUS'] <= 150)]['Datum']
                    edge_std = edge_data.std() if len(edge_data) > 1 else None
                    
                    condition_summary_data.append({
                        'Condition': condition,
                        'Measurement': 'Layer 1 Thickness',
                        'Mean': round(mean_val, 1),
                        'Std Dev': round(std_val, 1),
                        'Count': count_val,
                        'Center Std (0-89)': round(center_std, 1) if center_std is not None else None,
                        'Mid Std (89-120)': round(mid_std, 1) if mid_std is not None else None,
                        'Edge Std (120-150)': round(edge_std, 1) if edge_std is not None else None
                    })
        
        # Create DataFrames
        wafer_df = pd.DataFrame(wafer_summary_data)
        condition_df = pd.DataFrame(condition_summary_data)
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            if not wafer_df.empty:
                wafer_df.to_excel(writer, sheet_name='Summary by WaferID', index=False)
            if not condition_df.empty:
                condition_df.to_excel(writer, sheet_name='Summary by Condition', index=False)
            
            # Add metadata sheet
            metadata = pd.DataFrame({
                'Export Information': [
                    'Export Date/Time',
                    'Total XML Files Processed',
                    'Total Wafers',
                    'Total Conditions',
                    'Radial Groups Definition'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(processed_files),
                    len(df['WaferID'].unique()) if not df.empty else 0,
                    len(df['Condition'].unique()) if not df.empty else 0,
                    'Center (0-89), Mid (89-120), Edge (120-150)'
                ]
            })
            metadata.to_excel(writer, sheet_name='Export Info', index=False)
        
        return filename, True
        
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return None, False

def export_full_data_to_excel():
    """Export the full dataframe with all data including RADIUS and Conditions to Excel"""
    try:
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'DMT_Full_Data_{timestamp}.xlsx'
        
        if df.empty:
            print("No data to export")
            return None, False
        
        # Create a copy of the dataframe for export
        export_df = df.copy()
        
        # Format datetime column for better readability
        export_df['datetime'] = export_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Round RADIUS to 2 decimal places for cleaner display
        export_df['RADIUS'] = export_df['RADIUS'].round(2)
        
        # Reorder columns for better presentation
        column_order = [
            'datetime', 'LotNumber', 'WaferID', 'Condition', 'dmt', 'Label', 'Datum', 
            'XWaferLoc', 'YWaferLoc', 'RADIUS', 'location_id'
        ]
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in column_order if col in export_df.columns]
        export_df = export_df[available_columns]
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Export all data
            export_df.to_excel(writer, sheet_name='All Data', index=False)
            
            # Export Layer 1 Thickness data only
            thickness_data = export_df[export_df['Label'] == 'Layer 1 Thickness'].copy()
            if not thickness_data.empty:
                thickness_data.to_excel(writer, sheet_name='Layer 1 Thickness', index=False)
            
            # Export Goodness-of-Fit data only
            gof_data = export_df[export_df['Label'] == 'Goodness-of-Fit'].copy()
            if not gof_data.empty:
                gof_data.to_excel(writer, sheet_name='Goodness-of-Fit', index=False)
            
            # Add metadata sheet
            metadata = pd.DataFrame({
                'Export Information': [
                    'Export Date/Time',
                    'Total Records',
                    'Layer 1 Thickness Records',
                    'Goodness-of-Fit Records',
                    'Total XML Files Processed',
                    'Total Wafers',
                    'Total Lot Numbers',
                    'Total Conditions',
                    'Records with RADIUS data',
                    'Data Description'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(export_df),
                    len(export_df[export_df['Label'] == 'Layer 1 Thickness']),
                    len(export_df[export_df['Label'] == 'Goodness-of-Fit']),
                    len(processed_files),
                    len(export_df['WaferID'].unique()) if not export_df.empty else 0,
                    len(export_df['LotNumber'].unique()) if not export_df.empty else 0,
                    len(export_df['Condition'].unique()) if not export_df.empty else 0,
                    len(export_df[export_df['RADIUS'].notna()]),
                    'Complete measurement data with RADIUS calculations, lot numbers, and process conditions'
                ]
            })
            metadata.to_excel(writer, sheet_name='Export Info', index=False)
        
        return filename, True
        
    except Exception as e:
        print(f"Error exporting full data to Excel: {e}")
        return None, False

def filter_outliers(data, method='iqr', threshold=1.5):
    """
    Filter outliers from data using various methods
    
    Parameters:
    - data: pandas Series or DataFrame column
    - method: 'iqr', 'zscore', 'modified_zscore', or 'percentile'
    - threshold: threshold value for the method
    
    Returns:
    - mask: boolean mask where True indicates valid (non-outlier) data
    """
    if data.empty or data.isna().all():
        return pd.Series([True] * len(data), index=data.index)
    
    if method == 'iqr':
        # Interquartile Range method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
        
    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores <= threshold
        
    elif method == 'modified_zscore':
        # Modified Z-score using median absolute deviation
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        mask = np.abs(modified_z_scores) <= threshold
        
    elif method == 'percentile':
        # Percentile method
        lower_percentile = (100 - threshold) / 2
        upper_percentile = 100 - lower_percentile
        lower_bound = data.quantile(lower_percentile / 100)
        upper_bound = data.quantile(upper_percentile / 100)
        mask = (data >= lower_bound) & (data <= upper_bound)
        
    else:
        # No filtering
        mask = pd.Series([True] * len(data), index=data.index)
    
    return mask

def get_filtered_dataframe(outlier_method='none', outlier_threshold=1.5):
    """Get filtered dataframe based on outlier detection settings"""
    if outlier_method == 'none' or df.empty:
        return df
    
    # Apply outlier filtering only to Layer 1 Thickness data
    filtered_df = df.copy()
    thickness_mask = filtered_df['Label'] == 'Layer 1 Thickness'
    
    if thickness_mask.any():
        thickness_data = filtered_df.loc[thickness_mask, 'Datum']
        
        # Apply different filtering strategies
        if outlier_method in ['iqr', 'zscore', 'modified_zscore', 'percentile']:
            # Global filtering - remove outliers across all data
            valid_mask = filter_outliers(thickness_data, outlier_method, outlier_threshold)
            outlier_indices = thickness_data[~valid_mask].index
            filtered_df = filtered_df.drop(outlier_indices)
            
        elif outlier_method.endswith('_by_wafer'):
            # Per-wafer filtering
            base_method = outlier_method.replace('_by_wafer', '')
            outlier_indices = []
            
            for wafer_id in filtered_df['WaferID'].unique():
                wafer_thickness_mask = (filtered_df['Label'] == 'Layer 1 Thickness') & (filtered_df['WaferID'] == wafer_id)
                if wafer_thickness_mask.any():
                    wafer_thickness_data = filtered_df.loc[wafer_thickness_mask, 'Datum']
                    valid_mask = filter_outliers(wafer_thickness_data, base_method, outlier_threshold)
                    wafer_outlier_indices = wafer_thickness_data[~valid_mask].index
                    outlier_indices.extend(wafer_outlier_indices)
            
            if outlier_indices:
                filtered_df = filtered_df.drop(outlier_indices)
                
        elif outlier_method.endswith('_by_condition'):
            # Per-condition filtering
            base_method = outlier_method.replace('_by_condition', '')
            outlier_indices = []
            
            for condition in filtered_df['Condition'].unique():
                condition_thickness_mask = (filtered_df['Label'] == 'Layer 1 Thickness') & (filtered_df['Condition'] == condition)
                if condition_thickness_mask.any():
                    condition_thickness_data = filtered_df.loc[condition_thickness_mask, 'Datum']
                    valid_mask = filter_outliers(condition_thickness_data, base_method, outlier_threshold)
                    condition_outlier_indices = condition_thickness_data[~valid_mask].index
                    outlier_indices.extend(condition_outlier_indices)
            
            if outlier_indices:
                filtered_df = filtered_df.drop(outlier_indices)
    
    return filtered_df

# Calculate global Layer 1 Thickness range for scaling
thickness_data = df[df['Label'] == 'Layer 1 Thickness']['Datum']
if not thickness_data.empty:
    thickness_min = thickness_data.min()
    thickness_max = thickness_data.max()
    thickness_range = thickness_max - thickness_min
else:
    thickness_min = thickness_max = thickness_range = 0

# Callback for Layer 1 Thickness boxplot
@app.callback(
    Output('thickness-boxplot', 'children'),
    [Input('yscale-dropdown', 'value'),
     Input('outlier-method-dropdown', 'value'),
     Input('outlier-threshold-input', 'value')]
)
def update_thickness_boxplot(yscale_percent, outlier_method, outlier_threshold):
    filtered_df = get_filtered_dataframe(outlier_method, outlier_threshold)
    
    # Recalculate thickness range based on filtered data
    thickness_data = filtered_df[filtered_df['Label'] == 'Layer 1 Thickness']['Datum']
    if not thickness_data.empty:
        thickness_min = thickness_data.min()
        thickness_max = thickness_data.max()
        thickness_range = thickness_max - thickness_min
        
        if thickness_range > 0:
            y_min = thickness_min - (yscale_percent * thickness_range)
            y_max = thickness_max + (yscale_percent * thickness_range)
            y_range = [y_min, y_max]
        else:
            y_range = None
    else:
        y_range = None
        
    return make_boxplot('Layer 1 Thickness', y_range, filtered_df)

# Callback for RADIUS vs Thickness plots
@app.callback(
    Output('radius-thickness-plots', 'children'),
    [Input('yscale-dropdown', 'value'),
     Input('trend-legend-radio', 'value'),
     Input('outlier-method-dropdown', 'value'),
     Input('outlier-threshold-input', 'value')]
)
def update_radius_thickness_plots(yscale_percent, show_trend_legend, outlier_method, outlier_threshold):
    filtered_df = get_filtered_dataframe(outlier_method, outlier_threshold)
    
    # Recalculate thickness range based on filtered data
    thickness_data = filtered_df[filtered_df['Label'] == 'Layer 1 Thickness']['Datum']
    if not thickness_data.empty:
        thickness_min = thickness_data.min()
        thickness_max = thickness_data.max()
        thickness_range = thickness_max - thickness_min
        
        if thickness_range > 0:
            y_min = thickness_min - (yscale_percent * thickness_range)
            y_max = thickness_max + (yscale_percent * thickness_range)
            y_range = [y_min, y_max]
        else:
            y_range = None
    else:
        y_range = None
        
    return make_radius_thickness_plots(y_range, show_trend_legend, filtered_df)

# Callback for RADIUS vs Thickness by Condition plots
@app.callback(
    Output('radius-thickness-condition-plots', 'children'),
    [Input('yscale-dropdown', 'value'),
     Input('trend-legend-radio', 'value'),
     Input('outlier-method-dropdown', 'value'),
     Input('outlier-threshold-input', 'value')]
)
def update_radius_thickness_condition_plots(yscale_percent, show_trend_legend, outlier_method, outlier_threshold):
    filtered_df = get_filtered_dataframe(outlier_method, outlier_threshold)
    
    # Recalculate thickness range based on filtered data
    thickness_data = filtered_df[filtered_df['Label'] == 'Layer 1 Thickness']['Datum']
    if not thickness_data.empty:
        thickness_min = thickness_data.min()
        thickness_max = thickness_data.max()
        thickness_range = thickness_max - thickness_min
        
        if thickness_range > 0:
            y_min = thickness_min - (yscale_percent * thickness_range)
            y_max = thickness_max + (yscale_percent * thickness_range)
            y_range = [y_min, y_max]
        else:
            y_range = None
    else:
        y_range = None
        
    return make_radius_thickness_by_condition_plots(y_range, show_trend_legend, filtered_df)

# Callback for Excel export
@app.callback(
    Output('export-status', 'children'),
    Input('export-button', 'n_clicks'),
    prevent_initial_call=True
)
def export_excel(n_clicks):
    if n_clicks:
        filename, success = export_summary_tables_to_excel()
        if success:
            return html.Div([
                html.P(f"✅ Successfully exported to: {filename}", 
                       style={'color': 'green', 'fontWeight': 'bold', 'margin': '10px 0'}),
                html.P("File saved to current directory", 
                       style={'color': 'gray', 'fontSize': '12px'})
            ])
        else:
            return html.Div([
                html.P("❌ Error exporting to Excel", 
                       style={'color': 'red', 'fontWeight': 'bold', 'margin': '10px 0'})
            ])
    return html.Div()

# Callbacks for summary tables with outlier filtering
@app.callback(
    Output('wafer-summary-table', 'children'),
    [Input('outlier-method-dropdown', 'value'),
     Input('outlier-threshold-input', 'value')]
)
def update_wafer_summary_table(outlier_method, outlier_threshold):
    filtered_df = get_filtered_dataframe(outlier_method, outlier_threshold)
    return make_statistical_summary_table(filtered_df)

@app.callback(
    Output('condition-summary-table', 'children'),
    [Input('outlier-method-dropdown', 'value'),
     Input('outlier-threshold-input', 'value')]
)
def update_condition_summary_table(outlier_method, outlier_threshold):
    filtered_df = get_filtered_dataframe(outlier_method, outlier_threshold)
    return make_statistical_summary_by_condition_table(filtered_df)

# Callback for condition average thickness plot
@app.callback(
    Output('condition-average-thickness-plot', 'children'),
    [Input('outlier-method-dropdown', 'value'),
     Input('outlier-threshold-input', 'value')]
)
def update_condition_average_thickness_plot(outlier_method, outlier_threshold):
    filtered_df = get_filtered_dataframe(outlier_method, outlier_threshold)
    return make_condition_average_thickness_plot(filtered_df)

# Callback for condition standard deviation plot
@app.callback(
    Output('condition-std-dev-plot', 'children'),
    [Input('outlier-method-dropdown', 'value'),
     Input('outlier-threshold-input', 'value')]
)
def update_condition_std_dev_plot(outlier_method, outlier_threshold):
    filtered_df = get_filtered_dataframe(outlier_method, outlier_threshold)
    return make_condition_std_dev_plot(filtered_df)

# Callback for outlier status display
@app.callback(
    Output('outlier-status', 'children'),
    [Input('outlier-method-dropdown', 'value'),
     Input('outlier-threshold-input', 'value')]
)
def update_outlier_status(outlier_method, outlier_threshold):
    if outlier_method == 'none':
        return html.Div()
    
    filtered_df = get_filtered_dataframe(outlier_method, outlier_threshold)
    
    # Count original and filtered Layer 1 Thickness data
    original_count = len(df[df['Label'] == 'Layer 1 Thickness'])
    filtered_count = len(filtered_df[filtered_df['Label'] == 'Layer 1 Thickness'])
    removed_count = original_count - filtered_count
    
    if removed_count > 0:
        return html.Div([
            html.P(f"🔍 Outlier Filtering Active: {outlier_method.replace('_', ' ').title()} (threshold: {outlier_threshold})", 
                   style={'color': 'blue', 'fontWeight': 'bold', 'margin': '5px 0'}),
            html.P(f"📊 Removed {removed_count} outlier points out of {original_count} Layer 1 Thickness measurements ({removed_count/original_count*100:.1f}%)", 
                   style={'color': 'orange', 'margin': '5px 0'})
        ], style={'padding': '10px', 'backgroundColor': '#e7f3ff', 'borderRadius': '5px', 'border': '1px solid #b3d9ff'})
    else:
        return html.Div([
            html.P(f"🔍 Outlier Filtering Active: {outlier_method.replace('_', ' ').title()} (threshold: {outlier_threshold})", 
                   style={'color': 'blue', 'fontWeight': 'bold', 'margin': '5px 0'}),
            html.P("✅ No outliers detected with current settings", 
                   style={'color': 'green', 'margin': '5px 0'})
        ], style={'padding': '10px', 'backgroundColor': '#e7f5e7', 'borderRadius': '5px', 'border': '1px solid #b3e6b3'})

# Callback for full data Excel export
@app.callback(
    Output('export-full-data-status', 'children'),
    Input('export-full-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def export_full_data_excel(n_clicks):
    if n_clicks:
        filename, success = export_full_data_to_excel()
        if success:
            return html.Div([
                html.P(f"✅ Successfully exported full data to: {filename}", 
                       style={'color': 'green', 'fontWeight': 'bold', 'margin': '10px 0'}),
                html.P("File saved to current directory", 
                       style={'color': 'gray', 'fontSize': '12px'})
            ])
        else:
            return html.Div([
                html.P("❌ Error exporting full data to Excel", 
                       style={'color': 'red', 'fontWeight': 'bold', 'margin': '10px 0'})
            ])
    return html.Div()

# New function to create a bar chart showing average Layer 1 Thickness by condition
def make_condition_average_thickness_plot(filtered_df=None):
    """Create a bar chart showing average Layer 1 Thickness by condition"""
    working_df = filtered_df if filtered_df is not None else df
    
    # Filter for Layer 1 Thickness data and exclude 'NA' conditions
    thickness_data = working_df[(working_df['Label'] == 'Layer 1 Thickness') & 
                               (working_df['Condition'] != 'NA')].copy()
    
    if thickness_data.empty:
        return html.Div("No Layer 1 Thickness data with conditions available")
    
    # Calculate average thickness by condition
    condition_stats = thickness_data.groupby('Condition')['Datum'].agg(['mean', 'std', 'count']).reset_index()
    condition_stats.columns = ['Condition', 'Mean', 'StdDev', 'Count']
    condition_stats['Mean'] = condition_stats['Mean'].round(1)
    condition_stats['StdDev'] = condition_stats['StdDev'].round(1)
    
    # Sort by condition name for consistent display
    condition_stats = condition_stats.sort_values('Condition')
    
    # Create line chart
    fig = px.line(
        condition_stats,
        x='Condition',
        y='Mean',
        title='Average Layer 1 Thickness by Process Condition',
        labels={
            'Mean': 'Average Layer 1 Thickness',
            'Condition': 'Process Condition'
        },
        markers=True  # Show markers on line
    )
    
    # Add error bars for standard deviation
    fig.update_traces(
        error_y=dict(
            type='data',
            array=condition_stats['StdDev'],
            visible=True
        ),
        mode='lines+markers+text',
        text=condition_stats['Mean'].round(1),
        texttemplate='%{text:.1f}',
        textposition='top center',
        marker=dict(size=8),
        line=dict(width=3),
        hovertemplate='<b>Condition: %{x}</b><br>' +
                      'Average Thickness: %{y:.1f}<br>' +
                      'Std Dev: %{error_y.array:.1f}<br>' +
                      'Sample Count: %{customdata}<br>' +
                      '<extra></extra>',
        customdata=condition_stats['Count']
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Process Condition',
        yaxis_title='Average Layer 1 Thickness',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )
    
    # Rotate x-axis labels if there are many conditions
    if len(condition_stats) > 5:
        fig.update_layout(xaxis_tickangle=-45)
    
    return dcc.Graph(figure=fig)

# New function to create a line chart showing standard deviation by condition
def make_condition_std_dev_plot(filtered_df=None):
    """Create a line chart showing standard deviation of Layer 1 Thickness by condition"""
    working_df = filtered_df if filtered_df is not None else df
    
    # Filter for Layer 1 Thickness data and exclude 'NA' conditions
    thickness_data = working_df[(working_df['Label'] == 'Layer 1 Thickness') & 
                               (working_df['Condition'] != 'NA')].copy()
    
    if thickness_data.empty:
        return html.Div("No Layer 1 Thickness data with conditions available")
    
    # Calculate standard deviation by condition
    condition_stats = thickness_data.groupby('Condition')['Datum'].agg(['mean', 'std', 'count']).reset_index()
    condition_stats.columns = ['Condition', 'Mean', 'StdDev', 'Count']
    condition_stats['Mean'] = condition_stats['Mean'].round(1)
    condition_stats['StdDev'] = condition_stats['StdDev'].round(1)
    
    # Sort by condition name for consistent display
    condition_stats = condition_stats.sort_values('Condition')
    
    # Create line chart for standard deviation
    fig = px.line(
        condition_stats,
        x='Condition',
        y='StdDev',
        title='Standard Deviation of Layer 1 Thickness by Process Condition',
        labels={
            'StdDev': 'Standard Deviation',
            'Condition': 'Process Condition'
        },
        markers=True  # Show markers on line
    )
    
    # Update traces for better visualization
    fig.update_traces(
        mode='lines+markers+text',
        text=condition_stats['StdDev'].round(1),
        texttemplate='%{text:.1f}',
        textposition='top center',
        marker=dict(size=8, color='red'),
        line=dict(width=3, color='red'),
        hovertemplate='<b>Condition: %{x}</b><br>' +
                      'Standard Deviation: %{y:.1f}<br>' +
                      'Mean Thickness: %{customdata[0]:.1f}<br>' +
                      'Sample Count: %{customdata[1]}<br>' +
                      '<extra></extra>',
        customdata=condition_stats[['Mean', 'Count']].values
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Process Condition',
        yaxis_title='Standard Deviation',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )
    
    # Rotate x-axis labels if there are many conditions
    if len(condition_stats) > 5:
        fig.update_layout(xaxis_tickangle=-45)
    
    return dcc.Graph(figure=fig)

app.layout = html.Div([
    html.H1("XML Data Analysis"),
    
    # Control Panel
    html.Div([
        # YSCALE Control
        html.Div([
            html.Label("YSCALE:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='yscale-dropdown',
                options=[{'label': f'{i}%', 'value': i/100} for i in range(0, 51)],
                value=0.05,  # Default to 5%
                style={'width': '150px', 'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        # Trend Legend Control
        html.Div([
            html.Label("Show Trend Line Legend:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RadioItems(
                id='trend-legend-radio',
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False}
                ],
                value=True,  # Default to True (show legend)
                inline=True,
                style={'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        # Outlier Filtering Control
        html.Div([
            html.Label("Outlier Filtering:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='outlier-method-dropdown',
                options=[
                    {'label': 'No Filtering', 'value': 'none'},
                    {'label': 'IQR Method (Global)', 'value': 'iqr'},
                    {'label': 'IQR Method (Per Wafer)', 'value': 'iqr_by_wafer'},
                    {'label': 'IQR Method (Per Condition)', 'value': 'iqr_by_condition'},
                    {'label': 'Z-Score (Global)', 'value': 'zscore'},
                    {'label': 'Z-Score (Per Wafer)', 'value': 'zscore_by_wafer'},
                    {'label': 'Modified Z-Score (Global)', 'value': 'modified_zscore'},
                    {'label': 'Percentile Method (Global)', 'value': 'percentile'}
                ],
                value='none',  # Default to no filtering
                style={'width': '200px', 'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'marginRight': '20px'}),
        
        # Outlier Threshold Control
        html.Div([
            html.Label("Threshold:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Input(
                id='outlier-threshold-input',
                type='number',
                value=1.5,
                min=0.5,
                max=5.0,
                step=0.1,
                style={'width': '80px', 'display': 'inline-block'}
            )
        ], style={'display': 'inline-block'})
    ], style={'margin': '20px 0', 'padding': '15px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px'}),
    
    # Outlier Status Display
    html.Div(id='outlier-status', style={'margin': '10px 0'}),
    
    html.H2("Overall Data - Layer 1 Thickness"),
    html.Div(id='thickness-boxplot'),
    
    html.H2("Overall Data - Goodness-of-Fit"),
    make_boxplot('Goodness-of-Fit'),
    
    html.Hr(),
    
    html.H2("Layer 1 Thickness vs Goodness-of-Fit Correlation"),
    make_scatter_plot(),
    
    html.Hr(),
    
    html.H2("Layer 1 Thickness vs RADIUS by WaferID"),
    html.Div(id='radius-thickness-plots'),
    
    html.Hr(),
    
    html.H2("Layer 1 Thickness vs RADIUS by Condition"),
    html.Div(id='radius-thickness-condition-plots'),
    
    html.Hr(),
    
    html.H2("Statistical Summary by WaferID"),
    html.Div(id='wafer-summary-table'),
    
    html.Hr(),
    
    html.H2("Statistical Summary by Condition"),
    html.Div(id='condition-summary-table'),
    
    html.Hr(),
    
    # New section for average thickness by condition
    html.H2("Average Layer 1 Thickness by Condition"),
    html.Div(id='condition-average-thickness-plot'),
    
    html.Hr(),
    
    # New section for standard deviation by condition
    html.H2("Standard Deviation of Layer 1 Thickness by Condition"),
    html.Div(id='condition-std-dev-plot'),
    
    html.Hr(),
    
    # Export Section
    html.Div([
        html.H3("Export Data to Excel"),
        
        # Summary Tables Export
        html.Div([
            html.H4("Statistical Summary Tables", style={'margin': '10px 0 5px 0'}),
            html.P("Export both statistical summary tables to Excel file with multiple sheets", 
                   style={'margin': '0 0 10px 0', 'fontSize': '14px'}),
            html.Button(
                "Export Summary Tables", 
                id="export-button", 
                n_clicks=0,
                style={
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'fontSize': '16px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'marginBottom': '10px',
                    'marginRight': '10px'
                }
            ),
            html.Div(id='export-status')
        ], style={'marginBottom': '20px'}),
        
        # Full Data Export
        html.Div([
            html.H4("Complete Dataset", style={'margin': '10px 0 5px 0'}),
            html.P("Export the complete dataframe with all measurements, RADIUS calculations, and process conditions", 
                   style={'margin': '0 0 10px 0', 'fontSize': '14px'}),
            html.Button(
                "Export Full Dataset", 
                id="export-full-data-button", 
                n_clicks=0,
                style={
                    'backgroundColor': '#28a745',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'fontSize': '16px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'marginBottom': '10px'
                }
            ),
            html.Div(id='export-full-data-status')
        ])
    ], style={'margin': '20px 0', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
    html.Div([
        html.H3("Export Full Data"),
        html.P("Export the complete dataset including all measurements and calculated RADIUS"),
        html.Button(
            "Export Full Data to Excel", 
            id="export-full-data-button", 
            n_clicks=0,
            style={
                'backgroundColor': '#28a745',
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'fontSize': '16px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'marginBottom': '10px'
            }
        ),
       
        html.Div(id='export-full-data-status')
    ], style={'margin': '20px 0', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
    html.Hr(),
    
    html.H2("Processed XML Files"),
    make_files_table()
])

if __name__ == '__main__':
    app.run_server(debug=True)