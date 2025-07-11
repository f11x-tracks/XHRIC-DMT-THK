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

def make_boxplot(label, y_range=None):
    dff = df[df['Label'] == label]
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
    gof_data = df_with_location[df_with_location['Label'] == 'Goodness-of-Fit'][['datetime', 'WaferID', 'dmt', 'location_id', 'Datum']].rename(columns={'Datum': 'GoodnessOfFit'})
    thickness_data = df_with_location[df_with_location['Label'] == 'Layer 1 Thickness'][['datetime', 'WaferID', 'dmt', 'location_id', 'Datum']].rename(columns={'Datum': 'Layer1Thickness'})
    
    # Merge based on location (same measurement point)
    merged_data = pd.merge(gof_data, thickness_data, on=['datetime', 'WaferID', 'dmt', 'location_id'], how='inner')
    
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
        hover_data=['WaferID', 'datetime']
    )
    
    fig.update_layout(
        xaxis_title='Goodness-of-Fit',
        yaxis_title='Layer 1 Thickness',
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return dcc.Graph(figure=fig)

def make_radius_thickness_plots(y_range=None):
    # Filter for Layer 1 Thickness data with valid RADIUS
    thickness_data = df[(df['Label'] == 'Layer 1 Thickness') & (df['RADIUS'].notna())].copy()
    
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
            customdata=wafer_data['dmt'],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'RADIUS: %{x:.2f}<br>' +
                          'Thickness: %{y:.2f}<br>' +
                          'DMT Type: %{customdata}<br>' +
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
                    showlegend=False,  # Hide from legend to avoid clutter
                    hoverinfo='skip'   # Don't show hover for trend line
                ))
                
            except Exception as e:
                # Fallback to simple linear regression if LOWESS fails
                try:
                    coeffs = np.polyfit(wafer_sorted['RADIUS'], wafer_sorted['Datum'], 1)
                    x_fit = np.linspace(wafer_sorted['RADIUS'].min(), wafer_sorted['RADIUS'].max(), 50)
                    y_fit = np.polyval(coeffs, x_fit)
                    
                    fig.add_trace(go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode='lines',
                        name=f'Linear: {wafer_id}',
                        line=dict(dash='dot', width=2, color=wafer_color),
                        showlegend=False,
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

def make_radius_thickness_by_condition_plots(y_range=None):
    # Filter for Layer 1 Thickness data with valid RADIUS and non-NA conditions
    thickness_data = df[(df['Label'] == 'Layer 1 Thickness') & 
                       (df['RADIUS'].notna()) & 
                       (df['Condition'] != 'NA')].copy()
    
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
            customdata=condition_data[['dmt', 'WaferID']],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'RADIUS: %{x:.2f}<br>' +
                          'Thickness: %{y:.2f}<br>' +
                          'WaferID: %{customdata[1]}<br>' +
                          'DMT Type: %{customdata[0]}<br>' +
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
                    showlegend=False,  # Hide from legend to avoid clutter
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
                        showlegend=False,
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

def make_statistical_summary_table():
    """Create a statistical summary table showing mean and std dev for each WaferID"""
    if df.empty:
        return html.Div("No data available for statistical summary")
    
    # Get unique labels and wafer IDs
    unique_labels = sorted(df['Label'].unique())
    unique_wafers = sorted(df['WaferID'].unique())
    
    summary_data = []
    
    for wafer_id in unique_wafers:
        wafer_data = df[df['WaferID'] == wafer_id]
        
        # Get condition for this wafer
        condition = wafer_data['Condition'].iloc[0] if not wafer_data.empty else 'NA'
        
        # Only process Layer 1 Thickness data
        label_data = wafer_data[wafer_data['Label'] == 'Layer 1 Thickness']['Datum']
        
        if not label_data.empty:
            mean_val = label_data.mean()
            std_val = label_data.std()
            count_val = len(label_data)
            
            summary_data.append({
                'WaferID': wafer_id,
                'Condition': condition,
                'Measurement': 'Layer 1 Thickness',
                'Mean': round(mean_val, 4),
                'Std Dev': round(std_val, 4),
                'Count': count_val,
                'Min': round(label_data.min(), 4),
                'Max': round(label_data.max(), 4)
            })
    
    if not summary_data:
        return html.Div("No statistical data to display")
    
    # Sort summary data by Condition
    summary_data = sorted(summary_data, key=lambda x: x['Condition'])
    
    # Create the summary table using dash_table
    summary_table = dash_table.DataTable(
        data=summary_data,
        columns=[
            {"name": "Wafer ID", "id": "WaferID"},
            {"name": "Condition", "id": "Condition"},
            {"name": "Measurement Type", "id": "Measurement"},
            {"name": "Mean", "id": "Mean", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "Std Dev", "id": "Std Dev", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "Count", "id": "Count", "type": "numeric"},
            {"name": "Min", "id": "Min", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "Max", "id": "Max", "type": "numeric", "format": {"specifier": ".4f"}}
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
                'if': {'filter_query': '{Condition} = "No-Relax"'},
                'backgroundColor': 'rgba(255, 182, 193, 0.3)',
            },
            {
                'if': {'filter_query': '{Condition} contains "Cast"'},
                'backgroundColor': 'rgba(173, 216, 230, 0.3)',
            },
            {
                'if': {'filter_query': '{Condition} = "NA"'},
                'backgroundColor': 'rgba(220, 220, 220, 0.5)',
            }
        ],
        sort_action="native",
        filter_action="native",
        page_size=30
    )
    
    return html.Div([
        html.H3(f"Statistical Summary by WaferID ({len(unique_wafers)} wafers)"),
        html.P("Mean and standard deviation for each measurement type by wafer with condition"),
        summary_table
    ])

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
    Input('yscale-dropdown', 'value')
)
def update_thickness_boxplot(yscale_percent):
    if thickness_range > 0:
        y_min = thickness_min - (yscale_percent * thickness_range)
        y_max = thickness_max + (yscale_percent * thickness_range)
        y_range = [y_min, y_max]
    else:
        y_range = None
    return make_boxplot('Layer 1 Thickness', y_range)

# Callback for RADIUS vs Thickness plots
@app.callback(
    Output('radius-thickness-plots', 'children'),
    Input('yscale-dropdown', 'value')
)
def update_radius_thickness_plots(yscale_percent):
    if thickness_range > 0:
        y_min = thickness_min - (yscale_percent * thickness_range)
        y_max = thickness_max + (yscale_percent * thickness_range)
        y_range = [y_min, y_max]
    else:
        y_range = None
    return make_radius_thickness_plots(y_range)

# Callback for RADIUS vs Thickness by Condition plots
@app.callback(
    Output('radius-thickness-condition-plots', 'children'),
    Input('yscale-dropdown', 'value')
)
def update_radius_thickness_condition_plots(yscale_percent):
    if thickness_range > 0:
        y_min = thickness_min - (yscale_percent * thickness_range)
        y_max = thickness_max + (yscale_percent * thickness_range)
        y_range = [y_min, y_max]
    else:
        y_range = None
    return make_radius_thickness_by_condition_plots(y_range)

app.layout = html.Div([
    html.H1("XML Data Analysis"),
    
    # YSCALE Control
    html.Div([
        html.Label("YSCALE:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='yscale-dropdown',
            options=[{'label': f'{i}%', 'value': i/100} for i in range(0, 51)],
            value=0.05,  # Default to 5%
            style={'width': '150px', 'display': 'inline-block'}
        )
    ], style={'margin': '20px 0', 'padding': '10px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px'}),
    
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
    
    html.H2("Statistical Summary"),
    make_statistical_summary_table(),
    
    html.Hr(),
    
    html.H2("Processed XML Files"),
    make_files_table()
])

if __name__ == '__main__':
    app.run_server(debug=True)