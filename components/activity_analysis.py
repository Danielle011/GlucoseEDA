import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

class ActivityAnalysis:
    def __init__(self):
        """Initialize ActivityAnalysis component"""
        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        """Load and prepare data for analysis"""
        # Load all required data
        self.meal_df = pd.read_csv('data/processed_meal_data.csv')
        self.glucose_df = pd.read_csv('data/processed_glucose_data.csv')
        self.activity_df = pd.read_csv('data/aggregated_activity_data.csv')
        
        # Convert datetime columns
        self.meal_df['meal_time'] = pd.to_datetime(self.meal_df['meal_time'])
        self.glucose_df['DateTime'] = pd.to_datetime(self.glucose_df['DateTime'])
        self.activity_df['start_time'] = pd.to_datetime(self.activity_df['start_time'])
        self.activity_df['end_time'] = pd.to_datetime(self.activity_df['end_time'])
        
        # Calculate meal window duration and add labels
        self._calculate_meal_windows()
        self._add_activity_labels()
        self._add_carb_labels()

    def create_meal_activity_plots(self):
        """Create glucose response curves by activity levels, separated by carb category"""
        # Filter data for analysis
        analysis_df = self.meal_df[
            (self.meal_df['window_duration'] >= 110) &
            (self.meal_df['meal_type'] != 'Snack')
        ].copy()
        
        # Calculate glucose responses for each meal
        responses = []
        
        for _, meal in analysis_df.iterrows():
            # Get glucose values for 2h after meal
            meal_end = meal['meal_time'] + pd.Timedelta(minutes=120)
            mask = (
                (self.glucose_df['DateTime'] >= meal['meal_time']) &
                (self.glucose_df['DateTime'] <= meal_end) &
                (self.glucose_df['MeasurementNumber'] == meal['measurement_number'])
            )
            glucose_data = self.glucose_df[mask].copy()
            
            if len(glucose_data) < 10:  # Skip if too few readings
                continue
                
            # Calculate baseline
            baseline_idx = (
                glucose_data['DateTime'] - meal['meal_time']
            ).dt.total_seconds().abs().idxmin()
            baseline = glucose_data.loc[baseline_idx, 'GlucoseValue']
            
            # Calculate integer minutes from meal
            glucose_data['minutes'] = (
                (glucose_data['DateTime'] - meal['meal_time'])
                .dt.components['minutes'] + 
                (glucose_data['DateTime'] - meal['meal_time'])
                .dt.components['hours'] * 60
            ).astype(int)
            
            glucose_data['glucose_change'] = glucose_data['GlucoseValue'] - baseline
            
            # Add to responses with meal metadata
            responses.append({
                'meal_time': meal['meal_time'],
                'carb_label': meal['carb_label'],
                'activity_label': meal['activity_label'],
                'baseline': baseline,
                'minutes': list(glucose_data['minutes']),
                'glucose_change': list(glucose_data['glucose_change']),
                'glucose_values': list(glucose_data['GlucoseValue'])
            })
        
        # Create DataFrame with proper minute-by-minute data
        flat_responses = []
        for resp in responses:
            for min_idx, minute in enumerate(resp['minutes']):
                flat_responses.append({
                    'carb_label': resp['carb_label'],
                    'activity_label': resp['activity_label'],
                    'minutes': minute,
                    'glucose_change': resp['glucose_change'][min_idx]
                })
        
        response_df = pd.DataFrame(flat_responses)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'High Carb Meals (>75g)',
                'Moderate Carb Meals (30-75g)',
                'Low Carb Meals (<30g)'
            ),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Color scheme for activity levels
        colors = {
            'inactive': {
                'fill': 'rgba(239,85,59,0.4)',
                'line': 'rgba(239,85,59,0.9)'
            },
            'moderate': {
                'fill': 'rgba(99,110,250,0.4)',
                'line': 'rgba(99,110,250,0.9)'
            },
            'active': {
                'fill': 'rgba(0,204,150,0.4)',
                'line': 'rgba(0,204,150,0.9)'
            }
        }
        
        # Row mapping for carb categories
        carb_rows = {'high': 1, 'moderate': 2, 'low': 3}
        
        # Find overall y-axis range for all plots
        y_min = response_df['glucose_change'].min()
        y_max = response_df['glucose_change'].max()
        y_range = [min(-10, y_min * 1.1), max(120, y_max * 1.1)]  # Add some padding
        
        # Create average response curves for each combination
        for carb in ['high', 'moderate', 'low']:
            for activity in ['active', 'moderate', 'inactive']:
                mask = (
                    (response_df['carb_label'] == carb) &
                    (response_df['activity_label'] == activity)
                )
                
                if not mask.any():
                    continue
                    
                # Calculate mean glucose change for each minute
                stats = (response_df[mask]
                    .groupby('minutes')['glucose_change']
                    .agg(['mean', 'count', 'std'])
                    .reset_index()
                    .sort_values('minutes'))
                
                # Add trace for mean line
                fig.add_trace(
                    go.Scatter(
                        x=stats['minutes'],
                        y=stats['mean'],
                        name=f'{activity.capitalize()}',
                        mode='lines',
                        line=dict(
                            color=colors[activity]['line'],
                            width=2
                        ),
                        legendgroup=activity,
                        showlegend=carb == 'high',
                        hovertemplate=(
                            'Time: %{x}min<br>' +
                            'Glucose Change: %{y:.1f} mg/dL<br>' +
                            f'Activity: {activity.capitalize()}<br>' +
                            'n=%{customdata[0]:.0f}<br>' +
                            '<extra></extra>'
                        ),
                        customdata=np.column_stack([stats['count']])
                    ),
                    row=carb_rows[carb],
                    col=1
                )
                
                # Add error bands
                fig.add_trace(
                    go.Scatter(
                        x=stats['minutes'].tolist() + stats['minutes'].tolist()[::-1],
                        y=(stats['mean'] + stats['std']).tolist() + 
                        (stats['mean'] - stats['std']).tolist()[::-1],
                        fill='tonexty',
                        fillcolor=colors[activity]['fill'],
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip',
                        legendgroup=activity
                    ),
                    row=carb_rows[carb],
                    col=1
                )
        
        # Add vertical lines at 60 minutes for each subplot
        for row in range(1, 4):
            fig.add_vline(
                x=60,
                line_dash="dash",
                line_color="rgba(255,255,255,0.4)",
                annotation_text="1 hour",
                annotation_font_color="white",
                row=row,
                col=1
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            title={
                'text': 'Glucose Response by Activity Level for Different Carb Loads',
                'y': 0.95
            },
            showlegend=True,
            legend=dict(
                title="Activity Level",
                yanchor="top",
                y=0.95,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.1)"
            ),
            hovermode='x unified',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Minutes After Meal",
            row=3,
            col=1,
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[0, 120]
        )
        
        for row in range(1, 4):
            fig.update_yaxes(
                title_text="Glucose Change from Baseline (mg/dL)",
                row=row,
                col=1,
                gridcolor='rgba(128,128,128,0.2)',
                zerolinecolor='rgba(255,255,255,0.2)',
                range=y_range
            )
        
        return fig

    def display_activity_statistics(self):
        """Display statistical analysis results"""
        stats_results = self.perform_activity_statistics()
        
        st.write("### Statistical Analysis of Activity Impact on Peak Glucose Rise")
        st.write("Testing hypothesis: Inactive meals have higher peak glucose rise than active meals")
        
        # Create formatted tables
        strict_data = {
            'Carb Level': [],
            'Inactive n (median)': [],
            'Active n (median)': [],
            'p-value': []
        }
        
        all_data = {
            'Carb Level': [],
            'Inactive n (median)': [],
            'Active n (median)': [],
            'p-value': []
        }
        
        for carb in ['high', 'moderate', 'all']:
            # Strict Binary
            if f'{carb}_strict' in stats_results:
                result = stats_results[f'{carb}_strict']
                strict_data['Carb Level'].append(carb.capitalize())
                strict_data['Inactive n (median)'].append(
                    f"{result['n_inactive']} ({result['median_inactive']:.1f})"
                )
                strict_data['Active n (median)'].append(
                    f"{result['n_active']} ({result['median_active']:.1f})"
                )
                strict_data['p-value'].append(f"{result['p_value']:.4f}")
            
            # All Binary
            if f'{carb}_all' in stats_results:
                result = stats_results[f'{carb}_all']
                all_data['Carb Level'].append(carb.capitalize())
                all_data['Inactive n (median)'].append(
                    f"{result['n_inactive']} ({result['median_inactive']:.1f})"
                )
                all_data['Active n (median)'].append(
                    f"{result['n_active']} ({result['median_active']:.1f})"
                )
                all_data['p-value'].append(f"{result['p_value']:.4f}")
        
        # Display tables
        st.write("#### Strict Binary Classification (Excluding Middle Ground)")
        st.write("Inactive: <300 total steps AND ≤200 max interval steps")
        st.write("Active: >1000 total steps")
        st.table(pd.DataFrame(strict_data))
        
        st.write("#### Complete Binary Classification (Including All Meals)")
        st.write("Inactive: <300 total steps AND ≤200 max interval steps")
        st.write("Active: All others")
        st.table(pd.DataFrame(all_data))

    def _calculate_meal_windows(self):
        """Calculate duration until next meal or 120 mins"""
        # Sort meals by time
        self.meal_df = self.meal_df.sort_values('meal_time')
        
        # Calculate time to next meal
        self.meal_df['next_meal_time'] = self.meal_df['meal_time'].shift(-1)
        self.meal_df['window_duration'] = (
            (self.meal_df['next_meal_time'] - self.meal_df['meal_time'])
            .dt.total_seconds() / 60  # Convert to minutes
        )
        
        # Handle last meal and cap at 120 minutes
        self.meal_df['window_duration'] = self.meal_df['window_duration'].fillna(120)
        self.meal_df['window_duration'] = self.meal_df['window_duration'].clip(upper=120)

    def _add_activity_labels(self):
        """Add activity labels based on post-meal activity"""
        activity_data = []
        
        for _, meal in self.meal_df.iterrows():
            total_steps, max_interval_steps = self._calculate_activity_metrics(
                meal['meal_time'],
                meal['window_duration']
            )
            
            # Apply activity classification rules
            if total_steps < 600 and max_interval_steps <= 200:
                label = 'inactive'
            elif max_interval_steps > 1000 or total_steps > 1500:
                label = 'active'
            else:
                label = 'moderate'
                
            activity_data.append({
                'total_steps': total_steps,
                'max_interval_steps': max_interval_steps,
                'activity_label': label
            })
        
        # Add activity data to meal_df
        activity_df = pd.DataFrame(activity_data)
        self.meal_df = pd.concat([
            self.meal_df,
            activity_df
        ], axis=1)

    def _add_carb_labels(self):
        """Add carbohydrate content labels based on ADA guidelines"""
        conditions = [
            (self.meal_df['carbohydrates'] < 30),
            (self.meal_df['carbohydrates'] >= 30) & (self.meal_df['carbohydrates'] <= 75),
            (self.meal_df['carbohydrates'] > 75)
        ]
        choices = ['low', 'moderate', 'high']
        
        self.meal_df['carb_label'] = np.select(conditions, choices, default='moderate')

    def _calculate_activity_metrics(self, meal_time, window_duration):
        """Calculate activity metrics for meal window"""
        window_end = meal_time + pd.Timedelta(minutes=window_duration)
        
        # Get activity data for the window
        mask = (
            (self.activity_df['start_time'] >= meal_time) & 
            (self.activity_df['start_time'] < window_end)
        )
        window_activity = self.activity_df[mask]
        
        if len(window_activity) == 0:
            return 0, 0
            
        total_steps = window_activity['steps'].sum()
        max_interval_steps = window_activity['steps'].max()
        
        return total_steps, max_interval_steps

    def _calculate_individual_net_auc(self, glucose_data, start_time, time_limit):
        """Calculate NET AUC for individual meal response"""
        # Get data within time limit
        mask = glucose_data['minutes'] <= time_limit
        data = glucose_data[mask].copy()
        
        if len(data) < 2:  # Need at least 2 points for trapezoid
            return None
            
        # Calculate NET AUC using trapezoidal rule
        auc = np.trapz(data['glucose_change'], x=data['minutes'])
        return auc
    
    def create_activity_glucose_scatter(self, metric_type='net_auc', include_snacks=True):
        """
        Create scatter plot of first hour steps vs glucose metric
        metric_type: 'net_auc' or 'peak_rise'
        """
        # Filter data
        analysis_df = self.meal_df[
            self.meal_df['window_duration'] >= 110
        ].copy()
        
        if not include_snacks:
            analysis_df = analysis_df[analysis_df['meal_type'] != 'Snack']
        
        # Store individual metrics
        meal_metrics = []
        
        # Calculate metrics for each meal
        for _, meal in analysis_df.iterrows():
            # Get glucose values for 1h after meal
            meal_end = meal['meal_time'] + pd.Timedelta(minutes=60)
            glucose_mask = (
                (self.glucose_df['DateTime'] >= meal['meal_time']) &
                (self.glucose_df['DateTime'] <= meal_end) &
                (self.glucose_df['MeasurementNumber'] == meal['measurement_number'])
            )
            glucose_data = self.glucose_df[glucose_mask].copy()
            
            # Get activity data for 1h after meal
            activity_mask = (
                (self.activity_df['start_time'] >= meal['meal_time']) &
                (self.activity_df['start_time'] <= meal_end)
            )
            activity_data = self.activity_df[activity_mask].copy()
            
            if len(glucose_data) < 10:  # Skip if too few readings
                continue
                
            # Calculate baseline and glucose changes
            baseline_idx = (
                glucose_data['DateTime'] - meal['meal_time']
            ).dt.total_seconds().abs().idxmin()
            baseline = glucose_data.loc[baseline_idx, 'GlucoseValue']
            
            glucose_data['minutes'] = (
                (glucose_data['DateTime'] - meal['meal_time'])
                .dt.components['minutes'] + 
                (glucose_data['DateTime'] - meal['meal_time'])
                .dt.components['hours'] * 60
            ).astype(int)
            
            glucose_data['glucose_change'] = glucose_data['GlucoseValue'] - baseline
            
            # Calculate metrics
            net_auc = self._calculate_individual_net_auc(glucose_data, meal['meal_time'], 60)
            # Fixed - only look at first hour data
            peak_rise = glucose_data[glucose_data['minutes'] <= 60]['glucose_change'].max()
            
            # Calculate total steps in first hour
            total_steps = activity_data['steps'].sum() if not activity_data.empty else 0
            
            if net_auc is not None:
                meal_metrics.append({
                    'meal_time': meal['meal_time'],
                    'meal_type': meal['meal_type'],
                    'carb_label': meal['carb_label'],
                    'carbohydrates': meal['carbohydrates'],
                    'first_hour_steps': total_steps,
                    'net_auc_1h': net_auc,
                    'peak_rise_1h': peak_rise
                })
        
        # Create DataFrame
        metrics_df = pd.DataFrame(meal_metrics)
        
        # Select y-axis metric
        y_metric = 'net_auc_1h' if metric_type == 'net_auc' else 'peak_rise_1h'
        y_title = 'First Hour NET AUC (mg/dL × min)' if metric_type == 'net_auc' else 'Peak Glucose Rise (mg/dL)'
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color scheme
        colors = {
            'high': 'rgb(239,85,59)',
            'moderate': 'rgb(99,110,250)',
            'low': 'rgb(0,204,150)'
        }
        
        # Add traces for each carb category
        for carb in ['high', 'moderate', 'low']:
            mask = metrics_df['carb_label'] == carb
            fig.add_trace(
                go.Scatter(
                    x=metrics_df[mask]['first_hour_steps'],
                    y=metrics_df[mask][y_metric],
                    mode='markers',
                    name=f'{carb.capitalize()} Carb',
                    marker=dict(
                        color=colors[carb],
                        size=10,
                        line=dict(
                            color='white',
                            width=1
                        ),
                        opacity=0.7
                    ),
                    hovertemplate=(
                        'Steps: %{x}<br>' +
                        f'{y_title.split(" (")[0]}: %{{y:.1f}}<br>' +
                        'Carbs: %{customdata[0]:.1f}g<br>' +
                        'Type: %{customdata[1]}<br>' +
                        '<extra></extra>'
                    ),
                    customdata=metrics_df[mask][['carbohydrates', 'meal_type']]
                )
            )
        
        # Add trend line for all data
        z = np.polyfit(metrics_df['first_hour_steps'], metrics_df[y_metric], 1)
        p = np.poly1d(z)
        
        x_range = np.linspace(0, metrics_df['first_hour_steps'].max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                name='Trend Line',
                line=dict(
                    color='rgba(255,255,255,0.5)',
                    dash='dash'
                ),
                hoverinfo='skip'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'First Hour Steps vs {y_title.split(" (")[0]}' + 
                (' (Including Snacks)' if include_snacks else ' (Excluding Snacks)'),
            xaxis_title='Steps in First Hour After Meal',
            yaxis_title=y_title,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.1)"
            ),
            hovermode='closest'
        )
        
        # Add vertical reference lines for potential thresholds
        for steps in [500, 1000, 1500]:
            fig.add_vline(
                x=steps,
                line_dash="dot",
                line_color="rgba(255,255,255,0.2)",
                annotation_text=f"{steps} steps",
                annotation_position="top"
            )
        
        return fig
    
    def perform_activity_statistics(self):
        """Perform statistical analysis of activity impact on glucose peaks"""
        # Filter data
        analysis_df = self.meal_df[
            (self.meal_df['window_duration'] >= 110) &
            (self.meal_df['meal_type'] != 'Snack') &
            (self.meal_df['carb_label'].isin(['moderate', 'high']))
        ].copy()
        
        # Store results
        results = []
        
        for _, meal in analysis_df.iterrows():
            # Get first hour glucose data
            meal_end = meal['meal_time'] + pd.Timedelta(minutes=60)
            glucose_mask = (
                (self.glucose_df['DateTime'] >= meal['meal_time']) &
                (self.glucose_df['DateTime'] <= meal_end) &
                (self.glucose_df['MeasurementNumber'] == meal['measurement_number'])
            )
            glucose_data = self.glucose_df[glucose_mask].copy()
            
            # Get first hour activity data
            activity_mask = (
                (self.activity_df['start_time'] >= meal['meal_time']) &
                (self.activity_df['start_time'] <= meal_end)
            )
            activity_data = self.activity_df[activity_mask].copy()
            
            if len(glucose_data) < 10:
                continue
                
            # Calculate baseline and peak rise
            baseline_idx = (
                glucose_data['DateTime'] - meal['meal_time']
            ).dt.total_seconds().abs().idxmin()
            baseline = glucose_data.loc[baseline_idx, 'GlucoseValue']
            
            glucose_data['glucose_change'] = glucose_data['GlucoseValue'] - baseline
            peak_rise = glucose_data['glucose_change'].max()
            
            # Calculate activity metrics
            total_steps = activity_data['steps'].sum() if not activity_data.empty else 0
            max_interval_steps = activity_data['steps'].max() if not activity_data.empty else 0
            
            # Classify activity - Strict Binary
            if total_steps < 300 and max_interval_steps <= 200:
                activity_strict = 'inactive'
            elif total_steps > 1000:
                activity_strict = 'active'
            else:
                activity_strict = 'excluded'
                
            # Classify activity - All Binary
            if total_steps < 300 and max_interval_steps <= 200:
                activity_all = 'inactive'
            else:
                activity_all = 'active'
            
            results.append({
                'carb_label': meal['carb_label'],
                'activity_strict': activity_strict,
                'activity_all': activity_all,
                'peak_rise': peak_rise,
                'total_steps': total_steps,
                'max_interval_steps': max_interval_steps
            })
        
        results_df = pd.DataFrame(results)
        
        # Perform statistical tests
        stats_results = {}
        
        for carb in ['high', 'moderate', 'all']:
            # Filter data by carb category
            if carb == 'all':
                carb_data = results_df
            else:
                carb_data = results_df[results_df['carb_label'] == carb]
            
            # Strict Binary Analysis
            strict_data = carb_data[carb_data['activity_strict'] != 'excluded']
            if len(strict_data) > 0:
                inactive_peaks = strict_data[strict_data['activity_strict'] == 'inactive']['peak_rise']
                active_peaks = strict_data[strict_data['activity_strict'] == 'active']['peak_rise']
                
                if len(inactive_peaks) > 0 and len(active_peaks) > 0:
                    statistic, pvalue = stats.mannwhitneyu(
                        inactive_peaks, 
                        active_peaks, 
                        alternative='greater'
                    )
                    
                    stats_results[f'{carb}_strict'] = {
                        'n_inactive': len(inactive_peaks),
                        'n_active': len(active_peaks),
                        'median_inactive': inactive_peaks.median(),
                        'median_active': active_peaks.median(),
                        'p_value': pvalue
                    }
            
            # All Binary Analysis
            if len(carb_data) > 0:
                inactive_peaks = carb_data[carb_data['activity_all'] == 'inactive']['peak_rise']
                active_peaks = carb_data[carb_data['activity_all'] == 'active']['peak_rise']
                
                if len(inactive_peaks) > 0 and len(active_peaks) > 0:
                    statistic, pvalue = stats.mannwhitneyu(
                        inactive_peaks, 
                        active_peaks, 
                        alternative='greater'
                    )
                    
                    stats_results[f'{carb}_all'] = {
                        'n_inactive': len(inactive_peaks),
                        'n_active': len(active_peaks),
                        'median_inactive': inactive_peaks.median(),
                        'median_active': active_peaks.median(),
                        'p_value': pvalue
                    }
        
        return stats_results
    
    def render(self):
        """Main render method for activity analysis"""

        # Add glucose response curves
        st.subheader("Glucose Response Patterns")
        fig_response = self.create_meal_activity_plots()
        if fig_response is not None:
            st.plotly_chart(fig_response, use_container_width=True)

        st.title("Post-meal Activity Analysis")
        
        # Get filtered data counts
        analysis_df = self.meal_df[
            (self.meal_df['window_duration'] >= 110) &
            (self.meal_df['meal_type'] != 'Snack')
        ]
        
        # Display data summary
        st.write("Analysis includes:")
        cols = st.columns(3)
        
        # Count meals by carb category
        carb_counts = analysis_df['carb_label'].value_counts()
        with cols[0]:
            st.metric("Low Carb Meals", carb_counts.get('low', 0))
        with cols[1]:
            st.metric("Moderate Carb Meals", carb_counts.get('moderate', 0))
        with cols[2]:
            st.metric("High Carb Meals", carb_counts.get('high', 0))
            
        # Display activity-glucose relationship analysis
        st.subheader("Activity-Glucose Response Analysis")
        
        # Add scatter plots and statistical analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("NET AUC vs Steps (Including Snacks)")
            fig_auc_with = self.create_activity_glucose_scatter(
                metric_type='net_auc',
                include_snacks=True
            )
            st.plotly_chart(fig_auc_with, use_container_width=True)
        
        with col2:
            st.write("NET AUC vs Steps (Excluding Snacks)")
            fig_auc_without = self.create_activity_glucose_scatter(
                metric_type='net_auc',
                include_snacks=False
            )
            st.plotly_chart(fig_auc_without, use_container_width=True)
        
        # Display Peak Rise Analysis
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("Peak Rise vs Steps (Including Snacks)")
            fig_peak_with = self.create_activity_glucose_scatter(
                metric_type='peak_rise',
                include_snacks=True
            )
            st.plotly_chart(fig_peak_with, use_container_width=True)
        
        with col4:
            st.write("Peak Rise vs Steps (Excluding Snacks)")
            fig_peak_without = self.create_activity_glucose_scatter(
                metric_type='peak_rise',
                include_snacks=False
            )
            st.plotly_chart(fig_peak_without, use_container_width=True)
        
        # Display statistical analysis
        self.display_activity_statistics()