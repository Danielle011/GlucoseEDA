import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, time

class GlucoseAnalysis:
    def __init__(self):
        """Initialize GlucoseAnalysis component"""
        # Load glucose data
        self.glucose_df = pd.read_csv('data/processed_glucose_data.csv')
        self._prepare_dataframe()
        
    def _prepare_dataframe(self):
        """Prepare dataframe for analysis"""
        # Convert DateTime to pandas datetime
        self.glucose_df['DateTime'] = pd.to_datetime(self.glucose_df['DateTime'])
        
        # Add time of day column for easier filtering
        self.glucose_df['TimeOfDay'] = self.glucose_df['DateTime'].dt.time
        
        # Add period classification
        self.glucose_df['Period'] = self.glucose_df['TimeOfDay'].apply(self._classify_period)
    
    def _classify_period(self, time_val):
        """Classify time into periods"""
        if time(0, 0) <= time_val < time(8, 0):
            return 'Night'
        elif time(8, 0) <= time_val < time(11, 0):
            return 'Morning'
        elif time(11, 0) <= time_val < time(17, 0):
            return 'Afternoon'
        else:
            return 'Evening'

    def create_daily_pattern_plot(self):
        """Create daily pattern plot grouped by measurement session"""
        fig = go.Figure()
        
        # Get unique measurement numbers
        measurement_numbers = sorted(self.glucose_df['MeasurementNumber'].unique())
        
        # Create color scale for different sessions
        colors = [f'hsl({i * 360/len(measurement_numbers)}, 70%, 50%)'
                for i in range(len(measurement_numbers))]
        
        for idx, measurement in enumerate(measurement_numbers):
            session_data = self.glucose_df[
                self.glucose_df['MeasurementNumber'] == measurement
            ].copy()
            
            # Convert time to hour of day for grouping
            session_data['HourMinute'] = session_data['DateTime'].dt.hour + \
                                    session_data['DateTime'].dt.minute / 60
            
            # Group by hour-minute and calculate average and std
            grouped_data = session_data.groupby('HourMinute').agg({
                'GlucoseValue': ['mean', 'std']
            }).reset_index()
            
            # Rename columns for easier access
            grouped_data.columns = ['HourMinute', 'GlucoseMean', 'GlucoseStd']
            
            # Sort by hour-minute to ensure proper line connection
            grouped_data = grouped_data.sort_values('HourMinute')
            
            # Add the averaged line for this session
            fig.add_trace(
                go.Scatter(
                    x=grouped_data['HourMinute'],
                    y=grouped_data['GlucoseMean'],
                    name=f'Session {measurement}',
                    line=dict(color=colors[idx], width=2),
                    hovertemplate=(
                        'Session %{customdata[0]}<br>' +
                        'Time: %{x:.1f}h<br>' +
                        'Average Glucose: %{y:.1f} mg/dL<br>' +
                        'Std Dev: %{customdata[1]:.1f}<br>' +
                        '<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        np.full(len(grouped_data), measurement),
                        grouped_data['GlucoseStd']
                    ))
                )
            )

        # Add threshold line at 140
        fig.add_hline(y=140, line_dash="dash", line_color="red", 
                    opacity=0.5, name="Threshold")
        
        # Update layout
        fig.update_layout(
            title='Average Daily Glucose Patterns by CGM Session',
            xaxis_title='Time of Day (hours)',
            yaxis_title='Average Glucose (mg/dL)',
            xaxis=dict(
                tickmode='array',
                ticktext=['12 AM', '3 AM', '6 AM', '9 AM', '12 PM', 
                        '3 PM', '6 PM', '9 PM', '11:59 PM'],
                tickvals=[0, 3, 6, 9, 12, 15, 18, 21, 23.99],
                showgrid=True
            ),
            yaxis=dict(
                range=[70, 180],
                showgrid=True
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            hovermode='x unified'
        )
        
        return fig

    def calculate_daily_metrics(self):
        """Calculate daily metrics for glucose analysis with complete days only"""
        # First, let's check day completeness
        def is_complete_day(group):
            # Convert time to minutes since midnight for easier comparison
            minutes = group['DateTime'].dt.hour * 60 + group['DateTime'].dt.minute
            
            # Check if we have readings in both early and late parts of the day
            has_early = minutes.min() <= 30  # Data within first 30 minutes of day
            has_late = minutes.max() >= (23 * 60 + 30)  # Data within last 30 minutes of day
            
            # Check if we have consistent readings throughout the day
            # Expecting readings every 5 minutes (288 readings per day)
            # Allow for some missing readings but ensure good coverage
            min_readings = 240  # Allow for up to ~1.5 hours of missing data
            enough_readings = len(group) >= min_readings
            
            return has_early and has_late and enough_readings

        # Group by date and filter complete days
        daily_groups = self.glucose_df.groupby(self.glucose_df['DateTime'].dt.date)
        complete_days = []
        
        for date, group in daily_groups:
            if is_complete_day(group):
                complete_days.append(date)
        
        if not complete_days:
            st.warning("No complete days (24h) of data found. Showing statistics for all available data.")
            complete_days = list(daily_groups.groups.keys())
        
        # Calculate statistics only for complete days
        daily_stats = []
        for date in complete_days:
            day_data = daily_groups.get_group(date)
            stats = {
                'Date': date,
                'Average': day_data['GlucoseValue'].mean(),
                'Std': day_data['GlucoseValue'].std(),
                'TIR': (day_data['GlucoseValue'] < 140).mean() * 100,
            }
            stats['CV'] = (stats['Std'] / stats['Average']) * 100
            daily_stats.append(stats)
        
        daily_stats_df = pd.DataFrame(daily_stats)
        
        # Add number of readings for reference
        readings_count = {
            date: len(group) 
            for date, group in daily_groups
        }
        daily_stats_df['Readings'] = daily_stats_df['Date'].map(readings_count)
        
        return daily_stats_df

    def create_distribution_plots(self):
        """Create box plots for key glucose metrics"""
        # Calculate daily metrics
        daily_stats = self.calculate_daily_metrics()
        
        # Create figure with three subplots side by side
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'Daily Average Glucose<br>({len(daily_stats)} complete days)',
                f'Time in Range (<140 mg/dL)<br>({len(daily_stats)} complete days)',
                f'Glucose Variability (CV)<br>({len(daily_stats)} complete days)'
            )
        )
        
        # Add box plot for Average Glucose
        fig.add_trace(
            go.Box(
                y=daily_stats['Average'],
                name='Average',
                boxmean=True,
                marker_color='rgb(99,110,250)',
                hovertemplate=(
                    'Average Glucose: %{y:.1f} mg/dL<br>' +
                    'Date: %{customdata[0]}<br>' +
                    'Readings: %{customdata[1]}<br>' +
                    '<extra></extra>'
                ),
                customdata=np.column_stack((
                    daily_stats['Date'],
                    daily_stats['Readings']
                ))
            ),
            row=1, col=1
        )
        
        # Add box plot for Time in Range
        fig.add_trace(
            go.Box(
                y=daily_stats['TIR'],
                name='TIR',
                boxmean=True,
                marker_color='rgb(0,204,150)',
                hovertemplate=(
                    'Time in Range: %{y:.1f}%<br>' +
                    'Date: %{customdata[0]}<br>' +
                    'Readings: %{customdata[1]}<br>' +
                    '<extra></extra>'
                ),
                customdata=np.column_stack((
                    daily_stats['Date'],
                    daily_stats['Readings']
                ))
            ),
            row=1, col=2
        )
        
        # Add box plot for Coefficient of Variation
        fig.add_trace(
            go.Box(
                y=daily_stats['CV'],
                name='CV',
                boxmean=True,
                marker_color='rgb(239,85,59)',
                hovertemplate=(
                    'Coefficient of Variation: %{y:.1f}%<br>' +
                    'Date: %{customdata[0]}<br>' +
                    'Readings: %{customdata[1]}<br>' +
                    '<extra></extra>'
                ),
                customdata=np.column_stack((
                    daily_stats['Date'],
                    daily_stats['Readings']
                ))
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Distribution of Daily Glucose Metrics (Complete Days Only)',
                x=0.5
            ),
            showlegend=False,
            height=500,
            hovermode='y unified'
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text='mg/dL', row=1, col=1)
        fig.update_yaxes(title_text='Percentage', row=1, col=2)
        fig.update_yaxes(title_text='CV (%)', row=1, col=3)
        
        # Calculate overall statistics for metrics display
        overall_stats = {
            'avg_glucose': daily_stats['Average'].mean(),
            'avg_tir': daily_stats['TIR'].mean(),
            'avg_cv': daily_stats['CV'].mean(),
            'complete_days': len(daily_stats)
        }
        
        return fig, overall_stats

    def create_high_glucose_histogram(self):
        """Create histogram of high glucose occurrences"""
        # Create bins for different glucose ranges
        ranges = [(140, 160), (160, 180), (180, float('inf'))]
        colors = ['rgba(255,200,100,0.7)', 'rgba(255,150,50,0.7)', 
                 'rgba(255,50,50,0.7)']
        
        # Convert time to decimal hours for binning
        self.glucose_df['HourDecimal'] = (
            self.glucose_df['DateTime'].dt.hour +
            self.glucose_df['DateTime'].dt.minute / 60
        )
        
        fig = go.Figure()
        
        # Create histogram for each range
        for (lower, upper), color in zip(ranges, colors):
            mask = (self.glucose_df['GlucoseValue'] >= lower) & \
                  (self.glucose_df['GlucoseValue'] < upper)
            
            range_data = self.glucose_df[mask]
            
            fig.add_trace(
                go.Histogram(
                    x=range_data['HourDecimal'],
                    name=f'{lower}-{upper if upper != float("inf") else "+"} mg/dL',
                    nbinsx=24,
                    marker_color=color,
                    hovertemplate=(
                        'Time: %{x:.1f}h<br>' +
                        'Count: %{y}<br>' +
                        '<extra>%{fullData.name}</extra>'
                    )
                )
            )
        
        # Update layout
        fig.update_layout(
            title='High Glucose Events Distribution (24h)',
            xaxis_title='Time of Day (hours)',
            yaxis_title='Count of Events',
            barmode='stack',
            xaxis=dict(
                tickmode='array',
                ticktext=['12 AM', '3 AM', '6 AM', '9 AM', '12 PM', 
                         '3 PM', '6 PM', '9 PM', '11:59 PM'],
                tickvals=[0, 3, 6, 9, 12, 15, 18, 21, 23.99],
                showgrid=True
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            hovermode='x unified'
        )
        
        return fig

    def render(self):
        """Render all glucose analysis components"""
        # Create daily pattern plot
        st.subheader("Daily Glucose Patterns")
        daily_pattern_fig = self.create_daily_pattern_plot()
        st.plotly_chart(daily_pattern_fig, use_container_width=True)
        
        # Create high glucose histogram
        st.subheader("High Glucose Events Distribution")
        histogram_fig = self.create_high_glucose_histogram()
        st.plotly_chart(histogram_fig, use_container_width=True)

        # Create distribution plots
        st.subheader("Glucose Control Metrics Distribution")
        dist_fig, overall_stats = self.create_distribution_plots()
        
        # Display overall statistics in columns
        cols = st.columns(3)
        
        with cols[0]:
            st.metric(
                label="Overall Average Glucose",
                value=f"{overall_stats['avg_glucose']:.1f} mg/dL"
            )
        
        with cols[1]:
            st.metric(
                label="Overall Time in Range",
                value=f"{overall_stats['avg_tir']:.1f}%"
            )
        
        with cols[2]:
            st.metric(
                label="Overall Glucose Variability",
                value=f"{overall_stats['avg_cv']:.1f}%"
            )
        
        # Display distribution plots
        st.plotly_chart(dist_fig, use_container_width=True)