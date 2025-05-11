import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import pytz
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator
from .glucose_activity_statistics import GlucoseActivityStatistics

class ActivityAnalysis:
    def __init__(self):
        self.load_and_prepare_data()

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
            return 0, 0, 0
            
        total_steps = window_activity['steps'].sum()
        max_interval_steps = window_activity['steps'].max()
        total_flights = window_activity['flights'].sum()
        
        return total_steps, max_interval_steps, total_flights
    
    def render_statistical_analysis(self):
        """Render statistical analysis section"""
        st.subheader("6. Statistical Analysis of Activity's Effect on Glucose Response")
        
        # Initialize the statistics class with your dataframes
        stats = GlucoseActivityStatistics(self.meal_df, self.glucose_df, self.activity_df)
        
        # Let the statistics class handle the dashboard creation
        stats.add_to_streamlit()

    def load_and_prepare_data(self):
        # Load all required data
        self.meal_df = pd.read_csv('data/processed_meal_data.csv')
        self.activity_df = pd.read_csv('data/aggregated_activity_data.csv')
        self.glucose_df = pd.read_csv('data/processed_glucose_data.csv')  # Add this line
        
        # Convert datetime columns
        self.meal_df['meal_time'] = pd.to_datetime(self.meal_df['meal_time'])
        self.activity_df['start_time'] = pd.to_datetime(self.activity_df['start_time'])
        self.activity_df['end_time'] = pd.to_datetime(self.activity_df['end_time'])
        self.glucose_df['DateTime'] = pd.to_datetime(self.glucose_df['DateTime'])  # Add this line
        
        # Add carb labels
        conditions = [
            (self.meal_df['carbohydrates'] < 30),
            (self.meal_df['carbohydrates'] >= 30) & (self.meal_df['carbohydrates'] <= 75),
            (self.meal_df['carbohydrates'] > 75)
        ]
        choices = ['low', 'moderate', 'high']
        self.meal_df['carb_label'] = np.select(conditions, choices, default='moderate')
        
        # Calculate meal window duration
        self._calculate_meal_windows()
        
        # Add activity metrics
        activity_data = []
        for _, meal in self.meal_df.iterrows():
            total_steps, max_interval_steps, total_flights = self._calculate_activity_metrics(
                meal['meal_time'],
                meal['window_duration']
            )
            activity_data.append({
                'total_steps': total_steps,
                'max_interval_steps': max_interval_steps,
                'total_flights': total_flights
            })
        
        # Add activity data to meal_df
        activity_df = pd.DataFrame(activity_data)
        self.meal_df = pd.concat([self.meal_df.reset_index(drop=True), 
                                activity_df.reset_index(drop=True)], axis=1)

    def _calculate_meal_windows(self):
        """Calculate duration until next meal or 120 mins"""
        self.meal_df = self.meal_df.sort_values('meal_time')
        self.meal_df['next_meal_time'] = self.meal_df['meal_time'].shift(-1)
        self.meal_df['window_duration'] = (
            (self.meal_df['next_meal_time'] - self.meal_df['meal_time'])
            .dt.total_seconds() / 60
        )
        self.meal_df['window_duration'] = self.meal_df['window_duration'].fillna(120)
        self.meal_df['window_duration'] = self.meal_df['window_duration'].clip(upper=120)

    def create_temporal_activity_pattern(self):
        """Create 2x2 grid showing temporal activity pattern by meal type"""
        # Filter valid meals and get active meals only
        valid_meals = self.meal_df[self.meal_df['window_duration'] >= 120].copy()
        active_meals = valid_meals[
            (valid_meals['total_steps'] >= 600) | 
            (valid_meals['max_interval_steps'] > 200)
        ]

        # Create figure with 2x2 subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Breakfast', 'Lunch', 'Dinner', 'All Meals'),
            specs=[[{'secondary_y': True}]*2]*2,
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        # Process each meal type
        row_col_map = {
            'Breakfast': (1, 1),
            'Lunch': (1, 2),
            'Dinner': (2, 1),
            'All': (2, 2)
        }

        max_steps = 0
        max_flights = 0

        # Calculate overall max values first
        for meal_type, (row, col) in row_col_map.items():
            meal_data = active_meals if meal_type == 'All' else active_meals[active_meals['meal_type'] == meal_type]
            if len(meal_data) == 0:
                continue
                
            for interval in range(0, 121, 10):
                interval_activities = []
                for _, meal in meal_data.iterrows():
                    meal_time = meal['meal_time']
                    activity_start_time = meal_time.floor('10min')
                    target_interval_start = activity_start_time + pd.Timedelta(minutes=interval)
                    target_interval_end = target_interval_start + pd.Timedelta(minutes=10)
                    
                    mask = (
                        (self.activity_df['start_time'] >= target_interval_start) &
                        (self.activity_df['start_time'] < target_interval_end)
                    )
                    interval_activity = self.activity_df[mask]
                    
                    interval_activities.append({
                        'steps': interval_activity['steps'].sum(),
                        'flights': interval_activity['flights'].sum()
                    })
                
                if interval_activities:
                    activities_df = pd.DataFrame(interval_activities)
                    max_steps = max(max_steps, activities_df['steps'].max())
                    max_flights = max(max_flights, activities_df['flights'].max())

        # Now create plots for each meal type
        for meal_type, (row, col) in row_col_map.items():
            meal_data = active_meals if meal_type == 'All' else active_meals[active_meals['meal_type'] == meal_type]
            
            if len(meal_data) == 0:
                continue

            # Process activity for each meal
            interval_stats = []
            for interval in range(0, 121, 10):
                interval_activities = []
                
                for _, meal in meal_data.iterrows():
                    meal_time = meal['meal_time']
                    activity_start_time = meal_time.floor('10min')
                    target_interval_start = activity_start_time + pd.Timedelta(minutes=interval)
                    target_interval_end = target_interval_start + pd.Timedelta(minutes=10)
                    
                    mask = (
                        (self.activity_df['start_time'] >= target_interval_start) &
                        (self.activity_df['start_time'] < target_interval_end)
                    )
                    interval_activity = self.activity_df[mask]
                    
                    interval_activities.append({
                        'steps': interval_activity['steps'].sum(),
                        'flights': interval_activity['flights'].sum()
                    })
                
                activities_df = pd.DataFrame(interval_activities)
                
                stats = {
                    'interval': interval,
                    'avg_steps': activities_df['steps'].mean(),
                    'std_steps': activities_df['steps'].std(),
                    'avg_flights': activities_df['flights'].mean(),
                    'flights_q1': activities_df['flights'].quantile(0.25),
                    'flights_median': activities_df['flights'].median(),
                    'flights_q3': activities_df['flights'].quantile(0.75),
                }
                
                interval_stats.append(stats)

            # Create dataframe for this meal type
            interval_df = pd.DataFrame(interval_stats)

            # Add traces for this meal type
            # Steps line
            fig.add_trace(
                go.Scatter(
                    x=interval_df['interval'],
                    y=interval_df['avg_steps'],
                    mode='lines',
                    name='Steps' if row == 1 and col == 1 else None,
                    line=dict(color='rgb(31, 119, 180)', width=2),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col, secondary_y=False
            )

            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=interval_df['interval'].tolist() + interval_df['interval'].tolist()[::-1],
                    y=(interval_df['avg_steps'] + interval_df['std_steps']).tolist() +
                    (interval_df['avg_steps'] - interval_df['std_steps']).tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.2)',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col, secondary_y=False
            )

            # Flights line
            fig.add_trace(
                go.Scatter(
                    x=interval_df['interval'],
                    y=interval_df['avg_flights'],
                    mode='lines',
                    name='Flights' if row == 1 and col == 1 else None,
                    line=dict(color='rgb(255, 127, 14)', width=2),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col, secondary_y=True
            )

            # Box plots for flights
            for idx, interval in enumerate(interval_df['interval']):
                fig.add_trace(
                    go.Box(
                        x=[interval],
                        y=[interval_df.iloc[idx]['flights_q1'], 
                        interval_df.iloc[idx]['flights_median'],
                        interval_df.iloc[idx]['flights_q3']],
                        name=f'{interval} min',
                        marker_color='rgb(255, 127, 14)',
                        opacity=0.5,
                        showlegend=False
                    ),
                    row=row, col=col, secondary_y=True
                )

        # Update layout
        fig.update_layout(
            height=800,
            title='Temporal Activity Pattern Analysis',
            showlegend=True,
            template='plotly_dark',
            legend=dict(
                yanchor="top",
                y=0.95,
                xanchor="right",
                x=0.95
            )
        )

        # Update axes
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(
                    title_text="Minutes After Meal",
                    range=[0, 120],
                    row=row,
                    col=col
                )
                fig.update_yaxes(
                    title_text="Steps" if col == 1 else None,
                    range=[0, max_steps * 1.1],
                    row=row,
                    col=col,
                    secondary_y=False
                )
                fig.update_yaxes(
                    title_text="Flights" if col == 2 else None,
                    range=[0, max_flights * 1.1],
                    row=row,
                    col=col,
                    secondary_y=True
                )

        return fig

    def create_activity_distribution_plot(self):
        """Create 2x2 grid with stacked histograms and box plots for activity distribution"""
        # Filter for valid meals and active meals
        valid_meals = self.meal_df[self.meal_df['window_duration'] >= 120].copy()
        active_meals = valid_meals[
            (valid_meals['total_steps'] >= 600) |
            (valid_meals['max_interval_steps'] > 200)
        ]
        
        # Create figure with 2x2 subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Steps Distribution by Meal Type', 
                'Flights Distribution by Meal Type',
                'Steps Distribution (All Meals)',
                'Flights Distribution (All Meals)'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Colors for different meal types
        colors = {
            'Breakfast': 'rgb(31, 119, 180)',  # blue
            'Lunch': 'rgb(255, 127, 14)',      # orange
            'Dinner': 'rgb(44, 160, 44)',       # green
            'Snack': 'rgb(130, 202, 157)'
        }
        
        # Calculate bin ranges with new bin sizes
        min_steps = 0  # Start from 0 for better context
        max_steps = active_meals['total_steps'].max()  # Use actual maximum
        step_bin_size = 1000  # Bin size for steps
        step_bins = list(range(min_steps, int(max_steps) + step_bin_size, step_bin_size))
        
        min_flights = 0  # Start from 0 for better context
        max_flights = active_meals['total_flights'].max()  # Use actual maximum
        flight_bin_size = 5  # Bin size for flights
        flight_bins = list(range(min_flights, int(max_flights) + flight_bin_size, flight_bin_size))
        
        # Create stacked bars for steps
        for meal_type in ['Breakfast', 'Lunch', 'Dinner', 'Snack']:
            meal_data = active_meals[active_meals['meal_type'] == meal_type]
            
            if len(meal_data) == 0:
                continue
            
            # Calculate histogram data for steps
            hist_data = np.histogram(
                meal_data['total_steps'],
                bins=step_bins
            )
            
            # Create bin labels for steps using 'k' format
            bin_labels = [f"{bin_start/1000:.0f}k-{(bin_start + step_bin_size)/1000:.0f}k" 
                        for bin_start in step_bins[:-1]]
            
            # Add bar trace for steps
            fig.add_trace(
                go.Bar(
                    x=bin_labels,
                    y=hist_data[0],
                    name=meal_type,
                    marker_color=colors[meal_type],
                    hovertemplate=(
                        f"{meal_type}<br>" +
                        "Steps: %{x}<br>" +
                        "Count: %{y}<br>" +
                        "<extra></extra>"
                    )
                ),
                row=1, col=1
            )
        
        # Create stacked bars for flights
        for meal_type in ['Breakfast', 'Lunch', 'Dinner', 'Snack']:
            meal_data = active_meals[active_meals['meal_type'] == meal_type]
            
            if len(meal_data) == 0:
                continue
            
            # Calculate histogram data for flights
            hist_data = np.histogram(
                meal_data['total_flights'],
                bins=flight_bins
            )
            
            # Create bin labels for flights with range format
            bin_labels = [f"{bin_start:.0f}-{bin_start + flight_bin_size:.0f}" 
                        for bin_start in flight_bins[:-1]]
            
            # Add bar trace for flights
            fig.add_trace(
                go.Bar(
                    x=bin_labels,
                    y=hist_data[0],
                    name=meal_type,
                    marker_color=colors[meal_type],
                    showlegend=False,  # Hide duplicate legends
                    hovertemplate=(
                        f"{meal_type}<br>" +
                        "Flights: %{x}<br>" +
                        "Count: %{y}<br>" +
                        "<extra></extra>"
                    )
                ),
                row=1, col=2
            )
        
        # Add box plot for steps
        fig.add_trace(
            go.Box(
                y=active_meals['total_steps'],
                name='Steps',
                boxpoints='outliers',
                jitter=0.3,
                pointpos=0,  # Centered position for outliers
                marker=dict(
                    color='rgb(31, 119, 180)',
                    size=4  # Smaller points for better visibility
                ),
                hovertemplate=(
                    "Steps<br>" +
                    "Value: %{y:,.0f}<br>" +
                    "<extra></extra>"
                )
            ),
            row=2, col=1
        )
        
        # Add box plot for flights
        fig.add_trace(
            go.Box(
                y=active_meals['total_flights'],
                name='Flights',
                boxpoints='outliers',
                jitter=0.3,
                pointpos=0,  # Centered position for outliers
                marker=dict(
                    color='rgb(31, 119, 180)',
                    size=4  # Smaller points for better visibility
                ),
                hovertemplate=(
                    "Flights<br>" +
                    "Value: %{y:.1f}<br>" +
                    "<extra></extra>"
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,  # Increased height for 2x2 grid
            title_text="Distribution of Active Post-meal Physical Activity",
            showlegend=True,
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100, r=20, b=50, l=60)
        )
        
        # Update axes for histograms
        fig.update_xaxes(
            title_text="Total Steps in 2h Window",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Total Flights in 2h Window",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            row=1, col=2
        )
        
        # Update axes for box plots
        fig.update_xaxes(
            title_text="",
            showticklabels=False,
            row=2, col=1
        )
        fig.update_xaxes(
            title_text="",
            showticklabels=False,
            row=2, col=2
        )
        
        # Update y-axes
        fig.update_yaxes(
            title_text="Count of Meals",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Count of Meals",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Steps",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Flights",
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            row=2, col=2
        )
        
        return fig
    
    def create_activity_timing_quartile_plot(self):
        """Create plot showing activity timing patterns colored by total steps quartile groups"""
        # Filter for valid meals and active meals
        valid_meals = self.meal_df[self.meal_df['window_duration'] >= 120].copy()
        active_meals = valid_meals[
            (valid_meals['total_steps'] >= 600) |
            (valid_meals['max_interval_steps'] > 200)
        ]

        # Calculate quartiles for total steps
        quartiles = active_meals['total_steps'].quantile([0.25, 0.5, 0.75])
        
        # Create color mapping based on quartiles with distinct colors
        colors = {
            'Q1': 'rgba(44,123,182,0.5)',   # Blue
            'Q2': 'rgba(215,48,39,0.5)',    # Red
            'Q3': 'rgba(35,139,69,0.5)',    # Green
            'Q4': 'rgba(255,127,0,0.5)'     # Orange
        }
        
        # Colors for average lines (full opacity)
        avg_colors = {
            'Q1': 'rgb(44,123,182)',   # Blue
            'Q2': 'rgb(215,48,39)',    # Red
            'Q3': 'rgb(35,139,69)',    # Green
            'Q4': 'rgb(255,127,0)'     # Orange
        }

        def get_quartile_group(steps):
            if steps <= quartiles[0.25]:
                return 'Q1'
            elif steps <= quartiles[0.50]:
                return 'Q2'
            elif steps <= quartiles[0.75]:
                return 'Q3'
            else:
                return 'Q4'

        # Function to get steps for each 10-min interval after a meal
        def get_interval_steps(meal_time):
            intervals = []
            interval_start = meal_time.floor('10min')
            
            for i in range(12):
                current_start = interval_start + pd.Timedelta(minutes=(i*10))
                current_end = current_start + pd.Timedelta(minutes=10)
                
                mask = (
                    (self.activity_df['start_time'] >= current_start) &
                    (self.activity_df['start_time'] < current_end)
                )
                interval_activity = self.activity_df[mask]
                interval_steps = interval_activity['steps'].sum() if len(interval_activity) > 0 else 0
                
                intervals.append({
                    'interval_start': i*10,  # Numeric value for ordering
                    'steps': interval_steps
                })
            
            return intervals

        # Calculate intervals for each meal
        all_intervals = []
        for _, meal in active_meals.iterrows():
            meal_intervals = get_interval_steps(meal['meal_time'])
            meal_id = meal.name
            quartile_group = get_quartile_group(meal['total_steps'])
            total_steps = meal['total_steps']
            
            for interval in meal_intervals:
                interval['meal_id'] = meal_id
                interval['quartile_group'] = quartile_group
                interval['total_steps'] = total_steps
                all_intervals.append(interval)

        # Convert to DataFrame for easier analysis
        intervals_df = pd.DataFrame(all_intervals)

        # Create figure
        fig = go.Figure()

        # Process each quartile
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            quartile_data = intervals_df[intervals_df['quartile_group'] == quartile]
            
            # Calculate average and standard deviation
            avg_data = quartile_data.groupby('interval_start')['steps'].agg(['mean', 'std']).reset_index()
            
            # 1. First, add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=avg_data['interval_start'].tolist() + avg_data['interval_start'].tolist()[::-1],
                    y=(avg_data['mean'] + avg_data['std']).tolist() + 
                    (avg_data['mean'] - avg_data['std']).tolist()[::-1],
                    fill='toself',
                    fillcolor=colors[quartile].replace('0.5)', '0.15)'),
                    line=dict(width=0),
                    name=f'CI - {quartile}',
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            # 2. Add average line
            fig.add_trace(
                go.Scatter(
                    x=avg_data['interval_start'],
                    y=avg_data['mean'],
                    mode='lines',
                    line=dict(
                        color=avg_colors[quartile],
                        width=3
                    ),
                    name=f'Average - {quartile}',
                    legendgroup=quartile,
                    showlegend=True,
                    hovertemplate=(
                        f"Quartile {quartile}<br>" +
                        "Time: %{x:d}-" + str(10) + " min<br>" +
                        "Average Steps: %{y:.0f}<br>" +
                        "<extra></extra>"
                    )
                )
            )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Activity Timing Pattern by Total Steps Quartile',
                'x': 0.5,
                'y': 0.95
            },
            showlegend=True,
            height=600,
            hovermode='x unified',
            legend=dict(
                title="Total Steps Group",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            # Update x-axis with proper ordering
            xaxis=dict(
                title_text='Minutes After Meal',
                tickmode='array',
                ticktext=[f'{i}-{i+10}' for i in range(0, 120, 10)],  # Create range labels
                tickvals=list(range(0, 120, 10)),  # Use numeric values for proper ordering
                showgrid=True
            ),
            yaxis_title='Steps per 10-min Interval',
            # Add annotations for quartile ranges
            annotations=[
                dict(
                    x=1.02,
                    y=0.8,
                    xref="paper",
                    yref="paper",
                    text=f"Q1: ≤{quartiles[0.25]:.0f} steps",
                    showarrow=False,
                    font=dict(size=10)
                ),
                dict(
                    x=1.02,
                    y=0.7,
                    xref="paper",
                    yref="paper",
                    text=f"Q2: {quartiles[0.25]:.0f}-{quartiles[0.50]:.0f}",
                    showarrow=False,
                    font=dict(size=10)
                ),
                dict(
                    x=1.02,
                    y=0.6,
                    xref="paper",
                    yref="paper",
                    text=f"Q3: {quartiles[0.50]:.0f}-{quartiles[0.75]:.0f}",
                    showarrow=False,
                    font=dict(size=10)
                ),
                dict(
                    x=1.02,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text=f"Q4: >{quartiles[0.75]:.0f} steps",
                    showarrow=False,
                    font=dict(size=10)
                )
            ]
        )

        return fig

    def create_activity_carb_sankey(self):
        """Create Sankey diagram showing distribution of meals across activity levels and carb categories"""
        # 1. First level: Total meals and window validity
        total_meals = len(self.meal_df)
        valid_meals = self.meal_df[self.meal_df['window_duration'] >= 120].copy()
        invalid_meals = total_meals - len(valid_meals)

        # 2. Second level: Active vs Inactive classification
        active_meals = valid_meals[
            (valid_meals['total_steps'] >= 600) | 
            (valid_meals['max_interval_steps'] > 200)
        ]
        inactive_meals = valid_meals[
            (valid_meals['total_steps'] < 600) & 
            (valid_meals['max_interval_steps'] <= 200)
        ]

        # 3. Calculate quartiles for active meals (same as in create_activity_timing_quartile_plot)
        quartiles = active_meals['total_steps'].quantile([0.25, 0.5, 0.75])
        
        def get_quartile_group(steps):
            if steps <= quartiles[0.25]:
                return 'Q1'
            elif steps <= quartiles[0.50]:
                return 'Q2'
            elif steps <= quartiles[0.75]:
                return 'Q3'
            else:
                return 'Q4'

        # 4. Calculate carb level distributions for inactive meals
        inactive_carb_counts = {
            'low': len(inactive_meals[inactive_meals['carb_label'] == 'low']),
            'moderate': len(inactive_meals[inactive_meals['carb_label'] == 'moderate']),
            'high': len(inactive_meals[inactive_meals['carb_label'] == 'high'])
        }

        # 5. Calculate carb and quartile distributions for active meals
        active_distribution = {
            'low': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0},
            'moderate': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0},
            'high': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        }

        for _, meal in active_meals.iterrows():
            carb_level = meal['carb_label']
            quartile = get_quartile_group(meal['total_steps'])
            active_distribution[carb_level][quartile] += 1

        # 6. Create node labels
        node_labels = [
            # Level 1: Total
            f'Total Meals\n({total_meals})',
            # Level 2: Window validity
            f'Valid Window\n({len(valid_meals)})',
            f'Invalid Window\n({invalid_meals})',
            # Level 3: Activity levels
            f'Post-meal Inactive\n({len(inactive_meals)})',
            f'Post-meal Active\n({len(active_meals)})',
            # Level 4: Carb levels for inactive
            f'Low Carb\n({inactive_carb_counts["low"]})',
            f'Moderate Carb\n({inactive_carb_counts["moderate"]})',
            f'High Carb\n({inactive_carb_counts["high"]})',
            # Level 4&5: Carb levels and quartiles for active
            f'Low Carb Active\n({sum(active_distribution["low"].values())})',
            f'Moderate Carb Active\n({sum(active_distribution["moderate"].values())})',
            f'High Carb Active\n({sum(active_distribution["high"].values())})',
        ]

        # Add quartile nodes for each carb level
        for carb in ['low', 'moderate', 'high']:
            for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                node_labels.append(
                    f'{carb.title()} {quartile}\n({active_distribution[carb][quartile]})'
                )

        # 7. Create source-target pairs and values for links
        sources = []
        targets = []
        values = []

        # Level 1 to 2: Total to Valid/Invalid
        sources.extend([0, 0])
        targets.extend([1, 2])
        values.extend([len(valid_meals), invalid_meals])

        # Level 2 to 3: Valid to Active/Inactive
        sources.extend([1, 1])
        targets.extend([3, 4])
        values.extend([len(inactive_meals), len(active_meals)])

        # Level 3 to 4: Inactive to Carb levels
        sources.extend([3, 3, 3])
        targets.extend([5, 6, 7])
        values.extend([
            inactive_carb_counts['low'],
            inactive_carb_counts['moderate'],
            inactive_carb_counts['high']
        ])

        # Level 3 to 4: Active to Carb levels
        sources.extend([4, 4, 4])
        targets.extend([8, 9, 10])
        values.extend([
            sum(active_distribution['low'].values()),
            sum(active_distribution['moderate'].values()),
            sum(active_distribution['high'].values())
        ])

        # Level 4 to 5: Active Carb levels to Quartiles
        quartile_start_idx = 11
        for carb_idx, carb in enumerate(['low', 'moderate', 'high']):
            carb_source_idx = 8 + carb_idx  # Index of the carb level node
            for q_idx, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                target_idx = quartile_start_idx + (carb_idx * 4) + q_idx
                sources.append(carb_source_idx)
                targets.append(target_idx)
                values.append(active_distribution[carb][quartile])

        # 8. Create color scheme
        node_colors = [
            '#7CB9E8',  # Total
            '#90EE90', '#FFB4B4',  # Window validity
            '#C3B1E1', '#F8C8DC',  # Activity levels
            '#E6E6FA', '#E6E6FA', '#E6E6FA',  # Inactive carb levels
            '#FFE4E1', '#FFE4E1', '#FFE4E1',  # Active carb levels
        ]
        
        # Add colors for quartile nodes
        for _ in range(12):  # 3 carb levels * 4 quartiles
            node_colors.append('#F0F8FF')

        # 9. Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                hovertemplate='Count: %{value}<br>%{target.label}<extra></extra>'
            )
        )])

        # 10. Update layout
        fig.update_layout(
            title_text="Post-meal Activity and Carbohydrate Distribution Analysis",
            font=dict(size=10),
            height=800,
            margin=dict(t=60, l=20, r=20, b=20)
        )

        return fig
    
    def prepare_glucose_activity_data(self):
        """Prepare data for glucose and activity visualization by carb and activity quartiles"""
        # Filter for valid meals
        valid_meals = self.meal_df[self.meal_df['window_duration'] >= 120].copy()

        # Calculate activity quartiles (using same logic as before)
        active_meals = valid_meals[
            (valid_meals['total_steps'] >= 600) | 
            (valid_meals['max_interval_steps'] > 200)
        ]
        quartiles = active_meals['total_steps'].quantile([0.25, 0.5, 0.75])

        def get_quartile_group(steps):
            if steps <= quartiles[0.25]:
                return 'Q1'
            elif steps <= quartiles[0.50]:
                return 'Q2'
            elif steps <= quartiles[0.75]:
                return 'Q3'
            else:
                return 'Q4'

        # First get only active meals
        valid_meals = valid_meals[
            (valid_meals['total_steps'] >= 600) | 
            (valid_meals['max_interval_steps'] > 200)
        ]
        # Then add quartile labels to only active meals
        valid_meals['activity_quartile'] = valid_meals['total_steps'].apply(get_quartile_group)
        
        
        # Initialize data structure to store averaged values
        averaged_data = {
            carb_level: {
                quartile: {
                    'glucose_values': [],
                    'glucose_times': [],
                    'step_values': [],
                    'step_times': [],
                    'n_meals': 0
                }
                for quartile in ['Q1', 'Q2', 'Q3', 'Q4']
            }
            for carb_level in ['low', 'moderate', 'high']
        }
        
        # Process each meal
        for _, meal in valid_meals.iterrows():
            carb_level = meal['carb_label']
            quartile = meal['activity_quartile']
            
            # Get glucose data
            glucose_data = self.glucose_df[
                (self.glucose_df['DateTime'] >= meal['meal_time']) &
                (self.glucose_df['DateTime'] <= meal['meal_time'] + pd.Timedelta(minutes=120)) &
                (self.glucose_df['MeasurementNumber'] == meal['measurement_number'])
            ].copy()
            
            if len(glucose_data) > 0:
                # Normalize glucose to baseline
                baseline = glucose_data.iloc[0]['GlucoseValue']
                glucose_data['glucose_normalized'] = glucose_data['GlucoseValue'] - baseline
                glucose_data['minutes'] = (
                    (glucose_data['DateTime'] - meal['meal_time'])
                    .dt.total_seconds() / 60
                )
                
                # Get activity data
                activity_data = []
                for interval in range(0, 121, 10):
                    interval_start = meal['meal_time'] + pd.Timedelta(minutes=interval)
                    interval_end = interval_start + pd.Timedelta(minutes=10)
                    
                    mask = (
                        (self.activity_df['start_time'] >= interval_start) &
                        (self.activity_df['start_time'] < interval_end)
                    )
                    interval_activity = self.activity_df[mask]
                    activity_data.append({
                        'minutes': interval,
                        'steps': interval_activity['steps'].sum() if len(interval_activity) > 0 else 0
                    })
                
                # Store data
                averaged_data[carb_level][quartile]['glucose_values'].append(
                    glucose_data[['minutes', 'glucose_normalized']].values.tolist()
                )
                averaged_data[carb_level][quartile]['step_values'].append(
                    pd.DataFrame(activity_data).values.tolist()
                )
                averaged_data[carb_level][quartile]['n_meals'] += 1
        
        return averaged_data, quartiles

    def create_glucose_activity_grid(self):
        """Create 3x4 grid of glucose and activity plots by carb level and activity quartile"""
        # Get processed data
        averaged_data, quartiles = self.prepare_glucose_activity_data()
        
        # Create subplot titles with correct quartile access
        subplot_titles = []
        q_values = [quartiles[0.25], quartiles[0.50], quartiles[0.75]]
        
        for i in range(12):  # 3 rows x 4 columns
            if i % 4 == 0:  # Q1
                steps = int(q_values[0])
                title = f'Q1 (<{steps} steps)'
            elif i % 4 == 1:  # Q2
                steps = int(q_values[1])
                title = f'Q2 ({int(q_values[0])}-{steps} steps)'
            elif i % 4 == 2:  # Q3
                steps = int(q_values[2])
                title = f'Q3 ({int(q_values[1])}-{steps} steps)'
            else:  # Q4
                steps = int(q_values[2])
                title = f'Q4 (>{steps} steps)'
            subplot_titles.append(title)
        
        # Create figure with 3x4 grid
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True}]*4]*3,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Colors for different carb levels
        carb_colors = {
            'low': 'rgb(44,123,182)',
            'moderate': 'rgb(215,48,39)',
            'high': 'rgb(35,139,69)'
        }
        
        # Process each combination
        for row_idx, carb_level in enumerate(['low', 'moderate', 'high'], 1):
            for col_idx, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4'], 1):
                data = averaged_data[carb_level][quartile]
                
                if data['n_meals'] > 0:
                    # Process glucose data
                    glucose_times = []
                    glucose_values = []
                    for meal_data in data['glucose_values']:
                        for time, value in meal_data:
                            glucose_times.append(time)
                            glucose_values.append(value)
                    
                    # Calculate mean and std for glucose with proper rounding
                    df_glucose = pd.DataFrame({
                        'time': glucose_times,
                        'value': glucose_values
                    })
                    glucose_avg = df_glucose.groupby(
                        df_glucose['time'].round()  # Round to exact minute
                    )['value'].agg(['mean', 'std']).reset_index()
                    
                    # Process activity data
                    step_times = []
                    step_values = []
                    for meal_data in data['step_values']:
                        for time, value in meal_data:
                            step_times.append(time)
                            step_values.append(value)
                    
                    # Calculate mean for steps
                    df_steps = pd.DataFrame({
                        'time': step_times,
                        'value': step_values
                    })
                    steps_avg = df_steps.groupby('time')['value'].mean().reset_index()
                    
                    # Add baseline reference line
                    fig.add_hline(
                        y=0,
                        line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'),
                        row=row_idx, col=col_idx,
                        secondary_y=False
                    )
                    
                    # Add glucose confidence interval
                    fig.add_trace(
                        go.Scatter(
                            x=glucose_avg['time'].tolist() + glucose_avg['time'].tolist()[::-1],
                            y=(glucose_avg['mean'] + glucose_avg['std']).tolist() + 
                            (glucose_avg['mean'] - glucose_avg['std']).tolist()[::-1],
                            fill='toself',
                            fillcolor=carb_colors[carb_level].replace('rgb', 'rgba').replace(')', ',0.3)'),
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=row_idx, col=col_idx,
                        secondary_y=False
                    )
                    
                    # Add glucose line
                    fig.add_trace(
                        go.Scatter(
                            x=glucose_avg['time'],
                            y=glucose_avg['mean'],
                            name=f'{carb_level.title()} Carb',
                            line=dict(
                                color=carb_colors[carb_level],
                                width=3,
                                shape='spline'  # Add smoothing
                            ),
                            showlegend=(col_idx == 1),
                            hovertemplate=(
                                "Time: %{x:.0f} min<br>" +
                                "Glucose Change: %{y:.1f} mg/dL<br>" +
                                f"n={data['n_meals']}<br>" +
                                "<extra></extra>"
                            )
                        ),
                        row=row_idx, col=col_idx,
                        secondary_y=False
                    )
                    
                    # Add step bars
                    fig.add_trace(
                        go.Bar(
                            x=steps_avg['time'],
                            y=steps_avg['value'],
                            name='Steps' if (row_idx == 1 and col_idx == 1) else None,
                            marker_color='rgba(128,128,128,0.3)',
                            showlegend=(row_idx == 1 and col_idx == 1),
                            hovertemplate=(
                                "Time: %{x:.0f} min<br>" +
                                "Steps: %{y:.0f}<br>" +
                                "<extra></extra>"
                            )
                        ),
                        row=row_idx, col=col_idx,
                        secondary_y=True
                    )
        
        # Update layout
        fig.update_layout(
            height=900,
            width=1200,
            title_text='Glucose Response and Activity Patterns by Carb Level and Activity Quartile',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent background
            paper_bgcolor='rgba(0,0,0,0)'   # Transparent background
        )
        
        # Update axes
        for row in range(1, 4):
            for col in range(1, 5):
                # Update x-axes
                fig.update_xaxes(
                    title_text="Minutes After Meal" if row == 3 else None,
                    range=[0, 120],
                    dtick=30,
                    gridcolor='rgba(128,128,128,0.1)',
                    showgrid=True,
                    row=row, col=col
                )
                
                # Update primary y-axes (glucose)
                fig.update_yaxes(
                    title_text="Glucose Change (mg/dL)" if col == 1 else None,
                    range=[-20, 80],
                    dtick=20,
                    gridcolor='rgba(128,128,128,0.1)',
                    showgrid=True,
                    row=row, col=col,
                    secondary_y=False
                )
                
                # Update secondary y-axes (steps)
                fig.update_yaxes(
                    title_text="Steps per 10min" if col == 4 else None,
                    range=[0, 1000],
                    dtick=200,
                    showgrid=False,  # No grid for steps axis
                    row=row, col=col,
                    secondary_y=True
                )
        
        # Add annotations for carb levels
        for idx, carb_level in enumerate(['Low Carb (<30g)', 'Moderate Carb (30-75g)', 'High Carb (>75g)']):
            fig.add_annotation(
                text=carb_level,
                xref="paper",
                yref="paper",
                x=-0.1,
                y=0.8 - (idx * 0.33),
                showarrow=False,
                textangle=-90,
                font=dict(size=12)
            )
        
        return fig

    def render(self):
        """Render all activity analysis components"""
        st.header("Activity Analysis")
        
        # 1. Activity and Carbohydrate Distribution Overview
        st.subheader("1. Activity and Carbohydrate Distribution")
        
        # Add explanation
        st.markdown("""
            This Sankey diagram shows the flow of meals through different classifications:
            1. **Window Validity**: Meals with at least 120 minutes until the next meal
            2. **Activity Level**: 
            - Active: ≥600 total steps OR >200 steps in any 10-min interval
            - Inactive: <600 total steps AND ≤200 steps per interval
            3. **Carbohydrate Content**:
            - Low: <30g
            - Moderate: 30-75g
            - High: >75g
            4. **Activity Quartiles** (for active meals only):
            - Based on total steps in 2-hour window
            - Q1-Q4: From lowest to highest activity levels
        """)
        
        # Create and display Sankey diagram
        sankey_fig = self.create_activity_carb_sankey()
        st.plotly_chart(sankey_fig, use_container_width=True)
        
        # Visual separator
        st.divider()
        
        # 2. Activity Timing Analysis (formerly section 1)
        st.subheader("2. Activity Timing Analysis")
        
        st.info("""
        **Analysis by Total Steps Quartiles:**
        - Each line represents one meal's activity pattern
        - Lines are colored based on total 2-hour step count quartiles
        - Q1 (Light Blue): Lowest 25% of total steps
        - Q2 (Medium Blue): 25-50% of total steps
        - Q3 (Dark Blue): 50-75% of total steps
        - Q4 (Deep Blue): Highest 25% of total steps
        """)
            
        # Create and display the quartile plot
        quartile_fig = self.create_activity_timing_quartile_plot()
        st.plotly_chart(quartile_fig, use_container_width=True)
        
        # Visual separator
        st.divider()
        
        # 3. Temporal Activity Pattern (formerly section 2)
        st.subheader("3. Temporal Activity Pattern")
        temporal_fig = self.create_temporal_activity_pattern()
        st.plotly_chart(temporal_fig, use_container_width=True)
        
        # 4. Activity Distribution (formerly section 3)
        st.subheader("4. Activity Distribution")
        
        # Show criteria in a separate info box
        st.info("""
        **Active meals defined as:**
        - Total steps ≥600 in 2h window, OR
        - At least one 10-min interval with >200 steps
        """)
        
        # Show visualization description
        st.markdown("""
        The visualization shows two complementary views of post-meal physical activity distribution:

        **Top Row - Distribution by Meal Type:**
        - Stacked bars show meal counts for each activity range
        - Colors indicate different meal types (Breakfast, Lunch, Dinner)
        - Steps are grouped in 1,000-step intervals
        - Flights are grouped in 5-flight intervals

        **Bottom Row - Overall Distribution:**
        - Box plots show the statistical distribution of all active meals
        - Box shows quartiles (25th, 50th, 75th percentile)
        - Whiskers extend to most extreme values within 1.5 IQR
        - Points indicate outliers
        """)

        # Add new section
        st.subheader("5. Glucose Response by Carb and Activity Level")
        st.markdown("""
            This visualization shows the average glucose response and activity patterns for different 
            combinations of carbohydrate intake and activity levels:
            - Each row represents a carb level (Low/Moderate/High)
            - Each column represents an activity quartile (Q1-Q4, least to most active)
            - Blue line shows glucose change from baseline
            - Gray bars show average steps in 10-minute intervals
            - Shaded areas represent 95% confidence intervals
        """)
        
        grid_fig = self.create_glucose_activity_grid()
        st.plotly_chart(grid_fig, use_container_width=True)
        
        dist_fig = self.create_activity_distribution_plot()
        st.plotly_chart(dist_fig, use_container_width=True)
        
        #Add statistical analysis section
        self.render_statistical_analysis()