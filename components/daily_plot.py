import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz

# Setting the timezone to KST
KST = pytz.FixedOffset(540)

class DailyPlot:
    def __init__(self):
        # Load and prepare data
        self.glucose_df = pd.read_csv('data/processed_glucose_data.csv')
        self.activity_df = pd.read_csv('data/aggregated_activity_data.csv')
        self.meal_df = pd.read_csv('data/processed_meal_data.csv')
        
        # Prepare dataframes
        self._prepare_dataframes()
        
        # Get available dates
        self.available_dates = self.glucose_df['DateTime'].dt.date.unique()
        self.available_dates = sorted(self.available_dates)

        # Create a DataFrame with dates and their first measurement numbers
        daily_measurements = (self.glucose_df
            .groupby(self.glucose_df['DateTime'].dt.date)
            .agg({'MeasurementNumber': 'first'})
            .reset_index())
        
        # Store as list of tuples (date, measurement_number) for the dropdown
        self.date_options = [(row['DateTime'], row['MeasurementNumber']) 
                            for _, row in daily_measurements.iterrows()]
        self.date_options.sort()  # Sort by date


    def _prepare_dataframes(self):
        """Prepare dataframes for visualization"""
        # Convert datetime for glucose
        self.glucose_df['DateTime'] = pd.to_datetime(self.glucose_df['DateTime'])
        
        # Clean and convert activity data
        self.activity_df['start_time'] = pd.to_datetime(self.activity_df['start_time'])
        invalid_mask = self.activity_df['end_time'] == '0'
        if invalid_mask.any():
            self.activity_df.loc[invalid_mask, 'end_time'] = self.activity_df.loc[invalid_mask, 'start_time'].apply(
                lambda x: (x + pd.Timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S+09:00')
            )
        self.activity_df['end_time'] = pd.to_datetime(self.activity_df['end_time'])
        
        # Convert meal times
        self.meal_df['meal_time'] = pd.to_datetime(self.meal_df['meal_time'])

    def create_plot(self, selected_date):
        """Create daily plot for the selected date"""
        # Convert the date to a timezone-aware timestamp
        if isinstance(selected_date, datetime.date):
            start_time = pd.Timestamp(selected_date).tz_localize(KST)
        else:
            start_time = pd.Timestamp(selected_date).tz_convert(KST)
        
        end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        
        # Filter data for selected date
        day_glucose = self.glucose_df[
            (self.glucose_df['DateTime'] >= start_time) & 
            (self.glucose_df['DateTime'] <= end_time)
        ].copy()
        
        day_activity = self.activity_df[
            (self.activity_df['start_time'] >= start_time) & 
            (self.activity_df['start_time'] <= end_time)
        ].copy()
        
        day_meals = self.meal_df[
            (self.meal_df['meal_time'] >= start_time) & 
            (self.meal_df['meal_time'] <= end_time)
        ].copy()
        
        # Check if we have data for this day
        if len(day_glucose) == 0:
            st.warning(f"No glucose data available for {start_time.date()}")
            return None
        
        # Calculate daily average glucose
        daily_avg_glucose = day_glucose['GlucoseValue'].mean()
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add glucose threshold line
        fig.add_trace(
            go.Scatter(
                x=[start_time, end_time],
                y=[140, 140],
                name='Threshold (140 mg/dL)',
                line=dict(color='rgba(255,0,0,0.3)', dash='dash'),
                hovertemplate='Threshold: 140 mg/dL<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add glucose line
        fig.add_trace(
            go.Scatter(
                x=day_glucose['DateTime'],
                y=day_glucose['GlucoseValue'],
                name='Glucose',
                line=dict(color='blue', width=2),
                hovertemplate='Time: %{x|%H:%M}<br>Glucose: %{y:.1f} mg/dL<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add daily average line
        fig.add_trace(
            go.Scatter(
                x=[start_time, end_time],
                y=[daily_avg_glucose, daily_avg_glucose],
                name=f'Daily Average ({daily_avg_glucose:.1f} mg/dL)',
                line=dict(color='rgba(0,0,255,0.3)', dash='dash'),
                hovertemplate='Daily Average: %{y:.1f} mg/dL<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add step count bars
        if len(day_activity) > 0:
            fig.add_trace(
                go.Bar(
                    x=day_activity['start_time'],
                    y=day_activity['steps'],
                    name='Steps',
                    marker_color='rgba(128,128,128,0.3)',
                    hovertemplate='Time: %{x|%H:%M}<br>Steps: %{y}<extra></extra>'
                ),
                secondary_y=True
            )
        
        # Add meal markers
        for _, meal in day_meals.iterrows():
            is_main_meal = meal['meal_type'] in ['Breakfast', 'Lunch', 'Dinner']
            
            closest_glucose = day_glucose[day_glucose['DateTime'] <= meal['meal_time']]
            if not closest_glucose.empty:
                meal_glucose = closest_glucose['GlucoseValue'].iloc[-1]
                
                fig.add_trace(
                    go.Scatter(
                        x=[meal['meal_time']],
                        y=[meal_glucose],
                        mode='markers+text',
                        name=meal['meal_type'],
                        marker=dict(
                            symbol='star' if is_main_meal else 'circle',
                            size=12 if is_main_meal else 8,
                            color='red' if is_main_meal else 'orange'
                        ),
                        text=[meal['meal_type']],
                        textposition='top center',
                        customdata=[[
                            meal['meal_type'],
                            meal['food_name'],
                            meal['carbohydrates']
                        ]],
                        hovertemplate=(
                            "Meal: %{customdata[0]}<br>" +
                            "Food: %{customdata[1]}<br>" +
                            "Carbs: %{customdata[2]:.1f}g<br>" +
                            "Time: %{x|%H:%M}<br>" +
                            "Glucose: %{y:.1f} mg/dL" +
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ),
                    secondary_y=False
                )
        
        # Update layout
        fig.update_layout(
            title=f'Daily Glucose and Activity - {start_time.date()}',
            xaxis=dict(
                title='Time',
                tickformat='%H:%M',
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                range=[start_time, end_time]
            ),
            yaxis=dict(
                title='Glucose (mg/dL)',
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                range=[50, 200],
                tickmode='linear',
                tick0=50,
                dtick=25
            ),
            yaxis2=dict(
                title='Steps per 10min',
                showgrid=False,
                range=[0, 1700],
                tickmode='linear',
                tick0=0,
                dtick=200,
                fixedrange=True
            ),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=600
        )
        
        return fig

    def _calculate_daily_stats(self, day_glucose, day_activity):
        """Calculate daily statistics for glucose and activity"""
        stats = {}
        
        # Glucose statistics
        if not day_glucose.empty:
            # Average glucose
            stats['avg_glucose'] = day_glucose['GlucoseValue'].mean()
            
            # Time in range (under 140)
            total_readings = len(day_glucose)
            readings_in_range = len(day_glucose[day_glucose['GlucoseValue'] < 140])
            stats['time_in_range'] = (readings_in_range / total_readings) * 100 if total_readings > 0 else 0
            
            # Glucose variability (coefficient of variation)
            stats['glucose_variability'] = (day_glucose['GlucoseValue'].std() / 
                                          day_glucose['GlucoseValue'].mean()) * 100 if not day_glucose.empty else 0
        
        # Activity statistics
        if not day_activity.empty:
            stats['total_steps'] = day_activity['steps'].sum()
        else:
            stats['total_steps'] = 0
            
        return stats

    def render(self):
            """Render the daily plot component in Streamlit"""
            # Date selection code remains the same...
            selected_date = st.selectbox(
                "Select Date",
                options=[date for date, _ in self.date_options],
                format_func=lambda x: f"{x.strftime('%Y-%m-%d')} (Measurement Number: {dict(self.date_options)[x]})",
            )
            
            # Filter data for selected date
            if isinstance(selected_date, datetime.date):
                start_time = pd.Timestamp(selected_date).tz_localize(KST)
            else:
                start_time = pd.Timestamp(selected_date).tz_convert(KST)
            
            end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            
            day_glucose = self.glucose_df[
                (self.glucose_df['DateTime'] >= start_time) & 
                (self.glucose_df['DateTime'] <= end_time)
            ].copy()
            
            day_activity = self.activity_df[
                (self.activity_df['start_time'] >= start_time) & 
                (self.activity_df['start_time'] <= end_time)
            ].copy()

            # Create and display plot
            fig = self.create_plot(selected_date)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display daily statistics
                stats = self._calculate_daily_stats(day_glucose, day_activity)
                
                # Create columns for statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Average Glucose",
                        value=f"{stats['avg_glucose']:.1f} mg/dL"
                    )
                
                with col2:
                    st.metric(
                        label="Time in Range (<140)",
                        value=f"{stats['time_in_range']:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        label="Glucose Variability",
                        value=f"{stats['glucose_variability']:.1f}%"
                    )
                
                with col4:
                    st.metric(
                        label="Total Steps",
                        value=f"{stats['total_steps']:,}"
                    )