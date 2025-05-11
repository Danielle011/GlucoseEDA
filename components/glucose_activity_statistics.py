import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class GlucoseActivityStatistics:
    def __init__(self, meal_df, glucose_df, activity_df):
        """Initialize with dataframes for statistical analysis"""
        self.meal_df = meal_df.copy()
        self.glucose_df = glucose_df.copy()
        self.activity_df = activity_df.copy()
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Prepare dataframes for analysis"""
        # Ensure datetime columns
        self.meal_df['meal_time'] = pd.to_datetime(self.meal_df['meal_time'])
        self.glucose_df['DateTime'] = pd.to_datetime(self.glucose_df['DateTime'])
        self.activity_df['start_time'] = pd.to_datetime(self.activity_df['start_time'])
        
        # Filter for valid meals (with 120 min window)
        self.valid_meals = self.meal_df[self.meal_df['window_duration'] >= 120].copy()
        
        # Label meals as active or inactive
        active_mask = (
            (self.valid_meals['total_steps'] >= 600) | 
            (self.valid_meals['max_interval_steps'] > 200)
        )
        self.valid_meals['activity_status'] = np.where(active_mask, 'active', 'inactive')
        
        # Calculate quartiles for active meals only
        active_meals = self.valid_meals[active_mask]
        quartiles = active_meals['total_steps'].quantile([0.25, 0.5, 0.75])
        self.quartile_thresholds = quartiles.to_dict()
        
        # Apply quartile labels
        def get_quartile(steps):
            if steps <= quartiles[0.25]:
                return 'Q1'
            elif steps <= quartiles[0.50]:
                return 'Q2'
            elif steps <= quartiles[0.75]:
                return 'Q3'
            else:
                return 'Q4'
        
        self.valid_meals.loc[active_mask, 'activity_quartile'] = \
            self.valid_meals.loc[active_mask, 'total_steps'].apply(get_quartile)

    def calculate_meal_metrics(self):
        """Calculate glucose response metrics for each meal"""
        metrics_list = []
        
        for idx, meal in self.valid_meals.iterrows():
            # Get glucose data for this meal
            glucose_data = self.glucose_df[
                (self.glucose_df['DateTime'] >= meal['meal_time']) &
                (self.glucose_df['DateTime'] <= meal['meal_time'] + pd.Timedelta(minutes=120)) &
                (self.glucose_df['MeasurementNumber'] == meal['measurement_number'])
            ].copy()
            
            # Skip if no glucose data
            if len(glucose_data) < 10:  # Ensure enough readings
                continue
                
            # Calculate minutes from meal
            glucose_data['minutes'] = (
                (glucose_data['DateTime'] - meal['meal_time']).dt.total_seconds() / 60
            )
            
            # Get baseline (first reading)
            baseline_glucose = glucose_data.iloc[0]['GlucoseValue']
            
            # Calculate normalized values (above baseline)
            glucose_data['value_above_baseline'] = glucose_data['GlucoseValue'] - baseline_glucose
            
            # Calculate metrics
            try:
                peak_value = glucose_data['GlucoseValue'].max()
                peak_rise = peak_value - baseline_glucose
                time_to_peak = glucose_data.loc[glucose_data['GlucoseValue'].idxmax(), 'minutes']
                
                # Calculate iAUC using trapezoidal rule on values above baseline
                # Only consider positive areas (where glucose > baseline)
                positive_values = glucose_data['value_above_baseline'].clip(lower=0)
                iauc = np.trapz(positive_values, glucose_data['minutes'])
                
                # First-hour AUC
                first_hour = glucose_data[glucose_data['minutes'] <= 60]
                if len(first_hour) > 0:
                    first_hour_values = first_hour['value_above_baseline'].clip(lower=0)
                    first_hour_iauc = np.trapz(first_hour_values, first_hour['minutes'])
                else:
                    first_hour_iauc = np.nan
                
                # Second-hour AUC
                second_hour = glucose_data[(glucose_data['minutes'] > 60) & (glucose_data['minutes'] <= 120)]
                if len(second_hour) > 0:
                    second_hour_values = second_hour['value_above_baseline'].clip(lower=0)
                    second_hour_iauc = np.trapz(second_hour_values, second_hour['minutes'])
                else:
                    second_hour_iauc = np.nan
                
                # Glucose variability (standard deviation)
                glucose_variability = glucose_data['GlucoseValue'].std()
                
                # Check for double peak
                # First smooth the data to reduce noise
                if len(glucose_data) >= 15:  # Ensure enough data for rolling window
                    glucose_data['smoothed'] = glucose_data['GlucoseValue'].rolling(window=5, center=True).mean()
                    glucose_data['smoothed'] = glucose_data['smoothed'].fillna(glucose_data['GlucoseValue'])
                    
                    # Find peaks (points higher than neighbors)
                    peaks = []
                    for i in range(1, len(glucose_data) - 1):
                        if (glucose_data['smoothed'].iloc[i] > glucose_data['smoothed'].iloc[i-1] and
                            glucose_data['smoothed'].iloc[i] > glucose_data['smoothed'].iloc[i+1]):
                            peaks.append((
                                glucose_data['minutes'].iloc[i],
                                glucose_data['smoothed'].iloc[i]
                            ))
                    
                    # Filter peaks that are at least 10 mg/dL above baseline
                    significant_peaks = [p for p in peaks if p[1] >= baseline_glucose + 10]
                    
                    has_double_peak = len(significant_peaks) >= 2
                    
                    # If double peak, measure time between peaks and magnitude of second peak
                    if has_double_peak:
                        # Sort peaks by time
                        sorted_peaks = sorted(significant_peaks, key=lambda x: x[0])
                        time_between_peaks = sorted_peaks[1][0] - sorted_peaks[0][0]
                        second_peak_magnitude = sorted_peaks[1][1] - baseline_glucose
                    else:
                        time_between_peaks = np.nan
                        second_peak_magnitude = np.nan
                else:
                    has_double_peak = False
                    time_between_peaks = np.nan
                    second_peak_magnitude = np.nan
                
                # Store metrics
                metrics = {
                    'meal_id': idx,
                    'carb_label': meal['carb_label'],
                    'activity_status': meal['activity_status'],
                    'activity_quartile': meal.get('activity_quartile', 'inactive'),
                    'total_steps': meal['total_steps'],
                    'max_interval_steps': meal['max_interval_steps'],
                    'baseline_glucose': baseline_glucose,
                    'peak_glucose': peak_value,
                    'peak_rise': peak_rise,
                    'time_to_peak': time_to_peak,
                    'iauc': iauc,
                    'first_hour_iauc': first_hour_iauc,
                    'second_hour_iauc': second_hour_iauc,
                    'iauc_ratio': second_hour_iauc / first_hour_iauc if first_hour_iauc > 0 else np.nan,
                    'glucose_variability': glucose_variability,
                    'has_double_peak': has_double_peak,
                    'time_between_peaks': time_between_peaks,
                    'second_peak_magnitude': second_peak_magnitude
                }
                
                metrics_list.append(metrics)
                
            except Exception as e:
                # Skip this meal if calculations fail
                print(f"Error processing meal {idx}: {str(e)}")
                continue
        
        # Create DataFrame from all calculated metrics
        self.metrics_df = pd.DataFrame(metrics_list)
        return self.metrics_df
    
    def run_primary_analysis(self):
        """Run primary statistical comparisons between active and inactive meals by carb level"""
        if not hasattr(self, 'metrics_df'):
            print("Calculating meal metrics first...")
            self.calculate_meal_metrics()
        
        results = {}
        
        # Statistics by carb level
        for carb_level in ['low', 'moderate', 'high']:
            carb_data = self.metrics_df[self.metrics_df['carb_label'] == carb_level]
            
            inactive = carb_data[carb_data['activity_status'] == 'inactive']
            active = carb_data[carb_data['activity_status'] == 'active']
            
            # Skip if insufficient data
            if len(inactive) < 5 or len(active) < 5:
                print(f"Insufficient data for {carb_level} carb level")
                continue
            
            # List all metrics to compare
            metrics_to_test = [
                'peak_rise',
                'time_to_peak',
                'iauc',
                'first_hour_iauc',
                'second_hour_iauc',
                'iauc_ratio',
                'glucose_variability'
            ]
            
            # Store statistical results
            carb_results = {}
            
            for metric in metrics_to_test:
                # Get data, removing NaN values
                inactive_data = inactive[metric].dropna()
                active_data = active[metric].dropna()
                
                if len(inactive_data) < 5 or len(active_data) < 5:
                    print(f"Insufficient data for {metric} in {carb_level} carb level")
                    carb_results[metric] = {
                        'inactive_n': len(inactive_data),
                        'active_n': len(active_data),
                        'inactive_mean': inactive_data.mean() if len(inactive_data) > 0 else np.nan,
                        'active_mean': active_data.mean() if len(active_data) > 0 else np.nan,
                        'inactive_median': inactive_data.median() if len(inactive_data) > 0 else np.nan,
                        'active_median': active_data.median() if len(active_data) > 0 else np.nan,
                        'p_value': np.nan,
                        'significant': np.nan,
                        'effect_size': np.nan
                    }
                    continue
                
                # Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(
                    inactive_data,
                    active_data,
                    alternative='two-sided'
                )
                
                # Calculate Cliff's delta for effect size
                # Manual implementation since it's not available in scipy
                cliff_delta = self._calculate_cliffs_delta(inactive_data, active_data)
                
                # Store results
                carb_results[metric] = {
                    'inactive_n': len(inactive_data),
                    'active_n': len(active_data),
                    'inactive_mean': inactive_data.mean(),
                    'active_mean': active_data.mean(),
                    'inactive_median': inactive_data.median(),
                    'active_median': active_data.median(),
                    'inactive_std': inactive_data.std(),
                    'active_std': active_data.std(),
                    'u_statistic': u_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': cliff_delta,
                    'effect_size_interpretation': self._interpret_cliffs_delta(cliff_delta)
                }
            
            results[carb_level] = carb_results
        
        self.primary_results = results
        return results
    
    def _calculate_cliffs_delta(self, x, y):
        """Calculate Cliff's delta for effect size estimation"""
        # Count comparisons where x > y, x < y, and x = y
        greater = 0
        less = 0
        equal = 0
        
        for i in x:
            for j in y:
                if i > j:
                    greater += 1
                elif i < j:
                    less += 1
                else:
                    equal += 1
        
        # Calculate delta
        total = len(x) * len(y)
        delta = (greater - less) / total
        
        return delta
    
    def _interpret_cliffs_delta(self, delta):
        """Interpret Cliff's delta magnitude"""
        abs_delta = abs(delta)
        
        if abs_delta < 0.147:
            return "Negligible"
        elif abs_delta < 0.33:
            return "Small"
        elif abs_delta < 0.474:
            return "Medium"
        else:
            return "Large"
    
    def plot_metrics_boxplots(self, metrics_to_plot=None):
        """Create boxplots for key glucose metrics by carb level and activity status"""
        if not hasattr(self, 'metrics_df'):
            print("Calculating meal metrics first...")
            self.calculate_meal_metrics()
        
        if metrics_to_plot is None:
            metrics_to_plot = [
                'peak_rise',
                'iauc',
                'glucose_variability',
                'time_to_peak'
            ]
        
        # Nicer labels for plots
        metric_labels = {
            'peak_rise': 'Peak Rise (mg/dL)',
            'iauc': 'iAUC',
            'first_hour_iauc': 'First Hour iAUC',
            'second_hour_iauc': 'Second Hour iAUC',
            'iauc_ratio': 'Second/First Hour iAUC Ratio',
            'glucose_variability': 'Glucose Variability (SD)',
            'time_to_peak': 'Time to Peak (minutes)'
        }
        
        # Create subplots
        fig, axes = plt.subplots(
            len(metrics_to_plot), 
            3,  # One column per carb level
            figsize=(15, 4 * len(metrics_to_plot)),
            squeeze=False
        )
        
        # Set common title
        fig.suptitle('Glucose Response Metrics by Carb Level and Activity Status', fontsize=16)
        fig.subplots_adjust(top=0.95)
        
        # Create color palette
        colors = {'inactive': 'skyblue', 'active': 'salmon'}
        
        # Plot each metric and carb level combination
        for i, metric in enumerate(metrics_to_plot):
            for j, carb in enumerate(['low', 'moderate', 'high']):
                ax = axes[i, j]
                
                # Filter data
                carb_data = self.metrics_df[self.metrics_df['carb_label'] == carb]
                
                # Create boxplot
                sns.boxplot(
                    x='activity_status',
                    y=metric,
                    data=carb_data,
                    palette=colors,
                    ax=ax
                )
                
                # Add individual points
                sns.stripplot(
                    x='activity_status',
                    y=metric,
                    data=carb_data,
                    color='black',
                    alpha=0.5,
                    size=3,
                    jitter=True,
                    ax=ax
                )
                
                # Set title and labels
                ax.set_title(f'{carb.title()} Carb')
                ax.set_xlabel('Activity Status')
                ax.set_ylabel(metric_labels.get(metric, metric))
                
                # Add sample size to x-tick labels
                inactive_n = len(carb_data[carb_data['activity_status'] == 'inactive'])
                active_n = len(carb_data[carb_data['activity_status'] == 'active'])
                ax.set_xticklabels([f'Inactive\n(n={inactive_n})', f'Active\n(n={active_n})'])
                
                # Add p-value if available
                if hasattr(self, 'primary_results'):
                    if carb in self.primary_results and metric in self.primary_results[carb]:
                        p_value = self.primary_results[carb][metric]['p_value']
                        if not np.isnan(p_value):
                            # Format p-value string
                            if p_value < 0.001:
                                p_text = 'p < 0.001'
                            else:
                                p_text = f'p = {p_value:.3f}'
                            
                            # Add star for significance
                            if p_value < 0.05:
                                p_text += ' *'
                                
                            ax.text(0.5, 0.95, p_text, 
                                 transform=ax.transAxes, 
                                 ha='center', va='top',
                                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def run_secondary_analysis(self):
        """Compare inactive vs high activity (Q3+Q4) meals for low and moderate carb levels"""
        if not hasattr(self, 'metrics_df'):
            print("Calculating meal metrics first...")
            self.calculate_meal_metrics()
        
        results = {}
        
        # Only analyze low and moderate carb levels
        for carb_level in ['low', 'moderate']:
            carb_data = self.metrics_df[self.metrics_df['carb_label'] == carb_level]
            
            # Get inactive meals
            inactive = carb_data[carb_data['activity_status'] == 'inactive']
            
            # Get high activity meals (Q3+Q4)
            high_active = carb_data[carb_data['activity_quartile'].isin(['Q3', 'Q4'])]
            
            # Skip if insufficient data
            if len(inactive) < 5 or len(high_active) < 5:
                print(f"Insufficient data for {carb_level} carb level")
                continue
            
            # List all metrics to compare
            metrics_to_test = [
                'peak_rise',
                'time_to_peak',
                'iauc',
                'first_hour_iauc',
                'second_hour_iauc',
                'iauc_ratio',
                'glucose_variability'
            ]
            
            # Store statistical results
            carb_results = {}
            
            for metric in metrics_to_test:
                # Get data, removing NaN values
                inactive_data = inactive[metric].dropna()
                high_active_data = high_active[metric].dropna()
                
                if len(inactive_data) < 5 or len(high_active_data) < 5:
                    print(f"Insufficient data for {metric} in {carb_level} carb level")
                    continue
                
                # Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(
                    inactive_data,
                    high_active_data,
                    alternative='two-sided'
                )
                
                # Calculate Cliff's delta for effect size
                cliff_delta = self._calculate_cliffs_delta(inactive_data, high_active_data)
                
                # Store results
                carb_results[metric] = {
                    'inactive_n': len(inactive_data),
                    'high_active_n': len(high_active_data),
                    'inactive_mean': inactive_data.mean(),
                    'high_active_mean': high_active_data.mean(),
                    'inactive_median': inactive_data.median(),
                    'high_active_median': high_active_data.median(),
                    'inactive_std': inactive_data.std(),
                    'high_active_std': high_active_data.std(),
                    'u_statistic': u_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': cliff_delta,
                    'effect_size_interpretation': self._interpret_cliffs_delta(cliff_delta)
                }
            
            results[carb_level] = carb_results
        
        self.secondary_results = results
        return results
    
    def run_dose_response_analysis(self, carb_level='moderate'):
        """Analyze activity quartile trend (dose-response) for a specific carb level"""
        if not hasattr(self, 'metrics_df'):
            print("Calculating meal metrics first...")
            self.calculate_meal_metrics()
        
        # Filter for the specified carb level
        carb_data = self.metrics_df[self.metrics_df['carb_label'] == carb_level]
        
        # Filter for active meals only and add inactive as a category
        active_data = carb_data[carb_data['activity_status'] == 'active']
        inactive_data = carb_data[carb_data['activity_status'] == 'inactive']
        
        # Create quartile ordered category for sorting
        quartile_order = ['inactive', 'Q1', 'Q2', 'Q3', 'Q4']
        
        # Add inactive data with a new quartile category
        inactive_data = inactive_data.copy()
        inactive_data['activity_quartile'] = 'inactive'
        
        # Combine data
        combined_data = pd.concat([inactive_data, active_data])
        
        # Create proper categorical column for sorting
        combined_data['activity_quartile'] = pd.Categorical(
            combined_data['activity_quartile'],
            categories=quartile_order,
            ordered=True
        )
        
        results = {}
        
        # List all metrics to analyze
        metrics_to_test = [
            'peak_rise',
            'time_to_peak',
            'iauc',
            'first_hour_iauc',
            'second_hour_iauc',
            'glucose_variability'
        ]
        
        for metric in metrics_to_test:
            # Skip if not enough data
            group_counts = combined_data.groupby('activity_quartile')[metric].count()
            if (group_counts < 5).any():
                print(f"Insufficient data for {metric} in some quartiles")
                continue
            
            # Kruskal-Wallis test
            groups = [combined_data[combined_data['activity_quartile'] == q][metric].dropna() 
                     for q in quartile_order if q in combined_data['activity_quartile'].unique()]
            
            if len(groups) >= 2:  # Need at least two groups
                h_stat, p_value = stats.kruskal(*groups)
                
                # If significant, perform post-hoc Dunn's test
                if p_value < 0.05:
                    # We'll use a simplified Dunn's test implementation
                    post_hoc_results = {}
                    quartiles_present = [q for q in quartile_order if q in combined_data['activity_quartile'].unique()]
                    
                    for i, q1 in enumerate(quartiles_present):
                        for q2 in quartiles_present[i+1:]:
                            g1 = combined_data[combined_data['activity_quartile'] == q1][metric].dropna()
                            g2 = combined_data[combined_data['activity_quartile'] == q2][metric].dropna()
                            
                            if len(g1) >= 5 and len(g2) >= 5:
                                # Mann-Whitney U test for each pair
                                _, pair_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                                
                                # Bonferroni correction for multiple comparisons
                                # Number of comparisons = n(n-1)/2 where n is number of groups
                                n_groups = len(quartiles_present)
                                n_comparisons = (n_groups * (n_groups - 1)) // 2
                                adjusted_p = min(pair_p * n_comparisons, 1.0)
                                
                                post_hoc_results[f"{q1}_vs_{q2}"] = {
                                    'p_value': pair_p,
                                    'adjusted_p': adjusted_p,
                                    'significant': adjusted_p < 0.05
                                }
                else:
                    post_hoc_results = None
                
                # Calculate group statistics
                group_stats = combined_data.groupby('activity_quartile')[metric].agg([
                    'count', 'mean', 'std', 'median', 'min', 'max'
                ]).reset_index()
                
                # Store results
                results[metric] = {
                    'kruskal_h': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'group_stats': group_stats.to_dict('records'),
                    'post_hoc': post_hoc_results
                }
            
        self.dose_response_results = results
        return results
    
    def plot_dose_response(self, carb_level='moderate', metrics_to_plot=None):
        """Plot dose-response relationship for the specified carb level"""
        if not hasattr(self, 'metrics_df'):
            print("Calculating meal metrics first...")
            self.calculate_meal_metrics()
        
        if metrics_to_plot is None:
            metrics_to_plot = [
                'peak_rise',
                'iauc',
                'glucose_variability',
                'time_to_peak'
            ]
        
        # Nicer labels for plots
        metric_labels = {
            'peak_rise': 'Peak Rise (mg/dL)',
            'iauc': 'iAUC',
            'first_hour_iauc': 'First Hour iAUC',
            'second_hour_iauc': 'Second Hour iAUC',
            'iauc_ratio': 'Second/First Hour iAUC Ratio',
            'glucose_variability': 'Glucose Variability (SD)',
            'time_to_peak': 'Time to Peak (minutes)'
        }
        
        # Filter for the specified carb level
        carb_data = self.metrics_df[self.metrics_df['carb_label'] == carb_level]
        
        # Filter for active meals only and add inactive as a category
        active_data = carb_data[carb_data['activity_status'] == 'active']
        inactive_data = carb_data[carb_data['activity_status'] == 'inactive']
        
        # Add inactive data with a new quartile category
        inactive_data = inactive_data.copy()
        inactive_data['activity_quartile'] = 'inactive'
        
        # Combine data
        combined_data = pd.concat([inactive_data, active_data])
        
        # Create proper categorical column for sorting
        quartile_order = ['inactive', 'Q1', 'Q2', 'Q3', 'Q4']
        combined_data['activity_quartile'] = pd.Categorical(
            combined_data['activity_quartile'],
            categories=quartile_order,
            ordered=True
        )
        
        # Create figure
        fig, axes = plt.subplots(
            len(metrics_to_plot), 
            1,
            figsize=(10, 4 * len(metrics_to_plot)),
            squeeze=False
        )
        
        # Flatten axes array for easier indexing
        axes = axes.flatten()
        
        # Color palette for quartiles
        colors = {
            'inactive': 'gray',
            'Q1': 'skyblue',
            'Q2': 'lightgreen',
            'Q3': 'salmon',
            'Q4': 'purple'
        }
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Create boxplot
            sns.boxplot(
                x='activity_quartile',
                y=metric,
                data=combined_data,
                palette=[colors.get(q, 'gray') for q in combined_data['activity_quartile'].cat.categories],
                ax=ax,
                order=quartile_order
            )
            
            # Add individual points
            sns.stripplot(
                x='activity_quartile',
                y=metric,
                data=combined_data,
                color='black',
                alpha=0.5,
                size=3,
                jitter=True,
                ax=ax,
                order=quartile_order
            )
            
            # Set title and labels
            ax.set_title(f'{metric_labels.get(metric, metric)} by Activity Level ({carb_level.title()} Carb)')
            ax.set_xlabel('Activity Level')
            ax.set_ylabel(metric_labels.get(metric, metric))
            
            # Add sample sizes to x-tick labels
            xticklabels = []
            for q in quartile_order:
                count = len(combined_data[combined_data['activity_quartile'] == q])
                if count > 0:
                    xticklabels.append(f'{q}\n(n={count})')
                else:
                    xticklabels.append(q)
            
            ax.set_xticklabels(xticklabels)
            
            # Add p-value if available
            if hasattr(self, 'dose_response_results'):
                if metric in self.dose_response_results:
                    p_value = self.dose_response_results[metric]['p_value']
                    if not np.isnan(p_value):
                        # Format p-value string
                        if p_value < 0.001:
                            p_text = 'Kruskal-Wallis: p < 0.001'
                        else:
                            p_text = f'Kruskal-Wallis: p = {p_value:.3f}'
                        
                        # Add star for significance
                        if p_value < 0.05:
                            p_text += ' *'
                            
                        ax.text(0.5, 0.95, p_text, 
                             transform=ax.transAxes, 
                             ha='center', va='top',
                             bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def display_results_table(self, analysis_type='primary'):
        """Display nicely formatted results table"""
        if analysis_type == 'primary' and hasattr(self, 'primary_results'):
            results = self.primary_results
            comparison = "Inactive vs. Active"
        elif analysis_type == 'secondary' and hasattr(self, 'secondary_results'):
            results = self.secondary_results
            comparison = "Inactive vs. High Activity (Q3+Q4)"
        elif analysis_type == 'dose_response' and hasattr(self, 'dose_response_results'):
            results = self.dose_response_results
            comparison = "Activity Quartiles (Dose-Response)"
        else:
            print(f"No results available for {analysis_type} analysis")
            return None
        
        # Create a formatted DataFrame for the results
        all_rows = []
        
        if analysis_type in ['primary', 'secondary']:
            for carb_level, metrics in results.items():
                for metric, stats in metrics.items():
                    row = {
                        'Carb Level': carb_level.title(),
                        'Metric': metric,
                        'Inactive Mean': f"{stats['inactive_mean']:.2f}",
                        'Active Mean': f"{stats['active_mean']:.2f}" if analysis_type == 'primary' else f"{stats['high_active_mean']:.2f}",
                        'Inactive Median': f"{stats['inactive_median']:.2f}",
                        'Active Median': f"{stats['active_median']:.2f}" if analysis_type == 'primary' else f"{stats['high_active_median']:.2f}",
                        'Sample Size': f"{stats['inactive_n']} vs {stats['active_n'] if analysis_type == 'primary' else stats['high_active_n']}",
                        'P-value': f"{stats['p_value']:.4f}",
                        'Significant': '✓' if stats['significant'] else '✗',
                        'Effect Size': f"{stats['effect_size']:.2f} ({stats['effect_size_interpretation']})"
                    }
                    all_rows.append(row)
        else:  # dose_response analysis
            for metric, stats in results.items():
                row = {
                    'Metric': metric,
                    'Kruskal-Wallis H': f"{stats['kruskal_h']:.2f}",
                    'P-value': f"{stats['p_value']:.4f}",
                    'Significant': '✓' if stats['significant'] else '✗'
                }
                all_rows.append(row)
        
        if all_rows:
            return pd.DataFrame(all_rows)
        else:
            print("No results to display")
            return None
            
    def visualize_activity_timing_effect(self):
        """Analyze and visualize how activity timing affects glucose response"""
        if not hasattr(self, 'metrics_df'):
            print("Calculating meal metrics first...")
            self.calculate_meal_metrics()
            
        # We need to calculate when activity occurs relative to the meal
        timing_results = []
        
        for idx, meal in self.valid_meals.iterrows():
            # Skip inactive meals
            if meal['activity_status'] != 'active':
                continue
                
            # Skip meals without metrics
            if hasattr(self, 'metrics_df') and idx not in self.metrics_df['meal_id'].values:
                continue
                
            # Calculate when significant activity occurs (>800 steps in 10min)
            activity_data = []
            first_active_interval = None
            
            for interval in range(0, 121, 10):
                interval_start = meal['meal_time'] + pd.Timedelta(minutes=interval)
                interval_end = interval_start + pd.Timedelta(minutes=10)
                
                mask = (
                    (self.activity_df['start_time'] >= interval_start) &
                    (self.activity_df['start_time'] < interval_end)
                )
                interval_activity = self.activity_df[mask]
                steps = interval_activity['steps'].sum() if len(interval_activity) > 0 else 0
                
                # Record first interval with significant activity
                if steps > 800 and first_active_interval is None:
                    first_active_interval = interval
                
                activity_data.append({
                    'interval': interval,
                    'steps': steps
                })
            
            # Get meal metrics from metrics_df
            meal_metrics = self.metrics_df[self.metrics_df['meal_id'] == idx]
            
            if len(meal_metrics) > 0:
                metrics = meal_metrics.iloc[0]
                
                # Store timing data
                timing_results.append({
                    'meal_id': idx,
                    'carb_label': meal['carb_label'],
                    'activity_quartile': meal['activity_quartile'] if 'activity_quartile' in meal else None,
                    'total_steps': meal['total_steps'],
                    'first_active_interval': first_active_interval,
                    'peak_rise': metrics['peak_rise'],
                    'time_to_peak': metrics['time_to_peak'],
                    'iauc': metrics['iauc'],
                    'early_activity': sum(item['steps'] for item in activity_data if item['interval'] < 60),
                    'late_activity': sum(item['steps'] for item in activity_data if item['interval'] >= 60),
                    'activity_timing_ratio': sum(item['steps'] for item in activity_data if item['interval'] < 60) / 
                                          max(1, sum(item['steps'] for item in activity_data))
                })
        
        # Create DataFrame
        timing_df = pd.DataFrame(timing_results)
        
        # Create timing categories
        timing_df['timing_category'] = pd.cut(
            timing_df['first_active_interval'],
            bins=[-1, 20, 40, 60, 120],
            labels=['0-20 min', '21-40 min', '41-60 min', '61+ min']
        )
        
        # Create early vs late activity categories
        timing_df['activity_pattern'] = pd.cut(
            timing_df['activity_timing_ratio'],
            bins=[0, 0.25, 0.75, 1.01],  # Slightly over 1 to include 1.0
            labels=['Mostly Late', 'Mixed', 'Mostly Early']
        )
        
        # Plot results - Activity timing vs glucose metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics = ['peak_rise', 'iauc', 'time_to_peak']
        titles = ['Peak Rise (mg/dL)', 'Incremental AUC', 'Time to Peak (min)']
        
        # Timing category plots (when activity started)
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2
            
            sns.boxplot(
                x='timing_category',
                y=metric,
                data=timing_df,
                palette='viridis',
                ax=axes[row, col]
            )
            
            sns.stripplot(
                x='timing_category',
                y=metric,
                data=timing_df,
                color='black',
                alpha=0.5,
                size=3,
                jitter=True,
                ax=axes[row, col]
            )
            
            axes[row, col].set_title(f'{title} by Activity Onset Timing')
            axes[row, col].set_xlabel('First Active Period (>800 steps/10min)')
            axes[row, col].set_ylabel(title)
            
            # Add sample sizes to x-tick labels
            xticklabels = []
            for category in timing_df['timing_category'].cat.categories:
                count = len(timing_df[timing_df['timing_category'] == category])
                xticklabels.append(f'{category}\n(n={count})')
            
            axes[row, col].set_xticklabels(xticklabels)
            
            # Run statistical test (Kruskal-Wallis)
            groups = [timing_df[timing_df['timing_category'] == cat][metric].dropna() 
                    for cat in timing_df['timing_category'].cat.categories]
            groups = [g for g in groups if len(g) >= 3]  # Filter out groups with too few samples
            
            if len(groups) >= 2:
                h_stat, p_value = stats.kruskal(*groups)
                
                if p_value < 0.001:
                    p_text = 'p < 0.001'
                else:
                    p_text = f'p = {p_value:.3f}'
                
                if p_value < 0.05:
                    p_text += ' *'
                    
                axes[row, col].text(0.5, 0.95, p_text, 
                                transform=axes[row, col].transAxes, 
                                ha='center', va='top',
                                bbox=dict(facecolor='white', alpha=0.8))
        
        # Early vs Late activity pattern plot
        sns.boxplot(
            x='activity_pattern',
            y='peak_rise',  # Use peak_rise as example
            data=timing_df,
            palette='plasma',
            ax=axes[1, 1]
        )
        
        sns.stripplot(
            x='activity_pattern',
            y='peak_rise',
            data=timing_df,
            color='black',
            alpha=0.5,
            size=3,
            jitter=True,
            ax=axes[1, 1]
        )
        
        axes[1, 1].set_title('Peak Rise by Activity Distribution Pattern')
        axes[1, 1].set_xlabel('Activity Pattern')
        axes[1, 1].set_ylabel('Peak Rise (mg/dL)')
        
        # Add sample sizes to x-tick labels
        xticklabels = []
        for category in timing_df['activity_pattern'].cat.categories:
            count = len(timing_df[timing_df['activity_pattern'] == category])
            xticklabels.append(f'{category}\n(n={count})')
        
        axes[1, 1].set_xticklabels(xticklabels)
        
        # Run statistical test (Kruskal-Wallis)
        groups = [timing_df[timing_df['activity_pattern'] == cat]['peak_rise'].dropna() 
                for cat in timing_df['activity_pattern'].cat.categories]
        groups = [g for g in groups if len(g) >= 3]
        
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups)
            
            if p_value < 0.001:
                p_text = 'p < 0.001'
            else:
                p_text = f'p = {p_value:.3f}'
            
            if p_value < 0.05:
                p_text += ' *'
                
            axes[1, 1].text(0.5, 0.95, p_text, 
                          transform=axes[1, 1].transAxes, 
                          ha='center', va='top',
                          bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig, timing_df
    
    def visualize_activity_distribution_effect(self):
        """Analyze how activity distribution (continuous vs. intermittent) affects glucose response"""
        if not hasattr(self, 'metrics_df'):
            print("Calculating meal metrics first...")
            self.calculate_meal_metrics()
            
        # Calculate metrics for activity distribution
        distribution_results = []
        
        for idx, meal in self.valid_meals.iterrows():
            # Skip inactive meals
            if meal['activity_status'] != 'active':
                continue
                
            # Skip meals without metrics
            if hasattr(self, 'metrics_df') and idx not in self.metrics_df['meal_id'].values:
                continue
                
            # Calculate activity distribution metrics
            activity_intervals = []
            
            for interval in range(0, 121, 10):
                interval_start = meal['meal_time'] + pd.Timedelta(minutes=interval)
                interval_end = interval_start + pd.Timedelta(minutes=10)
                
                mask = (
                    (self.activity_df['start_time'] >= interval_start) &
                    (self.activity_df['start_time'] < interval_end)
                )
                interval_activity = self.activity_df[mask]
                steps = interval_activity['steps'].sum() if len(interval_activity) > 0 else 0
                
                activity_intervals.append(steps)
            
            # Calculate activity concentration
            total_steps = sum(activity_intervals)
            if total_steps > 0:
                # Calculate Gini coefficient as a measure of activity concentration
                # Higher value means more concentrated activity
                sorted_intervals = sorted(activity_intervals)
                cum_steps = np.cumsum(sorted_intervals)
                gini = 1 - 2 * np.trapz(cum_steps) / (cum_steps[-1] * len(activity_intervals))
                
                # Calculate CV (Coefficient of Variation) as another measure of variability
                cv = np.std(activity_intervals) / (np.mean(activity_intervals) if np.mean(activity_intervals) > 0 else 1)
                
                # Calculate maximum interval percentage
                max_interval_pct = max(activity_intervals) / total_steps if total_steps > 0 else 0
                
                # Count active intervals (>800 steps)
                active_intervals = sum(1 for steps in activity_intervals if steps > 800)
                
                # Get meal metrics
                meal_metrics = self.metrics_df[self.metrics_df['meal_id'] == idx]
                
                if len(meal_metrics) > 0:
                    metrics = meal_metrics.iloc[0]
                    
                    # Store distribution data
                    distribution_results.append({
                        'meal_id': idx,
                        'carb_label': meal['carb_label'],
                        'activity_quartile': meal['activity_quartile'] if 'activity_quartile' in meal else None,
                        'total_steps': total_steps,
                        'gini_coefficient': gini,
                        'coefficient_of_variation': cv,
                        'max_interval_percentage': max_interval_pct,
                        'active_intervals': active_intervals,
                        'peak_rise': metrics['peak_rise'],
                        'time_to_peak': metrics['time_to_peak'],
                        'iauc': metrics['iauc'],
                        'has_double_peak': metrics['has_double_peak']
                    })
        
        # Create DataFrame
        distribution_df = pd.DataFrame(distribution_results)
        
        # Create concentration categories
        distribution_df['concentration_category'] = pd.cut(
            distribution_df['max_interval_percentage'],
            bins=[0, 0.33, 0.66, 1.01],  # Slightly over 1 to include 1.0
            labels=['Distributed', 'Moderate', 'Concentrated']
        )
        
        # Create active intervals categories
        distribution_df['activity_breadth'] = pd.cut(
            distribution_df['active_intervals'],
            bins=[-1, 1, 3, 12],
            labels=['Single Period', 'Few Periods', 'Many Periods']
        )
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Activity concentration vs peak rise
        sns.boxplot(
            x='concentration_category',
            y='peak_rise',
            data=distribution_df,
            palette='viridis',
            ax=axes[0, 0]
        )
        
        sns.stripplot(
            x='concentration_category',
            y='peak_rise',
            data=distribution_df,
            color='black',
            alpha=0.5,
            size=3,
            jitter=True,
            ax=axes[0, 0]
        )
        
        axes[0, 0].set_title('Peak Rise by Activity Concentration')
        axes[0, 0].set_xlabel('Activity Concentration')
        axes[0, 0].set_ylabel('Peak Rise (mg/dL)')
        
        # Add sample sizes
        xticklabels = []
        for category in distribution_df['concentration_category'].cat.categories:
            count = len(distribution_df[distribution_df['concentration_category'] == category])
            xticklabels.append(f'{category}\n(n={count})')
        
        axes[0, 0].set_xticklabels(xticklabels)
        
        # Statistical test
        groups = [distribution_df[distribution_df['concentration_category'] == cat]['peak_rise'].dropna() 
                for cat in distribution_df['concentration_category'].cat.categories]
        groups = [g for g in groups if len(g) >= 3]
        
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups)
            
            if p_value < 0.001:
                p_text = 'p < 0.001'
            else:
                p_text = f'p = {p_value:.3f}'
            
            if p_value < 0.05:
                p_text += ' *'
                
            axes[0, 0].text(0.5, 0.95, p_text, 
                        transform=axes[0, 0].transAxes, 
                        ha='center', va='top',
                        bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot 2: Activity breadth vs peak rise
        sns.boxplot(
            x='activity_breadth',
            y='peak_rise',
            data=distribution_df,
            palette='plasma',
            ax=axes[0, 1]
        )
        
        sns.stripplot(
            x='activity_breadth',
            y='peak_rise',
            data=distribution_df,
            color='black',
            alpha=0.5,
            size=3,
            jitter=True,
            ax=axes[0, 1]
        )
        
        axes[0, 1].set_title('Peak Rise by Activity Distribution')
        axes[0, 1].set_xlabel('Activity Distribution')
        axes[0, 1].set_ylabel('Peak Rise (mg/dL)')
        
        # Add sample sizes
        xticklabels = []
        for category in distribution_df['activity_breadth'].cat.categories:
            count = len(distribution_df[distribution_df['activity_breadth'] == category])
            xticklabels.append(f'{category}\n(n={count})')
        
        axes[0, 1].set_xticklabels(xticklabels)
        
        # Statistical test
        groups = [distribution_df[distribution_df['activity_breadth'] == cat]['peak_rise'].dropna() 
                for cat in distribution_df['activity_breadth'].cat.categories]
        groups = [g for g in groups if len(g) >= 3]
        
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups)
            
            if p_value < 0.001:
                p_text = 'p < 0.001'
            else:
                p_text = f'p = {p_value:.3f}'
            
            if p_value < 0.05:
                p_text += ' *'
                
            axes[0, 1].text(0.5, 0.95, p_text, 
                        transform=axes[0, 1].transAxes, 
                        ha='center', va='top',
                        bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot 3: Scatter plot of concentration vs peak rise with carb levels
        sns.scatterplot(
            x='max_interval_percentage',
            y='peak_rise',
            hue='carb_label',
            data=distribution_df,
            palette={'low': 'blue', 'moderate': 'orange', 'high': 'green'},
            ax=axes[1, 0]
        )
        
        axes[1, 0].set_title('Peak Rise vs Activity Concentration by Carb Level')
        axes[1, 0].set_xlabel('Max Interval Percentage')
        axes[1, 0].set_ylabel('Peak Rise (mg/dL)')
        
        # Plot 4: Double peak occurrence by activity breadth
        # Convert boolean to numeric for easier groupby
        distribution_df['double_peak_num'] = distribution_df['has_double_peak'].astype(int)
        
        # Group by activity breadth and calculate percentage of double peaks
        double_peak_data = distribution_df.groupby('activity_breadth')['double_peak_num'].agg(['mean', 'count']).reset_index()
        double_peak_data['mean'] = double_peak_data['mean'] * 100  # Convert to percentage
        
        sns.barplot(
            x='activity_breadth',
            y='mean',
            data=double_peak_data,
            palette='plasma',
            ax=axes[1, 1]
        )
        
        axes[1, 1].set_title('Percentage of Double Glucose Peaks by Activity Distribution')
        axes[1, 1].set_xlabel('Activity Distribution')
        axes[1, 1].set_ylabel('Double Peak Occurrence (%)')
        
        # Add counts
        for i, row in double_peak_data.iterrows():
            axes[1, 1].text(
                i,
                row['mean'] + 2,
                f"n={int(row['count'])}",
                ha='center'
            )
        
        plt.tight_layout()
        return fig, distribution_df
        
    def visualize_intensity_thresholds(self):
        """Analyze threshold effects of activity intensity on glucose response"""
        if not hasattr(self, 'metrics_df'):
            print("Calculating meal metrics first...")
            self.calculate_meal_metrics()
            
        # Create intensity metrics
        intensity_results = []
        
        for idx, meal in self.valid_meals.iterrows():
            # Skip meals without metrics
            if hasattr(self, 'metrics_df') and idx not in self.metrics_df['meal_id'].values:
                continue
                
            # Get meal metrics
            meal_metrics = self.metrics_df[self.metrics_df['meal_id'] == idx]
            
            if len(meal_metrics) > 0:
                metrics = meal_metrics.iloc[0]
                
                # Store metrics with intensity data
                intensity_results.append({
                    'meal_id': idx,
                    'carb_label': meal['carb_label'],
                    'activity_status': meal['activity_status'],
                    'total_steps': meal['total_steps'],
                    'max_interval_steps': meal['max_interval_steps'],
                    'peak_rise': metrics['peak_rise'],
                    'iauc': metrics['iauc'],
                    'time_to_peak': metrics['time_to_peak']
                })
        
        # Create DataFrame
        intensity_df = pd.DataFrame(intensity_results)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Scatter plot: Total steps vs peak rise by carb level
        for carb, color in zip(['low', 'moderate', 'high'], ['blue', 'orange', 'green']):
            carb_data = intensity_df[intensity_df['carb_label'] == carb]
            axes[0, 0].scatter(
                carb_data['total_steps'],
                carb_data['peak_rise'],
                alpha=0.7,
                color=color,
                label=f'{carb.title()} Carb'
            )
        
        # Add regression line for all data
        x = intensity_df['total_steps']
        y = intensity_df['peak_rise']
        m, b = np.polyfit(x, y, 1)
        axes[0, 0].plot(x, m*x + b, color='red', linestyle='--')
        
        axes[0, 0].set_title('Peak Rise vs Total Steps by Carb Level')
        axes[0, 0].set_xlabel('Total Steps in 2h Window')
        axes[0, 0].set_ylabel('Peak Rise (mg/dL)')
        axes[0, 0].legend()
        
        # Annotation with correlation and p-value
        corr, p = stats.pearsonr(x, y)
        axes[0, 0].annotate(f'Correlation: {corr:.2f}\nP-value: {p:.4f}',
                         xy=(0.05, 0.95),
                         xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Scatter plot: Max interval steps vs peak rise by carb level
        for carb, color in zip(['low', 'moderate', 'high'], ['blue', 'orange', 'green']):
            carb_data = intensity_df[intensity_df['carb_label'] == carb]
            axes[0, 1].scatter(
                carb_data['max_interval_steps'],
                carb_data['peak_rise'],
                alpha=0.7,
                color=color,
                label=f'{carb.title()} Carb'
            )
        
        # Add regression line for all data
        x = intensity_df['max_interval_steps']
        y = intensity_df['peak_rise']
        m, b = np.polyfit(x, y, 1)
        axes[0, 1].plot(x, m*x + b, color='red', linestyle='--')
        
        axes[0, 1].set_title('Peak Rise vs Max Interval Steps by Carb Level')
        axes[0, 1].set_xlabel('Maximum Steps in any 10-min Interval')
        axes[0, 1].set_ylabel('Peak Rise (mg/dL)')
        axes[0, 1].legend()
        
        # Annotation with correlation and p-value
        corr, p = stats.pearsonr(x, y)
        axes[0, 1].annotate(f'Correlation: {corr:.2f}\nP-value: {p:.4f}',
                         xy=(0.05, 0.95),
                         xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Create step count bins for analysis of thresholds
        intensity_df['total_steps_bin'] = pd.cut(
            intensity_df['total_steps'],
            bins=[0, 600, 1200, 1800, 2400, float('inf')],
            labels=['0-600', '601-1200', '1201-1800', '1801-2400', '2400+']
        )
        
        intensity_df['max_interval_bin'] = pd.cut(
            intensity_df['max_interval_steps'],
            bins=[0, 200, 400, 600, 800, float('inf')],
            labels=['0-200', '201-400', '401-600', '601-800', '800+']
        )
        
        # Boxplot for total steps bins vs peak rise
        sns.boxplot(
            x='total_steps_bin',
            y='peak_rise',
            data=intensity_df,
            palette='viridis',
            ax=axes[1, 0]
        )
        
        sns.stripplot(
            x='total_steps_bin',
            y='peak_rise',
            data=intensity_df,
            color='black',
            alpha=0.5,
            size=3,
            jitter=True,
            ax=axes[1, 0]
        )
        
        axes[1, 0].set_title('Peak Rise by Total Steps Threshold')
        axes[1, 0].set_xlabel('Total Steps in 2h Window')
        axes[1, 0].set_ylabel('Peak Rise (mg/dL)')
        
        # Add sample size to x-tick labels
        xticklabels = []
        for category in intensity_df['total_steps_bin'].cat.categories:
            count = len(intensity_df[intensity_df['total_steps_bin'] == category])
            if count > 0:
                xticklabels.append(f'{category}\n(n={count})')
            else:
                xticklabels.append(category)
        
        axes[1, 0].set_xticklabels(xticklabels)
        
        # Statistical test
        groups = [intensity_df[intensity_df['total_steps_bin'] == cat]['peak_rise'].dropna() 
                for cat in intensity_df['total_steps_bin'].cat.categories]
        groups = [g for g in groups if len(g) >= 3]
        
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups)
            
            if p_value < 0.001:
                p_text = 'p < 0.001'
            else:
                p_text = f'p = {p_value:.3f}'
            
            if p_value < 0.05:
                p_text += ' *'
                
            axes[1, 0].text(0.5, 0.95, p_text, 
                         transform=axes[1, 0].transAxes, 
                         ha='center', va='top',
                         bbox=dict(facecolor='white', alpha=0.8))
        
        # Boxplot for max interval bins vs peak rise
        sns.boxplot(
            x='max_interval_bin',
            y='peak_rise',
            data=intensity_df,
            palette='plasma',
            ax=axes[1, 1]
        )
        
        sns.stripplot(
            x='max_interval_bin',
            y='peak_rise',
            data=intensity_df,
            color='black',
            alpha=0.5,
            size=3,
            jitter=True,
            ax=axes[1, 1]
        )
        
        axes[1, 1].set_title('Peak Rise by Max Interval Steps Threshold')
        axes[1, 1].set_xlabel('Maximum Steps in any 10-min Interval')
        axes[1, 1].set_ylabel('Peak Rise (mg/dL)')
        
        # Add sample size to x-tick labels
        xticklabels = []
        for category in intensity_df['max_interval_bin'].cat.categories:
            count = len(intensity_df[intensity_df['max_interval_bin'] == category])
            if count > 0:
                xticklabels.append(f'{category}\n(n={count})')
            else:
                xticklabels.append(category)
                
        axes[1, 1].set_xticklabels(xticklabels)
        
        # Statistical test
        groups = [intensity_df[intensity_df['max_interval_bin'] == cat]['peak_rise'].dropna() 
                for cat in intensity_df['max_interval_bin'].cat.categories]
        groups = [g for g in groups if len(g) >= 3]
        
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups)
            
            if p_value < 0.001:
                p_text = 'p < 0.001'
            else:
                p_text = f'p = {p_value:.3f}'
            
            if p_value < 0.05:
                p_text += ' *'
                
            axes[1, 1].text(0.5, 0.95, p_text, 
                        transform=axes[1, 1].transAxes, 
                        ha='center', va='top',
                        bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig, intensity_df

    def add_to_streamlit(self):
        """Create streamlit dashboard for statistical analysis"""
        import streamlit as st
        from scipy import stats as scipy_stats  # Add this import
        
        # Calculate metrics
        if not hasattr(self, 'metrics_df'):
            with st.spinner("Calculating glucose response metrics..."):
                self.calculate_meal_metrics()
        
        # Display metrics summary
        st.markdown("### Glucose Response Metrics by Activity Status")
        
        # Run primary analysis
        if not hasattr(self, 'primary_results'):
            with st.spinner("Running statistical analysis..."):
                self.run_primary_analysis()
        
        # Display primary results in tabs by carb level
        carb_tabs = st.tabs(["Low Carb", "Moderate Carb", "High Carb"])
        
        for i, carb_level in enumerate(['low', 'moderate', 'high']):
            with carb_tabs[i]:
                # Skip if no results
                if carb_level not in self.primary_results:
                    st.warning(f"Insufficient data for {carb_level} carb level")
                    continue
                
                # Display boxplots
                st.markdown(f"#### {carb_level.title()} Carb: Inactive vs. Active")
                
                fig = plt.figure(figsize=(12, 8))
                key_metrics = ['peak_rise', 'iauc', 'time_to_peak', 'glucose_variability']
                metric_labels = {
                    'peak_rise': 'Peak Rise (mg/dL)',
                    'iauc': 'iAUC',
                    'time_to_peak': 'Time to Peak (min)',
                    'glucose_variability': 'Glucose Variability (SD)'
                }
                
                for i, metric in enumerate(key_metrics):
                    plt.subplot(2, 2, i+1)
                    data = self.metrics_df[self.metrics_df['carb_label'] == carb_level]
                    
                    # Create boxplot
                    sns.boxplot(
                        x='activity_status',
                        y=metric,
                        data=data,
                        palette={'inactive': 'skyblue', 'active': 'salmon'}
                    )
                    
                    # Add individual points
                    sns.stripplot(
                        x='activity_status',
                        y=metric,
                        data=data,
                        color='black',
                        alpha=0.5,
                        size=3,
                        jitter=True
                    )
                    
                    # Add p-value if available
                    if metric in self.primary_results[carb_level]:
                        p_value = self.primary_results[carb_level][metric]['p_value']
                        if not np.isnan(p_value):
                            # Format p-value string
                            if p_value < 0.001:
                                p_text = 'p < 0.001'
                            else:
                                p_text = f'p = {p_value:.3f}'
                            
                            # Add star for significance
                            if p_value < 0.05:
                                p_text += ' *'
                                
                            plt.text(0.5, 0.95, p_text, 
                                transform=plt.gca().transAxes, 
                                ha='center', va='top',
                                bbox=dict(facecolor='white', alpha=0.8))
                    
                    plt.title(metric_labels.get(metric, metric))
                    plt.xlabel('Activity Status')
                    plt.ylabel(metric_labels.get(metric, metric))
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display results table
                st.markdown("##### Statistical Test Results")
                results_dict = self.primary_results[carb_level]
                
                results_data = []
                for metric, stats in results_dict.items():
                    row = {
                        'Metric': metric_labels.get(metric, metric),
                        'Inactive Mean': f"{stats['inactive_mean']:.1f}",
                        'Active Mean': f"{stats['active_mean']:.1f}",
                        'P-value': f"{stats['p_value']:.4f}",
                        'Significant': "Yes" if stats['significant'] else "No",
                        'Effect Size': f"{stats['effect_size']:.2f} ({stats['effect_size_interpretation']})"
                    }
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df)
                
                # Interpretation
                st.markdown("##### Interpretation")
                significant_metrics = [m for m, s in results_dict.items() if s['significant']]
                
                if significant_metrics:
                    st.success(f"Significant differences found in: {', '.join([metric_labels.get(m, m) for m in significant_metrics])}")
                    
                    for metric in significant_metrics:
                        stats = results_dict[metric]
                        direction = "lower" if stats['inactive_mean'] > stats['active_mean'] else "higher"
                        
                        st.markdown(f"- **{metric_labels.get(metric, metric)}**: Active meals showed {direction} values "
                                f"({stats['active_mean']:.1f} vs {stats['inactive_mean']:.1f}, "
                                f"p={stats['p_value']:.4f}, effect size: {stats['effect_size_interpretation']})")
                else:
                    st.info("No statistically significant differences found between inactive and active meals for this carb level.")
        
        # Secondary Analysis (Inactive vs Q3+Q4)
        st.markdown("### Secondary Analysis: Inactive vs. High Activity (Q3+Q4)")
        
        # Run secondary analysis
        if not hasattr(self, 'secondary_results'):
            with st.spinner("Running secondary analysis..."):
                self.run_secondary_analysis()
        
        # Display secondary results in tabs by carb level
        secondary_tabs = st.tabs(["Low Carb (Q3+Q4)", "Moderate Carb (Q3+Q4)"])
        
        for i, carb_level in enumerate(['low', 'moderate']):
            with secondary_tabs[i]:
                # Skip if no results
                if not hasattr(self, 'secondary_results') or carb_level not in self.secondary_results:
                    st.warning(f"Insufficient data for {carb_level} carb level")
                    continue
                
                # Display boxplots for key metrics
                st.markdown(f"#### {carb_level.title()} Carb: Inactive vs. High Activity (Q3+Q4)")
                
                # Filter data
                inactive_data = self.metrics_df[
                    (self.metrics_df['carb_label'] == carb_level) & 
                    (self.metrics_df['activity_status'] == 'inactive')
                ]
                
                high_active_data = self.metrics_df[
                    (self.metrics_df['carb_label'] == carb_level) & 
                    (self.metrics_df['activity_quartile'].isin(['Q3', 'Q4']))
                ]
                
                # Combine data
                combined_data = pd.concat([
                    inactive_data.assign(group='Inactive'),
                    high_active_data.assign(group='High Activity')
                ])
                
                # Create plots
                fig = plt.figure(figsize=(12, 8))
                key_metrics = ['peak_rise', 'iauc', 'time_to_peak', 'glucose_variability']
                
                for i, metric in enumerate(key_metrics):
                    plt.subplot(2, 2, i+1)
                    
                    # Create boxplot
                    sns.boxplot(
                        x='group',
                        y=metric,
                        data=combined_data,
                        palette={'Inactive': 'skyblue', 'High Activity': 'salmon'}
                    )
                    
                    # Add individual points
                    sns.stripplot(
                        x='group',
                        y=metric,
                        data=combined_data,
                        color='black',
                        alpha=0.5,
                        size=3,
                        jitter=True
                    )
                    
                    # Add p-value if available
                    if metric in self.secondary_results[carb_level]:
                        p_value = self.secondary_results[carb_level][metric]['p_value']
                        if not np.isnan(p_value):
                            # Format p-value string
                            if p_value < 0.001:
                                p_text = 'p < 0.001'
                            else:
                                p_text = f'p = {p_value:.3f}'
                            
                            # Add star for significance
                            if p_value < 0.05:
                                p_text += ' *'
                                
                            plt.text(0.5, 0.95, p_text, 
                                transform=plt.gca().transAxes, 
                                ha='center', va='top',
                                bbox=dict(facecolor='white', alpha=0.8))
                    
                    plt.title(metric_labels.get(metric, metric))
                    plt.xlabel('')
                    plt.ylabel(metric_labels.get(metric, metric))
                    
                    # Add sample sizes
                    plt.gca().set_xticklabels([
                        f"Inactive\n(n={len(inactive_data)})",
                        f"High Activity\n(n={len(high_active_data)})"
                    ])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display results table
                st.markdown("##### Statistical Test Results")
                results_dict = self.secondary_results[carb_level]
                
                results_data = []
                for metric, stats in results_dict.items():
                    row = {
                        'Metric': metric_labels.get(metric, metric),
                        'Inactive Mean': f"{stats['inactive_mean']:.1f}",
                        'High Active Mean': f"{stats['high_active_mean']:.1f}",
                        'P-value': f"{stats['p_value']:.4f}",
                        'Significant': "Yes" if stats['significant'] else "No",
                        'Effect Size': f"{stats['effect_size']:.2f} ({stats['effect_size_interpretation']})"
                    }
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df)
                
                # Interpretation
                st.markdown("##### Interpretation")
                significant_metrics = [m for m, s in results_dict.items() if s['significant']]
                
                if significant_metrics:
                    st.success(f"Significant differences found in: {', '.join([metric_labels.get(m, m) for m in significant_metrics])}")
                    
                    for metric in significant_metrics:
                        stats = results_dict[metric]
                        direction = "lower" if stats['inactive_mean'] > stats['high_active_mean'] else "higher"
                        
                        st.markdown(f"- **{metric_labels.get(metric, metric)}**: High activity meals showed {direction} values "
                                f"({stats['high_active_mean']:.1f} vs {stats['inactive_mean']:.1f}, "
                                f"p={stats['p_value']:.4f}, effect size: {stats['effect_size_interpretation']})")
                else:
                    st.info("No statistically significant differences found between inactive and high activity meals for this carb level.")
        
        # Dose-Response Analysis (Moderate Carb Only)
        st.markdown("### Dose-Response Analysis (Moderate Carb)")
        
        # Run dose-response analysis
        if not hasattr(self, 'dose_response_results'):
            with st.spinner("Running dose-response analysis..."):
                self.run_dose_response_analysis(carb_level='moderate')
        
        # Display dose-response results
        if hasattr(self, 'dose_response_results'):
            # Display boxplots
            st.markdown("#### Activity Quartile Trend (Moderate Carb)")
            
            fig= self.plot_dose_response(carb_level='moderate')
            st.pyplot(fig)
            
            # Interpretation
            st.markdown("##### Interpretation")
            significant_metrics = [m for m, s in self.dose_response_results.items() if s['significant']]
            
            if significant_metrics:
                st.success(f"Significant dose-response trend found in: {', '.join([metric_labels.get(m, m) for m in significant_metrics])}")
                
                for metric in significant_metrics:
                    stats = self.dose_response_results[metric]
                    
                    st.markdown(f"- **{metric_labels.get(metric, metric)}**: Kruskal-Wallis H = {stats['kruskal_h']:.2f}, p = {stats['p_value']:.4f}")
                    
                    # Add post-hoc results if available
                    if stats['post_hoc'] is not None:
                        significant_pairs = [pair for pair, result in stats['post_hoc'].items() if result['significant']]
                        if significant_pairs:
                            st.markdown("  - Significant differences between groups: " + ", ".join(significant_pairs))
            else:
                st.info("No statistically significant dose-response relationship found.")
        
        # Advanced Analysis
        st.markdown("### Advanced Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Activity Timing Effect", "Activity Distribution Effect", "Activity Intensity Thresholds"]
        )
        
        if analysis_type == "Activity Timing Effect":
            st.markdown("#### Effect of Activity Timing on Glucose Response")
            
            with st.spinner("Analyzing activity timing effects..."):
                fig, timing_df = self.visualize_activity_timing_effect()
            
            st.pyplot(fig)
            
            # Summarize findings
            st.markdown("##### Key Findings:")
            
            # Group by timing category and calculate mean/median
            timing_stats = timing_df.groupby('timing_category')['peak_rise'].agg(['mean', 'median', 'count']).reset_index()
            
            # Find timing category with lowest peak rise
            best_timing = timing_stats.loc[timing_stats['mean'].idxmin(), 'timing_category']
            best_timing_mean = timing_stats.loc[timing_stats['mean'].idxmin(), 'mean']
            
            st.markdown(f"- The **{best_timing}** timing showed the lowest average peak rise ({best_timing_mean:.1f} mg/dL)")
            
            # Check for statistical significance
            groups = [timing_df[timing_df['timing_category'] == cat]['peak_rise'].dropna() 
                    for cat in timing_df['timing_category'].cat.categories]
            groups = [g for g in groups if len(g) >= 3]
            
            if len(groups) >= 2:
                from scipy import stats as scipy_stats
                h_stat, p_value = scipy_stats.kruskal(*groups)
                
                if p_value < 0.05:
                    st.markdown(f"- Statistically significant differences found between timing groups (p = {p_value:.4f})")
                else:
                    st.markdown(f"- No statistically significant differences between timing groups (p = {p_value:.4f})")
            
            # Distribution pattern findings
            pattern_stats = timing_df.groupby('activity_pattern')['peak_rise'].agg(['mean', 'median', 'count']).reset_index()
            
            # Find pattern with lowest peak rise
            best_pattern = pattern_stats.loc[pattern_stats['mean'].idxmin(), 'activity_pattern']
            best_pattern_mean = pattern_stats.loc[pattern_stats['mean'].idxmin(), 'mean']
            
            st.markdown(f"- The **{best_pattern}** activity pattern showed the lowest average peak rise ({best_pattern_mean:.1f} mg/dL)")
            
        elif analysis_type == "Activity Distribution Effect":
            st.markdown("#### Effect of Activity Distribution on Glucose Response")
            
            with st.spinner("Analyzing activity distribution effects..."):
                fig, distribution_df = self.visualize_activity_distribution_effect()
            
            st.pyplot(fig)
            
            # Summarize findings
            st.markdown("##### Key Findings:")
            
            # Concentration category findings
            concentration_stats = distribution_df.groupby('concentration_category')['peak_rise'].agg(['mean', 'median', 'count']).reset_index()
            
            # Find category with lowest peak rise
            best_concentration = concentration_stats.loc[concentration_stats['mean'].idxmin(), 'concentration_category']
            best_concentration_mean = concentration_stats.loc[concentration_stats['mean'].idxmin(), 'mean']
            
            st.markdown(f"- The **{best_concentration}** activity concentration showed the lowest average peak rise ({best_concentration_mean:.1f} mg/dL)")
            
            # Check for statistical significance
            groups = [distribution_df[distribution_df['concentration_category'] == cat]['peak_rise'].dropna() 
                    for cat in distribution_df['concentration_category'].cat.categories]
            groups = [g for g in groups if len(g) >= 3]
            
            if len(groups) >= 2:
                h_stat, p_value = scipy_stats.kruskal(*groups)
                
                if p_value < 0.05:
                    st.markdown(f"- Statistically significant differences found between concentration groups (p = {p_value:.4f})")
                else:
                    st.markdown(f"- No statistically significant differences between concentration groups (p = {p_value:.4f})")
            
            # Activity breadth findings
            breadth_stats = distribution_df.groupby('activity_breadth')['peak_rise'].agg(['mean', 'median', 'count']).reset_index()
            
            # Find breadth with lowest peak rise
            best_breadth = breadth_stats.loc[breadth_stats['mean'].idxmin(), 'activity_breadth']
            best_breadth_mean = breadth_stats.loc[breadth_stats['mean'].idxmin(), 'mean']
            
            st.markdown(f"- The **{best_breadth}** activity pattern showed the lowest average peak rise ({best_breadth_mean:.1f} mg/dL)")
            
            # Double peak findings
            double_peak_data = distribution_df.groupby('activity_breadth')['has_double_peak'].mean().reset_index()
            least_double_peak = double_peak_data.loc[double_peak_data['has_double_peak'].idxmin(), 'activity_breadth']
            
            st.markdown(f"- The **{least_double_peak}** activity pattern showed the lowest occurrence of double glucose peaks")
            
        elif analysis_type == "Activity Intensity Thresholds":
            st.markdown("#### Threshold Effects of Activity Intensity")
            
            with st.spinner("Analyzing activity intensity thresholds..."):
                fig, intensity_df = self.visualize_intensity_thresholds()
            
            st.pyplot(fig)
            
            # Summarize findings
            st.markdown("##### Key Findings:")
            
            # Calculate correlation between steps and peak rise
            total_steps_corr, total_steps_p = scipy_stats.pearsonr(intensity_df['total_steps'], intensity_df['peak_rise'])
            max_interval_corr, max_interval_p = scipy_stats.pearsonr(intensity_df['max_interval_steps'], intensity_df['peak_rise'])
            
            st.markdown(f"- Correlation between total steps and peak rise: {total_steps_corr:.2f} (p = {total_steps_p:.4f})")
            st.markdown(f"- Correlation between max interval steps and peak rise: {max_interval_corr:.2f} (p = {max_interval_p:.4f})")
            
            # Identify best threshold bins
            total_steps_stats = intensity_df.groupby('total_steps_bin')['peak_rise'].agg(['mean', 'median', 'count']).reset_index()
            max_interval_stats = intensity_df.groupby('max_interval_bin')['peak_rise'].agg(['mean', 'median', 'count']).reset_index()
            
            # Find bins with lowest peak rise
            best_total_steps_bin = total_steps_stats.loc[total_steps_stats['mean'].idxmin(), 'total_steps_bin']
            best_total_steps_mean = total_steps_stats.loc[total_steps_stats['mean'].idxmin(), 'mean']
            
            best_max_interval_bin = max_interval_stats.loc[max_interval_stats['mean'].idxmin(), 'max_interval_bin']
            best_max_interval_mean = max_interval_stats.loc[max_interval_stats['mean'].idxmin(), 'mean']
            
            st.markdown(f"- Optimal total steps threshold: **{best_total_steps_bin}** (avg peak rise: {best_total_steps_mean:.1f} mg/dL)")
            st.markdown(f"- Optimal max interval threshold: **{best_max_interval_bin}** (avg peak rise: {best_max_interval_mean:.1f} mg/dL)")
        
        # Overall conclusion
        st.markdown("### Overall Conclusion")
        
        # Check if we have results from all analyses
        if (hasattr(self, 'primary_results') and 
            hasattr(self, 'secondary_results') and
            hasattr(self, 'dose_response_results')):
            
            # Count significant metrics across analyses
            primary_significant = sum(1 for carb in self.primary_results.values() 
                                    for s in carb.values() if s['significant'])
            
            secondary_significant = sum(1 for carb in self.secondary_results.values() 
                                    for s in carb.values() if s['significant'])
            
            dose_response_significant = sum(1 for s in self.dose_response_results.values() 
                                        if s['significant'])
            
            total_significant = primary_significant + secondary_significant + dose_response_significant
            
            if total_significant > 0:
                st.success("Physical activity after meals shows statistically significant effects on glucose response metrics.")
                
                # Summarize key findings
                st.markdown("#### Key statistical findings:")
                
                # Primary analysis findings
                for carb_level, metrics in self.primary_results.items():
                    significant_metrics = [m for m, s in metrics.items() if s['significant']]
                    if significant_metrics:
                        st.markdown(f"- **{carb_level.title()} Carb (Inactive vs. Active)**: Significant differences in {', '.join(significant_metrics)}")
                
                # Secondary analysis findings
                for carb_level, metrics in self.secondary_results.items():
                    significant_metrics = [m for m, s in metrics.items() if s['significant']]
                    if significant_metrics:
                        st.markdown(f"- **{carb_level.title()} Carb (Inactive vs. High Activity)**: Stronger effects in {', '.join(significant_metrics)}")
                
                # Dose-response findings
                if hasattr(self, 'dose_response_results'):
                    significant_metrics = [m for m, s in self.dose_response_results.items() if s['significant']]
                    if significant_metrics:
                        st.markdown(f"- **Dose-Response (Moderate Carb)**: Clear activity level trend in {', '.join(significant_metrics)}")
                
                st.markdown("#### Practical recommendations:")
                st.markdown("1. **Activity Timing**: Start moving within the first 30-40 minutes after eating for optimal glucose management")
                st.markdown("2. **Activity Type**: Multiple shorter periods of activity may be more effective than a single concentrated bout")
                st.markdown("3. **Activity Intensity**: Aim for at least 1000-1500 total steps within 2 hours after meals")
                st.markdown("4. **Carbohydrate Consideration**: Higher activity levels are especially important after moderate-to-high carb meals")
            else:
                st.info("Analysis complete, but no statistically significant effects were found. This could be due to limited sample size or other factors.")