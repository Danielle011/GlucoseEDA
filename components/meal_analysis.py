import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from collections import defaultdict

class MealAnalysis:
    def __init__(self):
        """Initialize MealAnalysis component"""
        self.load_and_prepare_data()
        self.setup_category_maps()

    def load_and_prepare_data(self):
        """Load and prepare data for analysis"""
        # Load data - REMOVE feature_df
        self.meal_df = pd.read_csv('data/processed_meal_data.csv')
        self.glucose_df = pd.read_csv('data/processed_glucose_data.csv')
        self.activity_df = pd.read_csv('data/aggregated_activity_data.csv')
        
        # Convert datetime columns
        self.meal_df['meal_time'] = pd.to_datetime(self.meal_df['meal_time'])
        self.glucose_df['DateTime'] = pd.to_datetime(self.glucose_df['DateTime'])
        self.activity_df['start_time'] = pd.to_datetime(self.activity_df['start_time'])
        self.activity_df['end_time'] = pd.to_datetime(self.activity_df['end_time'])
        
        # Create meal type category with proper order
        meal_order = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
        self.meal_df['meal_type'] = pd.Categorical(
            self.meal_df['meal_type'],
            categories=meal_order,
            ordered=True
        )
        
        # Setup food categories
        self.setup_category_maps()
        
        # Add food category columns to meal_df
        self._add_food_categories()
        
        # Calculate meal window duration and add labels
        self._calculate_meal_windows()
        self._add_activity_labels()
        self._add_carb_labels()
        self._calculate_glucose_responses()     

    def display_meal_criteria(self):
        """Display meal analysis criteria and distribution visualization"""
        # Display criteria explanation
        st.subheader("1. Analysis Criteria")
        
        with st.expander("**Click to see detailed criteria**", expanded=True):
            # 1. Meal Window
            st.markdown("#### 1. Meal Window")
            st.markdown("""
            - **Valid window**: At least 110 minutes until the next meal (maximum 120 minutes)
            - This ensures we can observe the full post-meal glucose response without interference from the next meal
            """)
            
            # 2. Post-meal Activity
            st.markdown("#### 2. Post-meal Activity Classification")
            st.markdown("""
            **Post-meal Rest:** Both conditions must be met
            - Less than 600 total steps during the 2-hour post-meal window
            - No more than 200 steps in any 10-minute interval during the post-meal window
            
            **Post-meal Movement:**
            - Any activity exceeding either of these thresholds
            - This represents having at least some meaningful movement during the post-meal period, not necessarily intense exercise
            """)
            
            # 3. Carbohydrate Categories
            st.markdown("#### 3. Carbohydrate Content Categories")
            st.markdown("""
            - **Low**: Less than 30 grams of carbohydrates per meal
            - **Moderate**: Between 30-75 grams of carbohydrates per meal
            - **High**: More than 75 grams of carbohydrates per meal
            """)

        # Calculate distributions for Sankey diagram
        total_meals = len(self.meal_df)
        valid_window = len(self.meal_df[self.meal_df['window_duration'] >= 110])
        invalid_window = total_meals - valid_window
        
        # For valid window meals, calculate activity distribution
        valid_meals = self.meal_df[self.meal_df['window_duration'] >= 110]
        
        rest_meals = valid_meals[
            (valid_meals['total_steps'] < 600) & 
            (valid_meals['max_interval_steps'] <= 200)
        ]
        movement_meals = valid_meals[
            (valid_meals['total_steps'] >= 600) | 
            (valid_meals['max_interval_steps'] > 200)
        ]
        
        # Calculate carb distributions for each activity type
        def get_carb_counts(df):
            return {
                'low': len(df[df['carbohydrates'] < 30]),
                'moderate': len(df[(df['carbohydrates'] >= 30) & (df['carbohydrates'] <= 75)]),
                'high': len(df[df['carbohydrates'] > 75])
            }
        
        rest_carbs = get_carb_counts(rest_meals)
        movement_carbs = get_carb_counts(movement_meals)
        
        # Display summary metrics
        st.subheader("2. Meal Aanalysis Criteria Distribution")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Meals",
                total_meals,
                f"Valid: {valid_window} ({valid_window/total_meals*100:.1f}%)"
            )
        
        with col2:
            st.metric(
                "Valid Window",
                valid_window,
                f"{invalid_window} meals filtered out"
            )
        
        with col3:
            st.metric(
                "Post-meal Rest",
                len(rest_meals),
                f"{len(rest_meals)/valid_window*100:.1f}% of valid meals"
            )
            
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node = {
                'pad': 15,
                'thickness': 20,
                'line': {'color': 'black', 'width': 0.5},
                'label': [
                    f'Total Meals\n({total_meals})', 
                    f'Valid Window\n({valid_window})', 
                    f'Invalid Window\n({invalid_window})',
                    f'Post-meal Rest\n({len(rest_meals)})', 
                    f'Post-meal Movement\n({len(movement_meals)})',
                    f'Low Carb\n({rest_carbs["low"]})', 
                    f'Moderate Carb\n({rest_carbs["moderate"]})', 
                    f'High Carb\n({rest_carbs["high"]})',
                    f'Low Carb\n({movement_carbs["low"]})', 
                    f'Moderate Carb\n({movement_carbs["moderate"]})', 
                    f'High Carb\n({movement_carbs["high"]})'
                ],
                'color': [
                    '#7CB9E8',  # Total
                    '#90EE90', '#FFB4B4',  # Window
                    '#C3B1E1', '#F8C8DC',  # Activity
                    '#E6E6FA', '#E6E6FA', '#E6E6FA',  # Rest carbs
                    '#FFE4E1', '#FFE4E1', '#FFE4E1'   # Movement carbs
                ]
            },
            link = {
                'source': [
                    # From Total
                    0, 0,
                    # From Valid Window
                    1, 1,
                    # From Rest
                    3, 3, 3,
                    # From Movement
                    4, 4, 4
                ],
                'target': [
                    # To Window
                    1, 2,
                    # To Activity
                    3, 4,
                    # To Rest Carbs
                    5, 6, 7,
                    # To Movement Carbs
                    8, 9, 10
                ],
                'value': [
                    # Window splits
                    valid_window,
                    invalid_window,
                    # Activity splits
                    len(rest_meals),
                    len(movement_meals),
                    # Rest carb splits
                    rest_carbs['low'],
                    rest_carbs['moderate'],
                    rest_carbs['high'],
                    # Movement carb splits
                    movement_carbs['low'],
                    movement_carbs['moderate'],
                    movement_carbs['high']
                ],
                'hovertemplate': 'Count: %{value}<br>%{target.label}<extra></extra>'
            }
        )])
        
        # Update layout
        fig.update_layout(

            font=dict(size=16),
            height=600,
            margin=dict(t=60, l=20, r=20, b=20)
        )
        
        # Display Sankey diagram
        st.plotly_chart(fig, use_container_width=True)

    def _add_food_categories(self):
        """Add food category columns to meal_df"""
        # Initialize category columns
        for category in self.category_maps.keys():
            self.meal_df[f'contains_{category}'] = False
        
        # Check each meal for categories
        for idx, row in self.meal_df.iterrows():
            food_items = [
                item.strip() 
                for item in row['food_name'].replace(' / ', '/').replace('/ ', '/').replace(' /', '/').split('/')
                if item.strip()
            ]
            
            # Check each food item against categories
            for item in food_items:
                if len(item) > 1:  # Skip single characters
                    for category, keywords in self.category_maps.items():
                        if any(keyword in item.lower() for keyword in keywords):
                            self.meal_df.at[idx, f'contains_{category}'] = True
            
            # Add total categories count
            category_cols = [col for col in self.meal_df.columns if col.startswith('contains_')]
            self.meal_df.at[idx, 'total_categories'] = self.meal_df.loc[idx, category_cols].sum()
        
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
            return 0, 0  # No activity data
        
        total_steps = window_activity['steps'].sum()
        max_interval_steps = window_activity['steps'].max()
        
        return total_steps, max_interval_steps
        
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

    def _calculate_glucose_response(self, meal_time, window_duration, measurement_number):
        """Calculate glucose response metrics for a single meal"""
        window_end = meal_time + pd.Timedelta(minutes=window_duration)
        
        # Get glucose values for the window
        mask = (
            (self.glucose_df['DateTime'] >= meal_time) &
            (self.glucose_df['DateTime'] <= window_end) &
            (self.glucose_df['MeasurementNumber'] == measurement_number)
        )
        glucose_data = self.glucose_df[mask].copy()
        
        if len(glucose_data) < 10:  # Skip if too few readings
            return None
            
        # Find baseline (closest reading to meal time)
        glucose_data['time_diff'] = abs(
            (glucose_data['DateTime'] - meal_time).dt.total_seconds()
        )
        baseline_idx = glucose_data['time_diff'].idxmin()
        baseline = glucose_data.loc[baseline_idx, 'GlucoseValue']
        
        # Calculate minutes from meal
        glucose_data['minutes'] = (
            (glucose_data['DateTime'] - meal_time).dt.total_seconds() // 60
        ).astype(int)  # Convert to integer minutes
        
        # Calculate peak rise
        peak = glucose_data['GlucoseValue'].max()
        peak_rise = peak - baseline
        time_to_peak = glucose_data.loc[
            glucose_data['GlucoseValue'].idxmax(), 'minutes'
        ]
        
        # Calculate AUC above baseline
        glucose_data = glucose_data.sort_values('minutes')
        auc = np.trapz(
            np.maximum(glucose_data['GlucoseValue'] - baseline, 0),
            glucose_data['minutes']
        )
        
        # Store time series for average calculation
        time_series = glucose_data[['minutes', 'GlucoseValue']].copy()
        time_series['value_above_baseline'] = time_series['GlucoseValue'] - baseline
        
        return {
            'baseline': baseline,
            'peak_rise': peak_rise,
            'time_to_peak': time_to_peak,
            'auc': auc,
            'time_series': time_series
        }

    def _calculate_glucose_responses(self):
        """Calculate glucose response metrics for all valid meals"""
        response_data = []
        time_series_data = []
        
        for _, meal in self.meal_df.iterrows():
            # Skip meals with short windows
            if meal['window_duration'] < 110:
                continue
                
            response = self._calculate_glucose_response(
                meal['meal_time'],
                meal['window_duration'],
                meal['measurement_number']
            )
            
            if response is not None:
                response_data.append({
                    'meal_id': meal.name,
                    'baseline': response['baseline'],
                    'peak_rise': response['peak_rise'],
                    'time_to_peak': response['time_to_peak'],
                    'auc': response['auc']
                })
                
                # Add time series data with meal identifiers
                time_series = response['time_series']
                time_series['meal_id'] = meal.name
                time_series['carb_label'] = meal['carb_label']
                time_series['activity_label'] = meal['activity_label']
                time_series_data.append(time_series)
        
        # Add response metrics to meal_df
        response_df = pd.DataFrame(response_data)
        self.meal_df = self.meal_df.join(response_df.set_index('meal_id'))
        
        # Store time series data
        self.glucose_responses = pd.concat(time_series_data, ignore_index=True)

    def create_carb_response_scatter(self):
        """Create separate scatter plots for carb vs glucose response metrics"""
        # Filter for inactive meals with valid responses
        inactive_meals = self.meal_df[
            (self.meal_df['activity_label'] == 'inactive') &
            (self.meal_df['peak_rise'].notna())
        ]
        
        # Create subplot figure
        fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Peak Rise vs Carbs', 'AUC vs Carbs'))
        
        # Add peak rise scatter
        fig.add_trace(
            go.Scatter(
                x=inactive_meals['carbohydrates'],
                y=inactive_meals['peak_rise'],
                mode='markers',
                name='Peak Rise',
                marker=dict(
                    color='blue',
                    size=10,
                    opacity=0.6
                ),
                hovertemplate=(
                    'Carbs: %{x:.1f}g<br>' +
                    'Peak Rise: %{y:.1f} mg/dL<br>' +
                    'Food: %{customdata}<br>' +
                    '<extra></extra>'
                ),
                customdata=inactive_meals['food_name']
            ),
            row=1, col=1
        )
        
        # Add AUC scatter
        fig.add_trace(
            go.Scatter(
                x=inactive_meals['carbohydrates'],
                y=inactive_meals['auc'],
                mode='markers',
                name='AUC',
                marker=dict(
                    color='red',
                    size=10,
                    opacity=0.6
                ),
                hovertemplate=(
                    'Carbs: %{x:.1f}g<br>' +
                    'AUC: %{y:.1f}<br>' +
                    'Food: %{customdata}<br>' +
                    '<extra></extra>'
                ),
                customdata=inactive_meals['food_name']
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Glucose Response vs Carbohydrate Content (Inactive Meals)',
            showlegend=False,
            height=500,
            width=1000
        )
        
        # Update x and y axes
        fig.update_xaxes(title_text="Carbohydrates (g)", row=1, col=1)
        fig.update_xaxes(title_text="Carbohydrates (g)", row=1, col=2)
        fig.update_yaxes(title_text="Peak Rise (mg/dL)", row=1, col=1)
        fig.update_yaxes(title_text="AUC", row=1, col=2)
        
        return fig

    def create_average_response_plot(self):
        """Create average response curves by carb category for inactive meals"""
        # Filter for inactive meals
        inactive_responses = self.glucose_responses[
            self.glucose_responses['activity_label'] == 'inactive'
        ]
        
        # Calculate average response for each carb category and minute
        avg_responses = inactive_responses.groupby(
            ['carb_label', 'minutes']
        )['value_above_baseline'].mean().reset_index()
        
        # Create figure
        fig = go.Figure()
        
        colors = {
            'low': 'rgba(99,110,250,0.7)',
            'moderate': 'rgba(239,85,59,0.7)',
            'high': 'rgba(0,204,150,0.7)'
        }
        
        # Add line for each carb category
        for carb_label in ['low', 'moderate', 'high']:
            category_data = avg_responses[avg_responses['carb_label'] == carb_label]
            category_data = category_data.sort_values('minutes')  # Ensure data is sorted by time
            
            fig.add_trace(
                go.Scatter(
                    x=category_data['minutes'],
                    y=category_data['value_above_baseline'],
                    name=f'{carb_label.title()} Carb',
                    line=dict(
                        color=colors[carb_label],
                        width=2
                    ),
                    hovertemplate='%{y:.1f} mg/dL<extra></extra>'
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Average Glucose Response by Carb Category (Inactive Meals)',
            xaxis_title='Minutes After Meal',
            yaxis_title='Glucose Above Baseline (mg/dL)',
            showlegend=True,
            height=500,
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=30,
                range=[0, 120]  # Ensure x-axis shows full 2-hour window
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            hovermode='x unified'
        )
        
        return fig

    def setup_category_maps(self):
        """Setup food category classification maps"""
        self.category_maps = {
            'refined_carb': {
                '라면', '우동', '파운드케이크', '쿠키', '피자', '식빵', '과자', '백미밥',
                '컵라면', '칼국수', '빵', '케이크', '쌀국수', '크루키', '비빔당면', '찐만두', 
                '볶음밥', '모듬전', '비빔막국수', '뻥튀기', '감자칩', '춘권', '감자전', 
                '치킨버거', '감자튀김', '부리또', '애플파이', '비빔밥', '옥수수밥', '김밥', 
                '야채김밥', '율무밥', '차조밥', '곤드레나물밥', '휘낭시에', '마들렌',
                '브루스케타', '시리얼', '난나나콘', 'CJ 비비고 왕교자', '고로케',
                '야채토스트', '들기름막국수', '물냉면', '버터갈릭감자', '리조또',
                '오리온 닥터유 단백질칩', '오리온 비쵸비', '치토스', '해태 자가비',
                '참브랜드 꿀밤맛 쫀디기', '스모어 오갓멜로 크리스피', '해태 롤리폴리 초코',
                '리치 오트밀 미니바이트', '샌드위치', '줄리스 피넛버터 샌드위치',
                '하겐다즈 크리스피 샌드위치', '잠봉뵈르샌드위치', '흑미밥', '훅미', '고구마', '어묵볶음'
            },
            'whole_grain': {
                '현미', '귀리', '보리', '귀리곤약밥', '보리밥', '현미밥',
                '그래놀라'
            },
            'beans': {
                '렌틸콩', '병아리콩', '검은콩', '도토리묵', '후무스'
            },
            'meat': {
                '닭가슴살', '삼겹살', '소고기', '돼지', '닭꼬치', '돈가스', '족발', '닭', '차돌박이',
                '제육', '불고기', '훈제오리', '고추마요치킨', '너비아니', '꽃등심스테이크', 
                '베이컨야채볶음', '구운치킨', '채끝스테이크', '양갈비구이', '삼계탕', '포케',
                'CJ 비비고 왕교자'
            },
            'fish': {
                '고등어', '연어', '조기', '갈치', '오징어젓', '쭈꾸미', '새우', '생선'
            },
            'egg': {
                '계란', '반숙', '프리타타', '오믈렛', '메추리알', '키토김밥',
                '스크렘블에그', '풀무원 동물복지 구운란', '에그인헬'
            },
            'tofu_bean': {
                '두부', '마파두부'
            },
            'vegetable': {
                '양배추', '브로콜리', '상추', '깻잎', '부추', '봄동', '샐러드',
                '양파', '파프리카', '가지', '오이', '호박', '고추',
                '무생채', '겉절이', '콩나물무침', '숙주나물무침', '과카몰리', '편백찜',
                '당근', '연근', '우엉', '양송이', '새송이', '버섯'
            },
            'fruit': {
                '토마토', '아보카도','사과', '복숭아', '무화과', '딸기', '수박', '청포도', '망고'
            },
            'dairy': {
                '요구르트', '그릭요거트', '우유', '카페라떼', '치즈', '부라타', 
                '그래놀라요거트', '파예 그릭 요거트'
            },
            'nuts': {
                '견과', '아몬드','땅콩버터', '피넛버터'
            },
            'pickled_veg': {
                '장아찌', '무말랭이', '단무지', '피클', '올리브', '깍두기', '석박지', '김치'
            },
            'protein_beverage': {
                '베지밀', '두유', '마이밀'
            },
            'sauce_seasoning': {
                '고추기름소스', '타르타르소스', '와사비', '쌈장', '고추장', '간장소스', 
                '땅콩소스', '발사믹', '백설탕'
            },
            'soup_stew': {
                '짬뽕국', '곰탕', '설렁탕', '미역국', '된장국', '된장찌개', '비지찌개', 
                '콩나물국', '감자탕', '강된장'
            },
            'sugar_beverage': {
                '아인슈페너', '티젠 콤부차', '밀크티'
            }
        }

    def analyze_food_composition(self):
        """Analyze food composition at both single food and meal levels"""
        # Single food analysis
        food_freq = defaultdict(int)
        food_categories = defaultdict(set)
        
        # Process each meal
        for _, row in self.meal_df.iterrows():
            food_items = [
                item.strip() 
                for item in row['food_name'].replace(' / ', '/').replace('/ ', '/').replace(' /', '/').split('/')
                if item.strip()
            ]
            
            # Count frequencies and categorize
            for item in food_items:
                if len(item) > 1:  # Skip single characters
                    food_freq[item] += 1
                    
                    # Direct keyword matching for categories
                    for category, keywords in self.category_maps.items():
                        if any(keyword in item.lower() for keyword in keywords):
                            food_categories[item].add(category)
        
        # Create DataFrame for single foods
        single_foods = []
        for food, freq in food_freq.items():
            food_data = {
                'food_name': food,
                'frequency': freq
            }
            
            # Add category columns
            for category in self.category_maps.keys():
                food_data[f'is_{category}'] = category in food_categories[food]
            
            # Add categories as string for easy viewing
            food_data['categories'] = ','.join(sorted(food_categories[food]))
            
            single_foods.append(food_data)
        
        self.single_foods_df = pd.DataFrame(single_foods)

    def display_top_foods(self):
        """Display top 20 most frequent foods with their categories in a table"""
        # Get top foods
        top_foods = self.single_foods_df.sort_values('frequency', ascending=False).head(20).reset_index(drop=True)
        
        # Create a cleaner display version with explicit rank column
        display_df = pd.DataFrame({
            'Rank': range(1, 21),  # Create ranks 1-10
            'Food': top_foods['food_name'],
            'Frequency': top_foods['frequency'],
            'Categories': top_foods['categories']
        })
        
        # Create custom CSS for table formatting with column width specifications
        st.markdown("""
            <style>
                table {
                    width: 100%;
                    margin: 1em 0;
                    border-collapse: separate;
                    border-spacing: 0;
                    table-layout: auto;
                }
                table td, table th {
                    padding: 0.75rem;
                    text-align: left;
                    white-space: nowrap;
                }
                /* Column-specific widths */
                .dataframe th:nth-child(1), .dataframe td:nth-child(1) {
                    width: 8%;  /* Rank column */
                }
                .dataframe th:nth-child(2), .dataframe td:nth-child(2) {
                    width: 45%;  /* Food column */
                }
                .dataframe th:nth-child(3), .dataframe td:nth-child(3) {
                    width: 12%;  /* Frequency column */
                }
                .dataframe th:nth-child(4), .dataframe td:nth-child(4) {
                    width: 35%;  /* Categories column */
                }
                table tbody tr:nth-of-type(odd) {
                    background-color: rgba(0, 0, 0, 0.05);
                }
                table th {
                    background-color: #f0f2f6;
                    border-bottom: 2px solid #dee2e6;
                }
                .stDataFrame {
                    font-size: 16px;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Display the DataFrame using st.dataframe
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )

    def create_category_distribution_plot(self):
        """Create category distribution plot"""
        # Calculate category frequencies
        category_cols = [col for col in self.single_foods_df.columns if col.startswith('is_')]
        category_counts = self.single_foods_df[category_cols].sum()
        category_counts.index = category_counts.index.str.replace('is_', '')
        
        # Sort by frequency
        category_counts = category_counts.sort_values(ascending=True)
        
        fig = go.Figure()
        
        # Create horizontal bar chart
        fig.add_trace(
            go.Bar(
                y=category_counts.index,
                x=category_counts.values,
                orientation='h',
                hovertemplate=(
                    'Category: %{y}<br>' +
                    'Count: %{x}<br>' +
                    '<extra></extra>'
                )
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Distribution of Food Categories',
            xaxis_title='Number of Foods',
            yaxis_title='Category',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_category_heatmap(self, cluster_data):
        """Create heatmap of category presence by cluster"""
        # Get relevant category columns
        category_cols = [col for col in self.meal_df.columns 
                        if col.startswith('contains_')]
        
        # Ensure all required columns exist in cluster_data
        missing_cols = [col for col in category_cols if col not in cluster_data.columns]
        if missing_cols:
            print(f"Warning: Missing category columns in cluster data: {missing_cols}")
            return None
        
        # Calculate category percentages for each cluster
        category_percentages = {}
        for cluster in sorted(cluster_data['response_cluster'].unique()):
            group_meals = cluster_data[cluster_data['response_cluster'] == cluster]
            total_meals = len(group_meals)
            
            percentages = {}
            for col in category_cols:
                category = col.replace('contains_', '')
                percentage = (group_meals[col].sum() / total_meals * 100)
                percentages[category] = percentage
                
            category_percentages[cluster] = percentages
        
        # Create heatmap data
        categories = [col.replace('contains_', '') for col in category_cols]
        clusters = sorted(cluster_data['response_cluster'].unique())
        
        z_data = [[category_percentages[cluster][cat] 
                for cluster in clusters] 
                for cat in categories]
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f'Cluster {c}' for c in clusters],
            y=categories,
            colorscale='Blues',
            text=np.round(z_data, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Category Presence (%) by Glucose Response Cluster',
            xaxis_title='Cluster',
            yaxis_title='Food Category',
            height=800,
            yaxis={'tickangle': 0}
        )
        
        return fig

    def create_feature_scatter_matrix(self, cluster_data):
        """Create scatter matrix of clustering features"""
        features = [
            'glucose_range',
            'auc_above_baseline',
            'glucose_variability',
            'net_auc'
        ]
        
        fig = go.Figure()
        
        # Create scatter plots for each pair of features
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i != j:  # Don't plot feature against itself
                    for cluster in sorted(cluster_data['response_cluster'].unique()):
                        cluster_subset = cluster_data[cluster_data['response_cluster'] == cluster]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=cluster_subset[feat1],
                                y=cluster_subset[feat2],
                                mode='markers',
                                name=f'Cluster {cluster}',
                                showlegend=(i == 0 and j == 1),  # Show legend only once
                                marker=dict(
                                    size=8,
                                    opacity=0.6
                                )
                            )
                        )
        
        # Update layout
        fig.update_layout(
            title='Feature Relationships by Cluster',
            grid=dict(rows=len(features)-1, columns=len(features)-1),
            height=800,
            width=800,
            showlegend=True
        )
        
        return fig

    def create_category_heatmap(self, cluster_data):
        """Create heatmap of category presence by cluster"""
        # Get relevant category columns
        category_cols = [col for col in self.meal_df.columns 
                        if col.startswith('contains_')]
        
        # Calculate category percentages for each cluster
        category_percentages = {}
        for cluster in sorted(cluster_data['response_cluster'].unique()):
            group_meals = cluster_data[cluster_data['response_cluster'] == cluster]
            total_meals = len(group_meals)
            
            percentages = {}
            for col in category_cols:
                category = col.replace('contains_', '')
                percentage = (group_meals[col].sum() / total_meals * 100)
                percentages[category] = percentage
                
            category_percentages[cluster] = percentages
        
        # Create heatmap data
        categories = [col.replace('contains_', '') for col in category_cols]
        clusters = sorted(cluster_data['response_cluster'].unique())
        
        z_data = [[category_percentages[cluster][cat] 
                for cluster in clusters] 
                for cat in categories]
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f'Cluster {c}' for c in clusters],
            y=categories,
            colorscale='Blues',
            text=np.round(z_data, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Category Presence (%) by Glucose Response Cluster',
            xaxis_title='Cluster',
            yaxis_title='Food Category',
            height=800,
            yaxis={'tickangle': 0}
        )
        
        return fig

    def create_category_and_response_plots(self):
        """Create heatmap of food category presence by carb level and glucose response scatter plots"""
        # Filter for inactive meals with valid window
        valid_meals = self.meal_df[
            (self.meal_df['window_duration'] >= 110) & 
            (self.meal_df['total_steps'] < 600) & 
            (self.meal_df['max_interval_steps'] <= 200)
        ].copy()
        
        # Create subplot figure - heatmap on top, two scatter plots below
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"colspan": 2}, None],  # Top row: full-width heatmap
                [{}, {}]                 # Bottom row: two scatter plots
            ],
            row_heights=[0.6, 0.4],
            subplot_titles=[
                "Food Category Presence by Carb Level",
                "Glucose Peak Rise vs Carb Content",
                "Glucose AUC vs Carb Content"
            ],
            vertical_spacing=0.15
        )
        
        # Add heatmap
        category_cols = [col for col in valid_meals.columns if col.startswith('contains_')]
        categories = [col.replace('contains_', '') for col in category_cols]
        carb_levels = ['low', 'moderate', 'high']
        
        z_data = []
        for category in categories:
            category_pcts = []
            for level in carb_levels:
                level_meals = valid_meals[valid_meals['carb_label'] == level]
                if len(level_meals) > 0:
                    pct = (level_meals[f'contains_{category}'].sum() / len(level_meals) * 100)
                    category_pcts.append(pct)
                else:
                    category_pcts.append(0)
            z_data.append(category_pcts)
        
        # Add heatmap trace
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=['Low Carb<br>(<30g)', 'Moderate Carb<br>(30-75g)', 'High Carb<br>(>75g)'],
                y=categories,
                colorscale='Blues',
                text=np.round(z_data, 1),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hoverongaps=False,
                colorbar=dict(
                    len=0.8,  # Make colorbar shorter
                    y=0.85,   # Position it near the top
                    title="Presence %"
                )
            ),
            row=1, col=1
        )
        
        # Add scatter plots
        colors = {'low': 'rgb(198,219,239)', 'moderate': 'rgb(107,174,214)', 'high': 'rgb(33,113,181)'}
        
        # Peak Rise scatter
        fig.add_trace(
            go.Scatter(
                x=valid_meals['carbohydrates'],
                y=valid_meals['peak_rise'],
                mode='markers',
                marker=dict(
                    color='rgba(70, 130, 180, 0.6)',
                    size=8
                ),
                text=valid_meals['food_name'],
                hovertemplate=(
                    'Carbs: %{x:.1f}g<br>' +
                    'Peak Rise: %{y:.1f} mg/dL<br>' +
                    'Food: %{text}<br>' +
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )
        
        # AUC scatter
        fig.add_trace(
            go.Scatter(
                x=valid_meals['carbohydrates'],
                y=valid_meals['auc'],
                mode='markers',
                marker=dict(
                    color='rgba(70, 130, 180, 0.6)',
                    size=8
                ),
                text=valid_meals['food_name'],
                hovertemplate=(
                    'Carbs: %{x:.1f}g<br>' +
                    'AUC: %{y:.1f}<br>' +
                    'Food: %{text}<br>' +
                    '<extra></extra>'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=False,
            title_text="Food Category Presence and Glucose Response by Carb Level"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Carbohydrates (g)", row=2, col=1)
        fig.update_xaxes(title_text="Carbohydrates (g)", row=2, col=2)
        fig.update_yaxes(title_text="Peak Rise (mg/dL)", row=2, col=1)
        fig.update_yaxes(title_text="AUC", row=2, col=2)
        
        return fig    
    
    def create_average_response_plot(self):
        """Create average glucose response curves by carb category"""
        # Filter for inactive meals with valid window
        valid_meals = self.meal_df[
            (self.meal_df['window_duration'] >= 110) & 
            (self.meal_df['total_steps'] < 600) & 
            (self.meal_df['max_interval_steps'] <= 200)
        ].copy()
        
        # Colors for carb categories
        colors = {
            'low': 'rgb(67, 147, 195)',     # Blue
            'moderate': 'rgb(178, 24, 43)',  # Red
            'high': 'rgb(27, 120, 55)'      # Green
        }        

        fig = go.Figure()
        
        # Calculate and plot average response for each carb category
        for carb_label in ['low', 'moderate', 'high']:
            category_meals = valid_meals[valid_meals['carb_label'] == carb_label]
            
            if len(category_meals) > 0:
                # Get all time series data for this category
                time_series_data = []
                for _, meal in category_meals.iterrows():
                    glucose_data = self.glucose_df[
                        (self.glucose_df['DateTime'] >= meal['meal_time']) &
                        (self.glucose_df['DateTime'] <= meal['meal_time'] + pd.Timedelta(minutes=110)) &
                        (self.glucose_df['MeasurementNumber'] == meal['measurement_number'])
                    ].copy()
                    
                    if len(glucose_data) > 0:
                        # Calculate minutes from meal and normalize glucose
                        glucose_data['minutes'] = (
                            (glucose_data['DateTime'] - meal['meal_time'])
                            .dt.total_seconds() / 60
                        )
                        baseline = glucose_data.iloc[0]['GlucoseValue']
                        glucose_data['glucose_normalized'] = glucose_data['GlucoseValue'] - baseline
                        
                        time_series_data.append(glucose_data[['minutes', 'glucose_normalized']])
                
                if time_series_data:
                    # Combine all time series
                    all_data = pd.concat(time_series_data)
                    
                    # Calculate mean and confidence intervals
                    grouped = all_data.groupby(
                        all_data['minutes'].round()
                    )['glucose_normalized'].agg(['mean', 'sem']).reset_index()
                    
                    # Calculate confidence intervals
                    conf_interval = 1.96 * grouped['sem']
                    
                    # Add trace
                    fig.add_trace(
                        go.Scatter(
                            name=f'{carb_label.title()} Carb',
                            x=grouped['minutes'],
                            y=grouped['mean'],
                            mode='lines',
                            line=dict(color=colors[carb_label], width=2),
                            hovertemplate=(
                                'Time: %{x:.0f} min<br>' +
                                'Glucose Change: %{y:.1f} mg/dL<br>' +
                                '<extra></extra>'
                            )
                        )
                    )
                    
                    # Add confidence interval
                    fig.add_trace(
                        go.Scatter(
                            name=f'{carb_label.title()} CI',
                            x=grouped['minutes'].tolist() + grouped['minutes'].tolist()[::-1],
                            y=(grouped['mean'] + conf_interval).tolist() + 
                            (grouped['mean'] - conf_interval).tolist()[::-1],
                            fill='toself',
                            fillcolor=colors[carb_label].replace('rgb', 'rgba').replace(')', ',0.2)'),
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title='Average Glucose Response by Carb Level (Inactive Meals)',
            xaxis_title='Minutes After Meal',
            yaxis_title='Change in Glucose (mg/dL)',
            showlegend=True,
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=20,
                range=[0, 110]
            )
        )
        
        return fig

    def render(self):
        """Main render method"""
        st.header("Meal & Glucose Response Analysis")
        
        # Part 1: Analysis Criteria and Data Overview
        self.display_meal_criteria()  # Keep this - shows Sankey diagram and criteria
        
        # Part 2: Food Composition Overview
        st.header("2. Top 20 Most Frequent Foods")
        if not hasattr(self, 'single_foods_df'):
            self.analyze_food_composition()
        self.display_top_foods()  # Keep this - shows top 20 foods table
        
        # Part 3: Food Category & Glucose Response Analysis (New section replacing macronutrient and clustering)
        st.header("3. Food Category & Glucose Response Analysis")
        
        # 3.1 Food Category Presence & Glucose Response
        st.subheader("3.1 Food Category Presence & Glucose Response")
        st.markdown("""
            Analysis filtered for:
            * Inactive meals: Both conditions must be met
            - Less than 600 total steps in post-meal window
            - No more than 200 steps in any 10-minute interval
            * Minimum 110-minute observation window before next meal
            (to ensure sufficient time to observe glucose response)
            * Meals categorized by carb content:
            - Low: < 30g
            - Moderate: 30-75g
            - High: > 75g
        """)
        # Create and display category and response plots
        category_response_fig = self.create_category_and_response_plots()
        st.plotly_chart(category_response_fig, use_container_width=True)
        
        # 3.2 Average Glucose Response Patterns
        st.subheader("3.2 Average Glucose Response Patterns")
        st.markdown("""
            - Average glucose response patterns with same filtering criteria as above
            - Shaded areas show 95% confidence intervals
            - Baseline glucose normalized to 0
            - X-axis shows minutes after meal (0-110 minutes)
        """)

        # Get valid meals
        valid_meals = self.meal_df[
            (self.meal_df['window_duration'] >= 110) & 
            (self.meal_df['total_steps'] < 600) & 
            (self.meal_df['max_interval_steps'] <= 200)
        ]

        # Create columns for meal counts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            low_carb_count = len(valid_meals[valid_meals['carb_label'] == 'low'])
            st.metric("Low Carb Meals", f"{low_carb_count} meals")
        
        with col2:
            moderate_carb_count = len(valid_meals[valid_meals['carb_label'] == 'moderate'])
            st.metric("Moderate Carb Meals", f"{moderate_carb_count} meals")
        
        with col3:
            high_carb_count = len(valid_meals[valid_meals['carb_label'] == 'high'])
            st.metric("High Carb Meals", f"{high_carb_count} meals")

        # Create and display average response plot
        avg_response_fig = self.create_average_response_plot()
        st.plotly_chart(avg_response_fig, use_container_width=True)