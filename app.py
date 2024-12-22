import streamlit as st
from components.daily_plot import DailyPlot
from components.glucose_analysis import GlucoseAnalysis
from components.meal_analysis import MealAnalysis
from components.activity_analysis import ActivityAnalysis

# Set page config
st.set_page_config(
    page_title="Blood Glucose EDA",
    layout="wide"
)

# Main title and description
st.title("Blood Glucose Data Exploration")
st.write("Interactive exploration of glucose, activity, and meal patterns")

# Overview Section (Always visible)
st.header("Daily Overview")
st.write("Comprehensive view of glucose, steps, and meals")

# Initialize and render daily plot
daily_plot = DailyPlot()
daily_plot.render()

# Visual separator
st.divider()

# Detailed Analysis Tabs
st.subheader("Detailed Analysis")
tab_glucose, tab_meal, tab_activity = st.tabs([
    "Glucose Analysis", "Meal Analysis", "Activity Analysis"
])

# Glucose Analysis Tab
with tab_glucose:
    glucose_analysis = GlucoseAnalysis()
    glucose_analysis.render()

# Meal Analysis Tab
with tab_meal:
    meal_analysis = MealAnalysis()
    meal_analysis.render()

# Activity Analysis Tab
with tab_activity:
    activity_analysis = ActivityAnalysis()
    activity_analysis.render()