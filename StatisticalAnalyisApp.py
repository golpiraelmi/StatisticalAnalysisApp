# This is based on Request2
# source userinput/bin/activate
# streamlit run app.py


import streamlit as st
import os
from pathlib import Path
from redcap import Project
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats.contingency import expected_freq
from scipy.stats import chi2_contingency
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()
from tableone import TableOne
from scipy.stats import shapiro
from IPython.display import display


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='white')
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
########################################
url='https://github.com/golpiraelmi/StatisticalAnalysisApp/raw/refs/heads/main/data/My_Dataset.xlsx'
df=pd.read_excel(url, engine='openpyxl')
# ########################################
columns = ['Calgary_Edmonton','Age', 'Sex','In Person','Online', 'Hours', 'Group', 'Math Score', 'Literature Score'] 

categorical = ['Sex','In Person','Online','Group'] 

nonnormal = ['Math Score','Literature Score']

# ########################################

def my_tableone (df, cols, cats, non_norm, groupby='Calgary_Edmonton'):
    
    new_p_values={}
    for variable in non_norm:
        # Convert the column to numeric, coerce errors to NaN
        df[variable] = pd.to_numeric(df[variable], errors='coerce')


    # Perform Mann-Whitney U test for each variable and print the p-values
    for variable in non_norm:
        statistic, p_value = mannwhitneyu(df[df[groupby]=='Calgary'][variable].dropna(), 
                                        df[df[groupby]=='Edmonton'][variable].dropna())
        
        new_p_values[variable] = round(p_value,3)


    for col in categorical:
        df[col] = df[col].apply(lambda x: np.nan if pd.isnull(x) or x == 'nan' or x==None else x)

    table = TableOne(df, columns=cols, categorical=cats, groupby=groupby, nonnormal=non_norm, htest_name=True,
                    label_suffix=True, pval=True, missing=True, normal_test=True)
    
    table1_df = table.tableone

    table1_df.columns = table1_df.columns.get_level_values(1)
    table1_df = table1_df.reset_index()


    table1_df["P-Value"] = table1_df["level_0"].apply(
        lambda var: new_p_values.get(var.split(",")[0], table1_df["P-Value"].loc[table1_df["level_0"] == var].iloc[0]))

    table1_df.set_index(["level_0", "level_1"], inplace=True)

    # Get the first index for each group in 'level_0'
    first_idx = table1_df.groupby('level_0').head(1).index

    # Set P-Value to NaN for all rows except the first row for each group
    table1_df['P-Value'] = table1_df.apply(lambda row: row['P-Value'] if (row.name in first_idx) else '', axis=1)
    table1_df['Test'] = table1_df['Test'].replace({'Kruskal-Wallis': 'Mann-Whitney'})

    return table1_df
########################################
# Add a title
st.title("Statistical Analysis Project")

# Add some text
st.write("WORK IN PROGRESS")
# Checkbox options
st.markdown("<p style='color:red; font-weight:bold;'>Please select the methods you want to include in the analysis:</p>", unsafe_allow_html=True)

# Checkbox options
option_1 = st.checkbox("Method A")
option_2 = st.checkbox("Method B")
option_3 = st.checkbox("Method C")

to_include = []

if option_1:
    to_include.append("MethodA")
if option_2:
    to_include.append("MethodB")
if option_3:
    to_include.append("MethodC")





if to_include:
    # Filter rows based on selected hospital codes
    df = df[df["Group"].isin(to_include)]

    # Display the number of unique StudyIDs
    st.write(f"Number of unique StudyIDs included: {df['StudyID'].nunique()}")

    #Ensure there is data available for plotting
    if not df['Math Score'].isna().all() and not df['Literature Score'].isna().all():
        col1, col2, col3 = st.columns([2, 1, 1])  # Define column layout
        
        with col1:  # Place the plot in the center column
            fig, ax = plt.subplots(figsize=(5, 3), dpi=300)  # Adjust figure size
            
            # Define a custom palette for groups
            palette = {'MethodA': '#1f77b4', 'MethodB': '#ff7f0e', 'MethodC': '#2ca02c'}
            
            # Plot group-specific scatterplots and regression lines
            for group in df['Group'].unique():
                temp_df = df[df['Group'] == group]
                sns.regplot(
                    x='Math Score', 
                    y='Literature Score', 
                    data=temp_df, 
                    scatter_kws={'s': 10, 'color': palette[group]},  # Color dots
                    line_kws={'color': palette[group], 'label': group},  # Color regression lines
                    ax=ax
                )
            
            # Customize the legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title="Methods", loc='upper left', fontsize=8)

            ax.set_title("Math Score vs Literature Score by Group", fontsize=10, fontweight='bold')
            ax.set_xlabel("Math Score", fontsize=10)
            ax.set_ylabel("Literature Score", fontsize=10)
            ax.grid(True)
            
            st.pyplot(fig)
    else:
        st.warning("No valid data available for Math Score and Literature Score to plot.")


    # Run the analysis
    table1_df = my_tableone(df, columns, categorical, nonnormal)
    table1_df_html = table1_df.to_html()

    st.markdown("### TableOne Summary")
    
        
    st.markdown(table1_df_html, unsafe_allow_html=True)



    


########################################
