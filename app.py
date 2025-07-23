import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../output/cleaned_salaries.csv')

st.title("Data Science & AI Job Salaries Dashboard 2025")

# Sidebar filters
job_titles = df['job_title'].unique()
selected_jobs = st.sidebar.multiselect('Select Job Titles', job_titles, default=job_titles[:5])

companies = df['company_name'].unique()
selected_companies = st.sidebar.multiselect('Select Companies', companies, default=companies[:5])

locations = df['company_location'].unique()
selected_locations = st.sidebar.multiselect('Select Locations', locations, default=locations[:5])

# Filter data based on selections
filtered_df = df[
    (df['job_title'].isin(selected_jobs)) & 
    (df['company_name'].isin(selected_companies)) & 
    (df['company_location'].isin(selected_locations))
]

st.write(f"Showing {filtered_df.shape[0]} records after filtering.")

# Plot 1: Average Salary by Job Title
avg_salary = filtered_df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)

fig1, ax1 = plt.subplots(figsize=(10,6))
sns.barplot(x=avg_salary.values, y=avg_salary.index, palette='coolwarm', ax=ax1)
ax1.set_title("Average Salary by Job Title")
ax1.set_xlabel("Average Salary (USD)")
ax1.set_ylabel("Job Title")
st.pyplot(fig1)

# Plot 2: Salary Distribution for Top 5 Companies (by avg salary)
top_companies = filtered_df.groupby('company_name')['salary_in_usd'].mean().sort_values(ascending=False).head(5).index
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.boxplot(data=filtered_df[filtered_df['company_name'].isin(top_companies)], 
            x='salary_in_usd', y='company_name', palette='magma', ax=ax2)
ax2.set_title("Salary Distribution for Top 5 Companies")
ax2.set_xlabel("Salary (USD)")
ax2.set_ylabel("Company Name")
st.pyplot(fig2)

# Plot 3: Number of Jobs per Location
location_counts = filtered_df['company_location'].value_counts()
fig3, ax3 = plt.subplots(figsize=(8,5))
sns.barplot(x=location_counts.values, y=location_counts.index, palette='viridis', ax=ax3)
ax3.set_title("Number of Jobs per Location")
ax3.set_xlabel("Number of Jobs")
ax3.set_ylabel("Location")
st.pyplot(fig3)

# Line Plot - Salary Trends over Years
fig4, ax4 = plt.subplots(figsize=(10, 5))
year_salary = df.groupby('work_year')['salary_in_usd'].mean()
sns.lineplot(x=year_salary.index, y=year_salary.values, marker='o', ax=ax4)
ax4.set_title('Average Salary Over Years')
ax4.set_xlabel('Year')
ax4.set_ylabel('Average Salary (USD)')
ax4.grid(True)
plt.tight_layout()
st.pyplot(fig4)

# Heatmap - Correlation between numerical features
fig5, ax5 = plt.subplots(figsize=(8, 6))
numerical_data = df.select_dtypes(include=['int64', 'float64'])
corr = numerical_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
ax5.set_title('Correlation Between Numerical Features')
plt.tight_layout()
st.pyplot(fig5)

st.markdown("---")
st.markdown("**Note:** Use the sidebar filters to customize your view.")

