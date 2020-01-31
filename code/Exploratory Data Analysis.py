# Title: Exploratory Data Analysis (EDA) on Indeed data
# Purpose: To analyse job postings data extracts from Indeed website using Job Postings data scraping.py
# Created by: Saani Rawat
# Last modifed: 01/30/2020
# Dependencies:
# 1. Job Postings data scraping.py
# Inputs
# 1.  data_science_jobs_df.csv
# Outputs
# 1. avg_max_salary_by_city.png
# 2. job_count_by_city.png

# Importing modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# root directory (Change based on your computer)
dir = "/Users/saannidhyarawat/Desktop/TDI/Application/TDI Capstone Proposal"

# Reading in the csv file and creating a pandas DataFrame
os.chdir(dir+"/data")
ds_jobs = pd.read_csv("data_science_jobs_df.csv")
df = pd.DataFrame(ds_jobs)

# Checking the first 5 obs and describing the dataset
df.head()
df.describe()
df.columns

# Renaming the columns
cols = {'Unnamed: 0' : 'ID', '0': 'Title', '1' : 'Company', '2' : 'Company_Address', '3' : 'Salary', '4': 'Summary'}
df = df.rename(columns = cols)

# Removing observations which do not have salary information
df_sal = df.loc[~df.loc[:, 'Salary'].isnull(),:]
len(df_sal)

# Removing any duplicates based on Title, Company and Salary
df_sal_no_dup = df_sal.drop_duplicates(subset = ['Title', 'Company', 'Company_Address', 'Salary'])

# str(df_sal_no_dup['Company_Address'][0]).find('NY')

# Extracting city out of company address
df_sal_no_dup.loc[:, 'City'] = df_sal_no_dup['Company_Address'].apply(lambda x: str(x).split(',')[0])

# Extracting year, month or day out of salary
df_sal_no_dup.loc[:, 'Sal_type'] = df_sal_no_dup['Salary'].apply(lambda x: str(x).split(' ')[-1])

# Extracting max of the salary range
df_sal_no_dup.loc[:, 'Sal_max'] = df_sal_no_dup['Salary'].apply(lambda x: str(x).split(' ')[-3])
# Removing dollar sign and converting column into string
df_sal_no_dup.loc[:, 'Sal_max'] = df_sal_no_dup['Salary'].apply(lambda x: str(x).split('$')[1])
df_sal_no_dup.loc[:, 'Sal_max'] = df_sal_no_dup['Sal_max'].apply(lambda x: str(x).split(' ')[0])


# Converting all monthly, hourly, and weekly salaries into yearly
## Yearly
mon_bool = df_sal_no_dup['Sal_type'] == 'year'
df_sal_no_dup.loc[(mon_bool), 'Annual_Max_Salary'] = df_sal_no_dup['Sal_max'].apply(lambda x: float(str(x).replace(',','')))
## Monthly
mon_bool = df_sal_no_dup['Sal_type'] == 'month'
df_sal_no_dup.loc[(mon_bool), 'Annual_Max_Salary'] = df_sal_no_dup['Sal_max'].apply(lambda x: float(str(x).replace(',','')) * 12)
## Hourly (assuming 40 work hours and 52 weeks)
hour_bool = df_sal_no_dup['Sal_type'] == 'hour'
df_sal_no_dup.loc[(hour_bool), 'Annual_Max_Salary'] = df_sal_no_dup['Sal_max'].apply(lambda x: float(str(x).replace(',','')) * 40 * 52)
## Weekly
week_bool = df_sal_no_dup['Sal_type'] == 'week'
df_sal_no_dup.loc[(week_bool), 'Annual_Max_Salary'] = df_sal_no_dup['Sal_max'].apply(lambda x: float(str(x).replace(',','')) * 52)
## Removing class Salary type and resetting index
df_sal_no_dup = df_sal_no_dup.loc[~(df_sal_no_dup['Sal_type'] == 'class'), :].reset_index(drop=True)

# Removing observations with no cities
df_sal_no_dup = df_sal_no_dup[~(df_sal_no_dup['City'] == 'nan')]

# Creating a box plot with n.o of data science job openings in different locations
# plot1 = df_sal_no_dup[['ID','City']].groupby(['City']).agg('count').plot(kind = 'bar', legend = False, title = "Number of Data Science Postings by City")
plot1 = df_sal_no_dup[['ID','City']].groupby(['City']).agg('count').reset_index().rename(columns = {'ID' : 'Count'})
sns.set_style("dark")
sns.barplot(x = plot1['City'], y = plot1['Count'])
plt.xticks(rotation = 90)
plt.title("Number of Data Science Postings by City")
os.chdir(dir+'/plots')
plt.tight_layout()
plt.savefig('job_count_by_city.png')
plt.show()

# Calculating and plotting average salaries offered in different locations (average of the max)
df_sal_no_dup['Annual_Max_Salary'] = df_sal_no_dup['Annual_Max_Salary'].astype(int)
# plot2 = df_sal_no_dup[['City','Annual_Max_Salary']].groupby('City').agg('mean').plot(kind = 'bar', color = 'gray', title = 'On Average, Maximum Salaries offered by different Cities')
plot2 = df_sal_no_dup[['City','Annual_Max_Salary']].groupby('City').agg('mean').reset_index()
sns.set_style("whitegrid")
sns.set()
sns.barplot(x = plot2['City'], y = plot2['Annual_Max_Salary'], color='brown')
plt.xticks(rotation = 90)
plt.ylabel('Salary')
plt.title("On Average, Maximum Salaries offered by different Cities")
plt.tight_layout()
os.chdir(dir+'/plots')
plt.savefig('avg_max_salary_by_city.png')
plt.show()
