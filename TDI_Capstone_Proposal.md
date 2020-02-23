
# TDI Capstone Proposal
######                        by Saani Rawat
######                        02/23/2020

Job markets, in today's world, are extremely competitive. When we talk about data science market, the level of competition is fierce. Employees want to identify the open positions that best suit their long-term goals and needs. On the other hand, employers wish to attract best talent out of the available pool of candidates and stay ahead of their competition.

The purpose of the project is to analyse the current market condition of data-related jobs in major metropolitan cities: New York City, San Francisco, Los Angeles, Charlotte, Boston and Washington. I used web scraping tools in Python to extrapolate job postings data. Data were obtained using web scraping techniques (using Python) on Indeed's website.
In this notebook, we use the data scraped using "Job Postings data scraping.py" to identify interest patterns and relationships in the extracted dataset.


##  1. Data Source 

Data for major U.S metropolitan cities was scraped from Indeed's website. source: https://www.indeed.com

To understand how the data was scraped, see <font color=red> Job Postings data scraping.py </font>, which can be found  [here](https://github.com/saannidhya/TDI-Capstone/blob/master/code/Job%20Postings%20data%20scraping.py)

##  2. Data Description 

Dataset name: data_science_jobs_df.csv

**Columns** -

*Unnamed column - contains index*

*0 - Job Title*

*1 - Company's Name*

*2 - Company's Location*

*3 - Salary*

*4 - Job Summary*


```python
# Importing libraries
import numpy as np
import pandas as pd 
import plotly.offline as pyo
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

    /kaggle/input/data_science_jobs_df.csv
    

## 3. Data Cleaning 

Scraped data is often times quite messy. Hence, appropriate cleaning needs to be performed on datasets to make them comprehensible and suitable for analysis.
In this section, we identify and eliminate missing values, duplicates and other non-sensical observations.


```python
# Reading in the dataset
data_science_jobs_df = pd.read_csv("../input/data_science_jobs_df.csv")
df = pd.DataFrame(data_science_jobs_df)
```


```python
# Checking the first 5 rows
data_science_jobs_df.head()
# The column names are not very intuitive. Let's change them.
cols = {'Unnamed: 0' : 'ID', '0': 'Title', '1' : 'Company', '2' : 'Company_Address', '3' : 'Salary', '4': 'Summary'}
df = df.rename(columns = cols)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Title</th>
      <th>Company</th>
      <th>Company_Address</th>
      <th>Salary</th>
      <th>Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Data Scientist Intern</td>
      <td>81qd</td>
      <td>New York, NY 10017 (Turtle Bay area)</td>
      <td>$120,000 - $180,000 a year</td>
      <td>The Data Scientist is responsible for the data...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Lead Data Scientist - IT/Financial</td>
      <td>Trova Advisory Group</td>
      <td>Brooklyn, NY</td>
      <td>$350,000 - $500,000 a year</td>
      <td>The candidate should be a self-starter and a g...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist (NLP Focused)</td>
      <td>Fakespot</td>
      <td>New York, NY 10005 (Financial District area)</td>
      <td>$80,000 - $150,000 a year</td>
      <td>You have a passion for solving complex analyti...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist/Data Science Instructor</td>
      <td>NYC Data Science Academy</td>
      <td>New York, NY</td>
      <td>$125,000 - $190,000 a year</td>
      <td>Train in-person or online boot camp students, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the n.o of obs in the dataset
len(df)
# df.loc[0:100,:]
```




    30712



A lot of the observations for the Salary column in the dataset have missing values. For our purposes, salary is the one of the most important columns. Hence, *removing job postings which do not reveal the offered salary*.


```python
# Removing observations with no salary information
df_sal = df.loc[~df.loc[:, 'Salary'].isnull(),:]
# Checking length of the new dataset
len(df_sal)
# nobs removed
len(df) - len(df_sal)
```




    26012



26012 observations did not contain salary information and hence, were removed from the dataset.

Data scraped from web pages often contains duplicates because the web pages often redirect to the same page during iteration. Hence, checking if any duplicate observations exist in our dataset with populated salary information.


```python
# Checking the n.o of duplicates in the dataset based on Title, Company, Company_Address and Salary
df_sal[df_sal.duplicated(['Title', 'Company', 'Company_Address', 'Salary'])]
# len(df_sal[df_sal.duplicated(['Title', 'Company', 'Company_Address', 'Salary'])])
# 3736
# An e.g of duplicate observation -
df_sal.loc[(df_sal['Company'] == 'Qloo') & (df_sal['Title'] == 'Senior Data Scientist') , :]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Title</th>
      <th>Company</th>
      <th>Company_Address</th>
      <th>Salary</th>
      <th>Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Qloo offers a competitive compensation and ben...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$125,000 - $190,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>293</th>
      <td>293</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>361</th>
      <td>361</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>393</th>
      <td>393</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>409</th>
      <td>409</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>555</th>
      <td>555</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>571</th>
      <td>571</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>587</th>
      <td>587</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>635</th>
      <td>635</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>651</th>
      <td>651</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>862</th>
      <td>862</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$80,000 - $150,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>877</th>
      <td>877</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>957</th>
      <td>957</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>1021</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1037</th>
      <td>1037</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1072</th>
      <td>1072</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1088</th>
      <td>1088</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>1104</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>1178</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$61,131 - $115,251 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1280</th>
      <td>1280</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$27 - $30 an hour</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1330</th>
      <td>1330</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>1346</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
    <tr>
      <th>1362</th>
      <td>1362</td>
      <td>Senior Data Scientist</td>
      <td>Qloo</td>
      <td>New York, NY 10002 (Lower East Side area)</td>
      <td>$120,000 - $155,000 a year</td>
      <td>Experience with common analysis tools like SQL...</td>
    </tr>
  </tbody>
</table>
</div>



Our data contains** a lot **of duplicates. We need to get rid of these as they can skew our analysis.


```python
# Removing duplicates
df_sal_no_dup = df_sal.drop_duplicates(subset = ['Title', 'Company', 'Company_Address', 'Salary'])
len(df_sal_no_dup)
```




    964



Now, we are left with 964 valid (dare I say clean) observations for our analysis.

Next, we need to clean the columns so that appropriate analysis can be performed.
1. We need to extract city out of company address 
2. Salary range is given instead of salary, and that too in different formats (monthly, yearly, daily etc.). We need to align them.



```python
# Extracting city out of company address
pd.options.mode.chained_assignment = None  # default='warn'
df_sal_no_dup.loc[:, 'City'] = df_sal_no_dup['Company_Address'].apply(lambda x: str(x).split(',')[0])
```


```python
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
```


```python
# Removing observations with no cities
df_sal_no_dup = df_sal_no_dup[~(df_sal_no_dup['City'] == 'nan')]
# nobs = 961 
```

Now, our data is ready for analysis. In the next section, we will explore this data and use visualization to recognize patterns.
    


## 4. Exploratory Data Analysis

In this section, we use data visualization techniques to answer interesting questions.

Specifically, we want to know the following -

*how many data science jobs are available per city?*


```python
# Creating a box plot with n.o of data science job openings in different locations
# plot1 = df_sal_no_dup[['ID','City']].groupby(['City']).agg('count').plot(kind = 'bar', legend = False, title = "Number of Data Science Postings by City")
plot1 = df_sal_no_dup[['ID','City']].groupby(['City']).agg('count').reset_index().rename(columns = {'ID' : 'Count'})
sns.set_style("dark")
sns.barplot(x = plot1['City'], y = plot1['Count'])
plt.xticks(rotation = 90)
plt.title("Number of Data Science Postings by City")
plt.tight_layout()
plt.show()
```


![png](TDI_Capstone_Proposal_files/TDI_Capstone_Proposal_27_0.png)


Based on the graph obove, New York has the highest number of job postings for data scientists. Also, notice that we were able to extract many more cities from company address than the cities specified in the introduction section of the document. This is because our data included job postings which belonged to neigborhoods close to aforementioned major metropolitan cities.

 *What average salaries are being offered in these cities?*


```python
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
plt.show()
```


![png](TDI_Capstone_Proposal_files/TDI_Capstone_Proposal_30_0.png)


Out of the cities, San Rafael, which is close to San Francisco, seems to be offering the highest salary for data scientist positions.

*Which companies are most actively looking for data scientists?*


```python
import chart_studio.plotly as py
## Companies with highest n.o of postings
plot3 = df_sal_no_dup[['ID','Company']].groupby(['Company']).agg('count').reset_index().rename(columns = {'ID': 'Count'})
plot3 = plot3.loc[plot3['Count'] >= 10, :]
data_bar = [go.Bar(x = plot3['Company'] , y = plot3['Count'], name = 'company_count_barplot', marker = dict(color = '#109618'), width = 0.5)]
layout = go.Layout(title = 'Companies with 10 or more Data Science job postings', xaxis_tickangle = -90)
fig = go.Figure(data = data_bar, layout = layout)
fig.layout.template = 'plotly_white'
pyo.iplot(fig)
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        



<div>
        
        
            <div id="f71ac915-9c53-4f71-8339-34e0774bdced" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("f71ac915-9c53-4f71-8339-34e0774bdced")) {
                    Plotly.newPlot(
                        'f71ac915-9c53-4f71-8339-34e0774bdced',
                        [{"marker": {"color": "#109618"}, "name": "company_count_barplot", "type": "bar", "width": 0.5, "x": ["Amazon.com Services LLC", "Booz Allen Hamilton", "Capital One - US", "Deloitte", "National Security Agency", "Seen by Indeed", "Triplebyte", "UCLA Health", "Verizon"], "y": [16, 10, 11, 67, 10, 11, 10, 11, 18]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "white", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "white", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "#C8D4E3", "linecolor": "#C8D4E3", "minorgridcolor": "#C8D4E3", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "#C8D4E3", "linecolor": "#C8D4E3", "minorgridcolor": "#C8D4E3", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "white", "showlakes": true, "showland": true, "subunitcolor": "#C8D4E3"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "white", "polar": {"angularaxis": {"gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": ""}, "bgcolor": "white", "radialaxis": {"gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "white", "gridcolor": "#DFE8F3", "gridwidth": 2, "linecolor": "#EBF0F8", "showbackground": true, "ticks": "", "zerolinecolor": "#EBF0F8"}, "yaxis": {"backgroundcolor": "white", "gridcolor": "#DFE8F3", "gridwidth": 2, "linecolor": "#EBF0F8", "showbackground": true, "ticks": "", "zerolinecolor": "#EBF0F8"}, "zaxis": {"backgroundcolor": "white", "gridcolor": "#DFE8F3", "gridwidth": 2, "linecolor": "#EBF0F8", "showbackground": true, "ticks": "", "zerolinecolor": "#EBF0F8"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""}, "baxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""}, "bgcolor": "white", "caxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#EBF0F8", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#EBF0F8", "zerolinewidth": 2}}}, "title": {"text": "Companies with 10 or more Data Science job postings"}, "xaxis": {"tickangle": -90}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('f71ac915-9c53-4f71-8339-34e0774bdced');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


Interestingly, Deloitte tops the chart by quite some distance. The company has 67 open spots in these prime locations. The next nearest competitors seem to be Verizon and Amazon, with 18 and 16 job postings respectively.


 *What are the salaries offered by these highly active companies?*


```python
## Out of the 9 companies with highest n.o of job postings, which company is offering the highest salary on average
plot4 = df_sal_no_dup.loc[df_sal_no_dup['Company'].isin(plot3['Company'].unique()), :]
plot4 = plot4[['Company','Annual_Max_Salary']].groupby('Company').agg('mean').reset_index().rename(columns = {'Annual_Max_Salary' : 'Average Salary'})
data_bar2 = [go.Bar(x = plot4['Company'], y = plot4['Average Salary'], name = 'company_mean_sal_barplot', marker = dict(color = '#FF7F0E'), width = 0.5)]
layout = go.Layout(title = 'Average Salary offered by Companies with 10 or more Data Science job postings', xaxis_tickangle = -90)
fig = go.Figure(data = data_bar2, layout = layout)
fig.layout.template = 'plotly_white'
pyo.iplot(fig)
```


<div>
        
        
            <div id="472d90d8-7ec4-48fc-b83b-519a63cc22be" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("472d90d8-7ec4-48fc-b83b-519a63cc22be")) {
                    Plotly.newPlot(
                        '472d90d8-7ec4-48fc-b83b-519a63cc22be',
                        [{"marker": {"color": "#FF7F0E"}, "name": "company_mean_sal_barplot", "type": "bar", "width": 0.5, "x": ["Amazon.com Services LLC", "Booz Allen Hamilton", "Capital One - US", "Deloitte", "National Security Agency", "Seen by Indeed", "Triplebyte", "UCLA Health", "Verizon"], "y": [107125.0, 106098.0, 93078.36363636363, 110016.76119402985, 103467.6, 113601.27272727272, 129000.0, 105419.0, 93098.77777777778]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "white", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "white", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "#C8D4E3", "linecolor": "#C8D4E3", "minorgridcolor": "#C8D4E3", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "#C8D4E3", "linecolor": "#C8D4E3", "minorgridcolor": "#C8D4E3", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "white", "showlakes": true, "showland": true, "subunitcolor": "#C8D4E3"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "white", "polar": {"angularaxis": {"gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": ""}, "bgcolor": "white", "radialaxis": {"gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "white", "gridcolor": "#DFE8F3", "gridwidth": 2, "linecolor": "#EBF0F8", "showbackground": true, "ticks": "", "zerolinecolor": "#EBF0F8"}, "yaxis": {"backgroundcolor": "white", "gridcolor": "#DFE8F3", "gridwidth": 2, "linecolor": "#EBF0F8", "showbackground": true, "ticks": "", "zerolinecolor": "#EBF0F8"}, "zaxis": {"backgroundcolor": "white", "gridcolor": "#DFE8F3", "gridwidth": 2, "linecolor": "#EBF0F8", "showbackground": true, "ticks": "", "zerolinecolor": "#EBF0F8"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""}, "baxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""}, "bgcolor": "white", "caxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#EBF0F8", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#EBF0F8", "zerolinewidth": 2}}}, "title": {"text": "Average Salary offered by Companies with 10 or more Data Science job postings"}, "xaxis": {"tickangle": -90}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('472d90d8-7ec4-48fc-b83b-519a63cc22be');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


There is not much variance in the salaries offered by these companies with highest number of job postings. You would expect this as the most aggressive companies would want to push for a competitive salary, as they try to attract best talent.

The above graphs help us better understand the data that we are dealing with. However, they also provide some insights.
For example, some job postings are outside the main cities. Hence, they need to be mapped to the nearest city.
Hence, we would need to create an additional column that maps these job postings to the nearest metropolitan city.


```python
# Checking n.o of unique cities
df_sal_no_dup['City'].unique()

# df_sal_no_dup.loc[df_sal_no_dup['City'] == 'Davidson', :]

# Grouping the cities
# if New York, Brooklyn, Jersey City, Fort Lee then NYC
# if Fort Mill, Huntersville, Davidson, Charlotte then Charlotte
# if San Rafael, Oakland, Walnut Creek, San Francisco then San Francisco
# Boston
# if Burbank, Torrance, Woodland Hills, Cypress then Los Angeles
# if Fort Meade, Hyattsville, Arlington, Greenbelt then Washington

city_mapping = {'New York': 'New York City', 'Brooklyn' : 'New York City', 'Jersey City' : 'New York City', 'Fort Lee' : 'New York City',
                'Fort Mill':'Charlotte','Huntersville':'Charlotte','Davidson':'Charlotte',
                'San Rafael':'San Francisco','Oakland':'San Francisco','Walnut Creek':'San Francisco',
                'Burbank':'Los Angeles','Torrance':'Los Angeles','Woodland Hills':'Los Angeles', 'Cypress':'Los Angeles',
                'Fort Meade':'Washington','Hyattsville':'Washington','Arlington':'Washington','Greenbelt':'Washington'
        }
# print(city_mapping)
```


```python
# Re-mapping to major cities
df_sal_no_dup['Major_City'] = df_sal_no_dup['City']
df_sal_no_dup = df_sal_no_dup.replace({'Major_City': city_mapping})
# df_sal_no_dup.loc[df_sal_no_dup['Major_City'].isnull(),:]
```

## 5. Future Scope 

1. Perform in-depth inferential analysis on data science job market statistics and make recommendations based on job location, type, salary and company.
2. Create a predictive model that predicts whether a particular candidate will accept or reject the offer based on various features associated with a company's job posting.
3. Rank different companies based on features provided by the applicant in order to optimise job application process and identify most suitable companies
