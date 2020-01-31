# Title: Indeed web data scraping
# Purpose: To extract data from Indeed's website using bs4 python module
# Created by: Saani Rawat
# Last modifed: 01/30/2020
# Output:
# 1.data_science_jobs_df_test.csv

# Check to see if website allows data scraping do - <<website>>/robots.txt

from bs4 import  BeautifulSoup
import requests
import pandas as pd


# https://www.indeed.com/jobs?q=data+scientist&l=Charlotte%2C+NC
# https://www.indeed.com/jobs?q=data+scientist&l=San+Francisco%2C+CA
# https://www.indeed.com/jobs?q=data+scientist&l=New+York%2C+NY
# https://www.indeed.com/jobs?q=data+scientist&l=Boston%2C+MA
# https://www.indeed.com/jobs?q=&l=Los+Angeles%2C+CA
# https://www.indeed.com/jobs?q=&l=Washington%2C+DC

# Data Science Job postings in New York City, Charlotte, San Francisco, Boston, Los Angeles and Washington
cities = ["New+York", "Charlotte", "San+Francisco", "Boston", "Los Angeles", "Washington"]
states = ["NY", "NC", "CA", "MA", "CA", "DC"]

jobs_info = []
for city, state in zip(cities, states):
    if city == "Charlotte":
        for i in range(10, 105, 15):
            source = requests.get("https://www.indeed.com/jobs?q=data+scientist&l="+city+"%2C+"+state+"&start="+str(i)).text

            # soup = BeautifulSoup(source, "html.parser")
            soup = BeautifulSoup(source, "lxml")

            for row in range(0,len(soup.findAll("div", class_= "title"))):
                # Job title
                title = soup.findAll("div", class_= "title")[row].a.text.strip()

                # Company name
                company_name = soup.findAll("div", class_ = "sjcl")[row].span.text.strip()

                # Company address
                try:
                    company_address = soup.findAll("div", class_ = "location accessible-contrast-color-location")[row].text.strip()
                except Exception as e:
                    company_address = None

                # Salary information
                try:
                    salary = soup.findAll("span", class_ = "salaryText")[row].text.strip()
                except Exception as e:
                    salary = None

                # Summary
                try:
                    job_summary = soup.findAll("div", class_ = "summary")[row].text.strip()
                except Exception as e:
                    job_summary = None
                # job_summary = soup.findAll("div", class_ = "summary")[row].text

                jobs_info.append([title, company_name, company_address, salary, job_summary])
    else:
        for i in range(10, 1050, 15):
            source = requests.get(
                "https://www.indeed.com/jobs?q=data+scientist&l=" + city + "%2C+" + state + "&start=" + str(
                    i)).text

            # soup = BeautifulSoup(source, "html.parser")
            soup = BeautifulSoup(source, "lxml")

            for row in range(0, len(soup.findAll("div", class_="title"))):
                # Job title
                title = soup.findAll("div", class_="title")[row].a.text.strip()

                # Company name
                company_name = soup.findAll("div", class_="sjcl")[row].span.text.strip()

                # Company address
                try:
                    company_address = soup.findAll("div", class_="location accessible-contrast-color-location")[row].text.strip()
                except Exception as e:
                    company_address = None

                # Salary information
                try:
                    salary = soup.findAll("span", class_="salaryText")[row].text.strip()
                except Exception as e:
                    salary = None

                # Summary
                try:
                    job_summary = soup.findAll("div", class_="summary")[row].text.strip()
                except Exception as e:
                    job_summary = None
                # job_summary = soup.findAll("div", class_ = "summary")[row].text

                jobs_info.append([title, company_name, company_address, salary, job_summary])

data_science_jobs_df = pd.DataFrame(jobs_info)
data_science_jobs_df.to_csv("data_science_jobs_df_test.csv")
