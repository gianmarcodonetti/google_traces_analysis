# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:13:32 2015

@author: Giammi
"""

import os
import urllib.request as URLreq
import gzip

# SETUP ===========================================================================================
path = input("\nPlease, insert the complete path where I can save the result: ")
os.chdir(path)

first_part = 0 # First part to retrieve
last_part = 20 # Last part
number_of_digits = 5

while True:
    first_part = int(input("\nPlease, choose the first part: "))
    if first_part > 0:
        break
while True:
    last_part = int(input("\nPlease, choose the last part: "))
    if last_part > first_part:
        break


# RETRIEVING ======================================================================================
'''
    Here we retrieve the needed files from the common data storage (.gz)
    In the meanwhile we also extract them to obtain the .csv.

'''

baseURL = "https://commondatastorage.googleapis.com/clusterdata-2011-2/"
endURL = "-of-00500.csv.gz"

jobsURL = "job_events/part-"
tasksURL = "task_events/part-"

machinesURL = "machine_events/part-00000-of-00001.csv.gz"
schema = "schema.csv"

gz_folder = "./gz/"
csv_folder = "./csv/"

'''Events and Taks'''
for i in range(first_part,last_part):
    # Convert into the proper form
    part = str(i)
    while( len(part) < number_of_digits ):
        part = "0" + part

    jobs_gz = jobsURL + part + endURL
    task_gz = tasksURL + part + endURL

    # Download the files of jobs and tasks
    URLreq.urlretrieve(baseURL + jobs_gz, gz_folder + jobs_gz)
    URLreq.urlretrieve(baseURL + task_gz, gz_folder + task_gz)

    # Write the csv files
    inF = gzip.open(gz_folder + jobs_gz, 'rb')
    outF = open(csv_folder + jobs_gz[:-3], 'wb')
    outF.write( inF.read() )
    inF.close()
    outF.close()

    inF = gzip.open(gz_folder + task_gz, 'rb')
    outF = open(csv_folder + task_gz[:-3], 'wb')
    outF.write( inF.read() )
    inF.close()
    outF.close()


'''Schema'''
URLreq.urlretrieve(baseURL + schema, csv_folder + schema)