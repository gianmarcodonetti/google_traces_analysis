# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:14:27 2015

Merge the traces file into a single one, given a certain window of analysis

@author: Giammi
"""

# SETUP ==========================================================================================
import pandas as pd
import os
import numpy as np

'''
    Here you can insert the path where you are working
    At this location should be all the "part-xxxxx-of-00500.csv" files
'''
path = input("\nPlease, insert the complete path where I can find all the part.csv files: ")
os.chdir(path)

schema = "schema.csv"
schema = pd.read_csv(schema, delimiter=',')

begin = "part-"
end = "-of-00500.csv"

jobs_folder = "job_events/"
tasks_folder = "task_events/"

first_part = 0 # Starting part to retrieve
last_part = 50 # Last part
number_of_digits = 5

# DEFINING SOME FUNCTIONS ========================================================================
# Read all the jobs file in the proper folder
def readJobs():
    jobs = []
    for i in range(first_part,last_part):
        part = str(i)
        while(len(part) < number_of_digits ):
            part = "0" + part
        file = jobs_folder + begin + part + end
        jobs.append( pd.read_csv(file, delimiter=',', header = None) )
        jobs[i].columns = schema.content[:8].tolist()
        jobs[i].drop(jobs[i].columns[[1, 4, 6, 7]], axis=1, inplace=True)
    return jobs


# Read all the tasks file in the proper folder
def readTasks():
    tasks = []
    for i in range(first_part,last_part):
        part = str(i)
        while(len(part) < number_of_digits ):
            part = "0" + part
        file = tasks_folder + begin + part + end
        tasks.append( pd.read_csv(file, delimiter=',', header = None) )
        tasks[i].columns = schema.content[8:21].tolist()
        tasks[i].drop(tasks[i].columns[[1, 4, 6, 9, 10, 11, 12]], axis=1, inplace=True)
    return tasks


# Traduce a number specified in 'char' to a microseconds number
def microseconds(n, char):
    if char == 's':
        return n * 10**6
    elif char == 'm':
        return n * 10**6 * 60
    elif char == 'h':
        return n * 10**6 * 60 * 60  
    elif char == 'd':
        return n * 10**6 * 60 * 60 * 24
    else:
        return -1

# Only the files with the traces in the window will remain
def deleteFilesOutsideWindow(files, opening, closing):
    copy = files[:]
    i = 0
    opening_found = False
    while not opening_found:
        if max( copy[i].time.tolist() ) >= opening:
            opening_found = True
        else:
            copy.pop(i)
    closing_found = False
    while not closing_found and i < len(copy):
        if max( copy[i].time.tolist() ) > closing:
            closing_found = True
        i += 1
    while i < len(copy):
        copy.pop(i)
    return copy

# Given a list of files, a starting and an ending point, returns a DataFrame of this window
def mergeFiles(files, opening, closing):
    copy = files[:]
    opening_found = False
    first = copy[0]    
    i = int(np.median(first.index))
    step = int((len(first.index))/4)
    while not opening_found and i >= 1:
        if int(first[first.index == i].time) < opening:
            i = int(i + step)
            if step != 1:
                step = int((step)/2)
        elif int(first[first.index == i].time) >= opening and int(first[first.index == i-1].time) < opening:
            opening_found = True
        elif int(first[first.index == i].time) > opening:  
            i = int(i - step)
            if step != 1:
                step = int((step)/2)
        else:
            i -= 1
    first = first.iloc[i:]
    copy[0] = first

    closing_found = False
    last = copy[len(copy)-1]
    i = int(np.median(last.index))
    step = int(len(last.index)/4)
    while not closing_found and i >= 1 and i <= last.index[-1]:
        if int(last[last.index == i].time) < closing:
            i = int(i + step)
            if step != 1:
                step = int((step)/2)
        elif int(last[last.index == i].time) >= closing and int(last[last.index == i-1].time) < closing:
            closing_found = True
        elif int(last[last.index == i].time) > closing:  
            i = int(i - step)
            if step != 1:
                step = int((step)/2)
        else:
            i -= 1
    last = last.iloc[:i-last.index[0]]
    copy[len(copy)-1] = last
    return pd.concat(copy, ignore_index=True)

# WORKING ========================================================================================
while True:
    work = input("\nPlease, type 'j' whether we're merging jobs files, 't' in case of tasks: ")
    if work in ['j', 't']:
        break
if work == 'j':
    jobs = readJobs()
else:
    tasks = readTasks()


'''
    Now, choose the first day to analyse and the first day not to analyse.
    In this way, the output file would consider all the events between startDay (included)
    and endDay (not included).
'''
while True:
    startDay = int(input("\nPlease, choose the first day to analyse [... of May]: "))
    if startDay in range(0, 30):
        break
while True:
    endDay = int(input("\nPlease, choose the first day NOT to analyse [... of May]: "))
    if endDay in range(1, 31) and endDay > startDay:
        break

start = microseconds(600, 's')
may2 = microseconds(5, 'h') + start
if startDay == 0:
    start_day = 0
else:
    start_day = start
    if startDay > 1:
        start_day = may2
        for i in range (0, startDay - 2):
            start_day += microseconds(24, 'h')
end_day = start
if endDay > 1:
    end_day = may2
    for i in range (0, endDay - 2):
        end_day += microseconds(24, 'h')

inter = deleteFilesOutsideWindow(jobs, start_day, end_day)
final = mergeFiles(inter, start_day, end_day)

if work == 't':
    final.drop(final.columns[[4, 5]], axis=1, inplace=True)

# STORE THE RESULT ===============================================================================
path = input("\nPlease, insert the complete path where I can save the result: ")
os.chdir(path)

filename = "job_events_May"+str(startDay)
if endDay != startDay + 1:
    filename += "May"+str(endDay-1)
filename += "_4columns.csv"
final.to_csv(filename, index = False)

print("\nI have just created a csv file with the result, please check: "+filename)