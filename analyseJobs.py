# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:53:34 2015

@author: Giammi
"""

# SETUP ==========================================================================================
import pandas as pd
import sys
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

event_codes = {
    '0' : 'submit',
    '1' : 'schedule',
    '2' : 'evict',
    '3' : 'fail',
    '4' : 'finish',
    '5' : 'kill',
    '6' : 'lost'
}
finishes = ['2', '3', '4', '5', '6']

# DEFINING SOME FUNCTIONS =======================================================================
def showExecTime(startPoint, initialString = ""):
    eex = time.time()
    seconds = round(eex - startPoint, 2)
    minutes = (seconds/60)
    hours = int(minutes/60)
    minutes = int(minutes % 60)
    seconds = round(seconds % 60, 2)
    print("\n- "+initialString+" Execution time: %sh %sm %ss -" % (hours, minutes, seconds))

# Create a dictionary for the events_window.
# Each entry of this dictionary is a dictionary itself, containing all the events for each job.
# The jobID is the key
def createJobsDict(events_window, event1, event2):
    jobs = {}
    ended_diff = 0
    print("Creating the full dictionary...")
    events = events_window.loc[(events_window['eventType'] == int(event1)) |
                                (events_window['eventType'] == int(event2))]
    for i in finishes:
        if i != event1 and i != event2:
            badSet = events_window.loc[(events_window['eventType'] == int(i))].jobID
            ended_diff += len(badSet)
            for badJob in badSet:
                events = events.loc[events['jobID'] != badJob]
    
    length = len(events.index)
    p = 0
    begin = time.time()
    for i in events.index:
        
        # Printing the percentage of work completion
        p += 1
        perc = p / length * 100
        perc = round(perc, 1)
        sys.stdout.write("\r  "+str(perc)+" %")
        #sys.stdout.flush()
        
        cell = events[events.index == i]        
        
        jobID = int(cell.jobID)
        key = str(jobID)
        if key not in jobs:
            # I create a dictionary with all the info for each job
            jobs[key] = {}
            schedClass = int(cell.schedulingClass)           
            jobs[key]['schedulingClass'] = schedClass
        timeStamp = int(int(cell.time))
        event = int(cell.eventType)
        
        #jobs[jobID][event_codes[event]] = timeStamp
        
        jobs[key][str(event)] = timeStamp
    
    showExecTime(begin, "Dictionary created.")
    return jobs, ended_diff

# Build a dictionary with all the job that has the start event and the end event.
# The user set this two parameters.
# Return also a list containing all the times (in seconds) to perform such actions.
def cleanJobsDict(jobs, end, start = '0'):
    res = {}
    times = []
    started_before = 0
    ended_after = 0
    for key in jobs:
        d = jobs[key]
        if start in d and end in d:
            res[key] = jobs[key]
            time = (res[key][end] - res[key][start]) / 10**6
            times.append(time)
        elif start not in d and end in d:
            started_before += 1
        else: #start in d and end not in d
            ended_after += 1
    return res, times, started_before, ended_after

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

# Given a list of files, a starting and an ending point, returns a DataFrame of this window
def interestingWindow(files, opening, closing):
    #begin = time.time()
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
    #showExecTime(begin, "Window found.")
    return pd.concat(copy, ignore_index=True)


# LET'S DO THE MATH ==============================================================================
defaultInput = "C:/Users/Giammi/OneDrive/Università/Computer Systems Performance Evaluation"
defaultInput += "/progetto/traces/job_events_May2_4columns.csv"

defaultInput = "job_events_May2_4columns.csv"

# Setting arguments required by our scrpit 
parser = argparse.ArgumentParser(description="Parse the given csv file.\n" +
	"Find statistics about the duration between two specified events about JOBS.\n"+
         "By default, analyse May 2nd.\n")

parser.add_argument('-i', '--input', type=str, default=defaultInput,
                    help="name/path of the .csv file")
parser.add_argument('--start', type=int, default=0,
                    help="event type to consider as starting point")
parser.add_argument("end", type=int,
                    help="event type to consider as ending point "+str(finishes))
parser.add_argument('-g', '--granularity', type=float, default=1,
                    help="granularity espressed in hours (default 1)")

parser.add_argument('--startDay', type=int, default=2,
                    help="the first day of analysis (default 2)")
parser.add_argument('--endDay', type=int, default=3,
                    help="the first day NOT of analysis (default 3)")


# Parsing the arguments
args = parser.parse_args()
filename = args.input
event_start = str(args.start)
event_end = str(args.end)
granularity = args.granularity

startDay = args.startDay
endDay = args.endDay

#print('\n',type(filename), type(event_start), type(event_end), type(granularity))

job_events = filename
job_events = pd.read_csv(job_events, delimiter=',')
job_events.columns = ['time', 'jobID', 'eventType', 'schedulingClass']

event_chosen = event_end

''' Computation '''
sex = time.time()

step = int(microseconds(granularity, 'h'))

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

analysis_frame = (end_day - start_day)/ 10**6 / 60 / 60 # in hours

jobs_dict = []
jobs_ok, times, started_before, ended_differently, ended_after = [], [], [], [], []
l = [] # interestingWindow needs a list of files
l.append(job_events)
for i in range(0, int(analysis_frame/granularity)):
    # For each different window in the day
    print('\nWindow',i+1,'of',int(analysis_frame/granularity))
    # calculate the starting and the ending point
    opening = start_day + i * step
    closing = opening + step
    # Get the interesting window
    window = interestingWindow(l, opening, closing)
    # Creating a dictionary for this window
    dic, ed = createJobsDict(window, event_start, event_chosen)
    ended_differently.append(ed)
    jobs_dict.append(dic)
    # Cleaning and analyzing the window
    ok, t, sb, ea = cleanJobsDict(dic, event_chosen, event_start)
    jobs_ok.append(ok)
    times.append(t)
    started_before.append(sb)
    ended_after.append(ea)

sys.stdout.flush()


''' Save the results '''
data = times

granularity = str(granularity)
#==============================================================================
# pickle.dump(times, open('job_times_granularity'+(granularity), 'wb'))
#==============================================================================

stats_df = pd.DataFrame(columns = ['window', 'n', 'started_before', 'ended_differently',
                                   'ended_after', 'mean', 'median', 'stdDev', 'var',
                                   'minimum', 'maximum'])
for i in range(0, len(data)):
    if len(data[i]) != 0:
        n = len(data[i])
        sb = started_before[i]
        ed = ended_differently[i]
        ea = ended_after[i]
        mean = np.mean(data[i])
        median = np.median(data[i])
        stdDev = np.std(data[i])
        var = np.var(data[i])
        minimum = np.min(data[i])
        maximum = np.max(data[i])
        stats_df.loc[len(stats_df)] = [i, n, sb, ed, ea, mean, median, stdDev, var,
                                         minimum, maximum]
stats_df.to_csv('stats_df_job_granularity'+str(granularity)+'.csv', index = False)
print("File csv created.")


# Show the execution time ========================================================================
showExecTime(sex, "Analysis done.")


''' Showing some plots '''
plt.figure()
plt.boxplot(data,notch=True,showmeans=True,whis=1.5)

rangemax = 12000
rangemin = 0
plt.figure()
for i in range(0, len(data)):
    if len(data[i]) != 0:
        stats.describe(data[i])
        minvalue = min(data[i])
        maxvalue = max(data[i])
        mean = np.mean(data[i])
        var = np.var(data[i])
        median = np.median(data[i])
        plt.hist(data[i],normed=True,bins=200,alpha=.4,range=(rangemin,rangemax),label=str(i))
plt.legend(loc='upper right')
plt.show()

# Some distributions =============================================================================

#==============================================================================
# # Plotting an Exponential distribution --------------------------------------
# target = 1
# beta = 1.0/target
# 
# Y = np.random.exponential(beta, 5000)
# plt.hist(Y, normed=True, bins=100, alpha=.4)
# plt.plot([0,max(Y)],[target,target],'r--')
# plt.ylim(0,target*1.1)
# plt.show()
#==============================================================================

#==============================================================================
# # Plotting a Pareto distribution --------------------------------------------
# a, m = 1.02314, 13 # shape and mode = 0.2, 13 is equal!!!!
# s = (np.random.pareto(a, 1000) + 1) * m
# 
# count, bins, _ = plt.hist(s, 100, normed=True, range=(0, 1000))
# fit = a*m**a / bins**(a+1)
# plt.plot(bins, max(count)*fit/max(fit), linewidth=0.5, color='r')
# plt.show()
#==============================================================================
'''
# exact value for Pareto
alpha = 1.02314
k = 13
mymean = k*alpha/(alpha-1)
myvar  = k**2*alpha/(alpha-2) - ( k*alpha/(alpha-1) )**2
mymean, myvar
'''
#==============================================================================
# # Plotting an Erlang distribution -------------------------------------------
# shape, scale = mean, var # mean and dispersion
# s = np.random.gamma(shape, scale, 1000)
# import scipy.special as sps
# count, bins, ignored = plt.hist(s, 100, normed=True)
# y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
# plt.plot(bins, y, linewidth=2, color='r')
# plt.show()
#==============================================================================