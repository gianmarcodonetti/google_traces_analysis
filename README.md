***********************
## HOW TO
The programs analyseJobs.py and analyseTasks.py are the most interesting files: given a csv file, they produce statistical data and information about the duration of tasks and job.

***********************
## INPUT :
	
#### end:
	the only hard-required argument
	type = int
	This is the type of event to consider in order to end the calculation of the duration.

#### -i, --input: 
	- in case of Jobs  default = "job_events_May2_4columns.csv"
		The csv file must contain the columns "Time,JobId,Event,SchedulingClass"
	- in case of Tasks default = "task_events_May2_6columns.csv"
		The csv file must contain the columns "Time,JobId,TaskIndex,Event,SchedulingClass,Priority"

#### --start: 
	default = 0 (SUBMIT)
	type = int
	This is the type of event to consider in order to start calculating the duration.
	
#### -g, --granularity: 
		type = int
		default = 1
		The granularity espressed in hours. It indicates the width of the time window to group tasks/jobs.

#### --startDay: 
		default = 2
		type = int
		You can specify a specific day of March to consider as starting day (It should be contained in the input file)
	
#### --endDay: 
		default = 3
		type = int
		Similarly to startDay, you can specify the ending day (It should be contained in the input file)

*****************************
## OUTPUT :

The programs generate numerical data, histograms and boxplots.
At the end of the execution, two plots are shown to the user, who can modify, rescale, rename the axis and perform other simple action in the python visualization tool.
After that, he can save the plots wherever he want and, once the two images are closed, the statistical data are saved in a csv file.

*****************************
### OTHER FILES

- retrieveGoogleTraces.py:
	Useful in order to download the Google Traces files of jobs and tasks from https://commondatastorage.googleapis.com/clusterdata-2011-2/ and extract them as csv files.
	After launching it, the user has to insert the path where the results should be saved and the number related to the first and the last part to download.


- mergingTracesParts.py:
	Useful in order to put together the downloaded files.
	After launching it, the user will be asked to insert some inputs:
		1. The path where the all the "part-xxxxx-of-yyyyyy.csv" files are stored
		2. The type of elemnts to analyse ('j' for jobs, 't' for task)
		3. startDay = the first day to analyse
		4. endDay   = the first day NOT to analyse
		5. The path where the result will be stored
	In this way, the output file would consider all the events between startDay (included) and endDay (not included).

