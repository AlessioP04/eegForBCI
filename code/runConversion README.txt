These are scripts to convert the 109 subject BCI-2000 data (.edf) to 
EEGLAB data format (.set). 

Input data:
The input assumes that data is in a directory BCI2000 with the files 
for subject 1 in subdirectory S001 and the files for
subject 109 in the subdirectory S109. Each subject has 14 runs. The
files for the first run of subject 1 are S001R01.edf and S001R01.edf.event,
respectively.

Output data:
The output should be a root directory for the output files organized by
subject. This root directory should exist.

Event names:
	Original Schalk's datasets has only three event name, T0, T1, and T2.
	Converted EEGLAB set files have unique event name according to their task types.

	For example, S001R03.edf and S001R04.edf, both have same event name, T0, T1, and T2.
    But after conversion, S001R03.edf and S001R04.edf have different event names.
        S001R03.set has TASK1T0, TASK1T1, TASK1T2 event name,
        S001R04.set has TASK2T0, TASK2T1, TASK2T2 event name.

Below explains how to generate the unique event name.

	unique event name = task name + original event tag (T0, T1 or T2)

    task name for each run file
        BASE1 , R01 file , Baseline (eye open)
        BASE2 , R02 file , Baseline (eye closed)
        TASK1 , R03,R07,R11 files , Left Right Real
        TASK2 , R04,R08,R12 files , Left Right Imagine
        TASK3 , R05,R09,R13 files , Up Down Real
        TASK4 , R06,R010,R14 files , Up Down Imagine

To verify the result, check the converted S001R03.set file through EEGLAB.
	S001R03.set must have 30 events.
	The first event, TASK1T0 is at 4.2 seconds.
	The second event, TASK1T2 is at 8.3 seconds.
	