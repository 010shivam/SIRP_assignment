import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt,welch
import argparse
import os
import pickle
parser = argparse.ArgumentParser(description="Inputs Directoreies")

parser.add_argument(
    "-in_dir",
    type=str,
    required=True,
    help="Input Directory of datasets"
)

parser.add_argument(
    "-out_dir",
    type=str,
    required=True,
    help="Output directory"
)
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir

#Filtering Datasets 

def filtering(fs,lowcut,highcut,in_dir,file):
    nasal = pd.read_csv(f"{in_dir}/{file}/nasal_airflow.csv")
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    b,a = butter(N=4,Wn=[low,high],btype ='bandpass')
    signal =nasal['nasal_airflow'].values # signal array
    timestamp= nasal['Timestamp'].values
    # print(b,a)
    filtered_signal = filtfilt(b,a,signal)
    return filtered_signal,timestamp

participants_files = os.listdir(in_dir)

# Windowing and Labelling

#Converting timestamp to datetime format for convinience

def create_windows(window_size,step,timestamp,filtered_signal):
    timestamp = pd.to_datetime(timestamp,format="%d.%m.%Y %H:%M:%S,%f")
    timestamp =timestamp.values
    windows =[]
    window_time =[]
    for i in range(0,len(filtered_signal)- window_size,step):
        start = timestamp[i]
        end = timestamp[i+window_size-1]
        windows.append(filtered_signal[i:i+window_size])
        window_time.append((start,end))

    windows =np.array(windows)
    window_time = np.array(window_time)
    return windows, window_time

def create_labels(flow_events,window_time):
    event_labels =[]
    events_np = flow_events.to_numpy()
    for i in range(len(window_time)):
        win_start,win_end =  window_time[i]
        win_duration = win_end- win_start

        max_overlap =0
        event_label ="Normal"

        for j in range(len(events_np)):
            curr_event = flow_events.iloc[j]
            overlap = min(curr_event['end'],win_end)-max(curr_event['start'],win_start)
            overlap_ratio = overlap/win_duration
                
            if overlap_ratio > max_overlap:
                    max_overlap=overlap_ratio
                    event_label = curr_event["Events"]
        if max_overlap>0.5:
            event_labels.append(event_label)
        else:
            event_labels.append("Normal")
    return event_labels

timestamp = []
filtered_signal = []
fs=32
lowcut =0.17
highcut =0.4
ts = 30 # time span
window_size = fs*ts # Calulating the window timespan
step =int(0.5*window_size) #The overlap is 50% 
windows = []
window_time = []
event_labels = []
participants =[]
for file in participants_files:
    new_signal,new_timestamp=filtering(fs,lowcut,highcut,in_dir,file)
    
    new_windows,new_window_time = create_windows(window_size,step,new_timestamp,new_signal)
    n = len(new_windows)
    new_part = [file]*n

    flow_events = pd.read_csv(f'{in_dir}/{file}/flow_events.csv')
    flow_events['start'] =pd.to_datetime(flow_events['Timerange'].str.split(" ").str[0] + " " + flow_events['Timerange'].str.split(" ").str[1].str.split("-").str[0],format="%d.%m.%Y %H:%M:%S,%f")
    flow_events['end'] =pd.to_datetime(flow_events['Timerange'].str.split(" ").str[0] + " " + flow_events['Timerange'].str.split(" ").str[1].str.split("-").str[1],format="%d.%m.%Y %H:%M:%S,%f")
    flow_events.drop(columns={'Timerange'},inplace=True)
    
    new_event_labels = create_labels(flow_events,new_window_time)

    timestamp =np.concatenate((timestamp,new_timestamp),axis=0)
    filtered_signal =np.concatenate((filtered_signal,new_signal),axis=0)

    if len(windows)==0:
        windows=new_windows
    else:
        windows =np.concatenate((windows,new_windows),axis=0)
    
    if len(window_time)==0:
        window_time=new_window_time
    else:
        window_time =np.concatenate((window_time,new_window_time),axis=0)
    
    event_labels =np.concatenate((event_labels,new_event_labels),axis=0)
    participants =np.concatenate((participants,new_part),axis=0)
    
print(len(windows),len(window_time),len(event_labels))
print("Creating CSV file...")
breathing_dataset = pd.DataFrame({"timestamp":timestamp,"airflow":filtered_signal})
breathing_dataset.to_csv(f"{out_dir}/breathing_dataset.csv")
print("Breathing_dataset.csv created")
print("Creating Pickle file")
with open(f"{out_dir}/breathing_dataset.pkl","wb") as f:
    pickle.dump(breathing_dataset,f)

print("Breathing_dataset.pkl created")
print("Creating CSV file...")
sleep_stage_dataset = {"window_time":window_time,"signals":windows,"labels":event_labels,"participants":participants}
with open(f"{out_dir}/sleep_stage_dataset.pkl","wb") as f:
    pickle.dump(sleep_stage_dataset,f)
print("sleep_stages_dataset.pkl created")