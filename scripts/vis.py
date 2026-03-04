import pandas as pd
import numpy as np
import argparse
from utils import convert_csv, convert_flow_events
import os
import matplotlib.pyplot as plt

parser =  argparse.ArgumentParser(description="Visualisation pdf generator")

parser.add_argument(
    "-name",
    type=str,
    required=True,
    help="Directory path"
)

args = parser.parse_args()

dir_name =  args.name
print("Converting text files to csv....")
# Converting text file to csv for convinience
convert_csv(dir_name)
convert_flow_events(dir_name)
print("Conversion of text to csv done")
# Converting to to visualisations

#Importing datasets and converting the timestamps to datetime format
nasal_flow = pd.read_csv(f'{dir_name}/nasal_airflow.csv')
thorac_mov = pd.read_csv(f'{dir_name}/thoracic_movement.csv')
spo2 = pd.read_csv(f'{dir_name}/spo2.csv')
nasal_flow['Timestamp'] =pd.to_datetime(nasal_flow['Timestamp'],format="mixed")
thorac_mov['Timestamp'] =pd.to_datetime(thorac_mov['Timestamp'],format="mixed")
spo2['Timestamp'] =pd.to_datetime(spo2['Timestamp'],format="mixed")
nasal_flow = nasal_flow.set_index('Timestamp')
thorac_mov = thorac_mov.set_index('Timestamp')
spo2 = spo2.set_index('Timestamp')

# Flow events importing and handling the timerange column
flow_events = pd.read_csv(f'{dir_name}/flow_events.csv')
flow_events['Timerange'].str.split(" ").str[1].str.split("-").str[0]
flow_events['start'] =pd.to_datetime(flow_events['Timerange'].str.split(" ").str[0] + " " + flow_events['Timerange'].str.split(" ").str[1].str.split("-").str[0],format="%d.%m.%Y %H:%M:%S,%f")
flow_events['end'] =pd.to_datetime(flow_events['Timerange'].str.split(" ").str[0] + " " + flow_events['Timerange'].str.split(" ").str[1].str.split("-").str[1],format="%d.%m.%Y %H:%M:%S,%f")


import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
name_extract = dir_name.split("/")[1]
pdf_path = f"internship/Visualizations/{name_extract}_visualization.pdf"
print("Converting Visualisation to pdfs")
with PdfPages(pdf_path) as pdf:

    window = pd.Timedelta("5min")
    start = max(nasal_flow.index.min(),spo2.index.min(),thorac_mov.index.min())
    end = min(nasal_flow.index.max(),spo2.index.max(),thorac_mov.index.max())

    event_begin =0
    event_end = 0

    current = start
    while current<end:
        n_chunk = nasal_flow.loc[current: current + window]
        t_chunk = thorac_mov.loc[current: current + window]
        s_chunk = spo2.loc[current: current + window]
        if n_chunk.empty or t_chunk.empty or s_chunk.empty:
            current += window
            continue
        while event_end< flow_events.shape[0] and flow_events.iloc[event_end]['start']< current+window:
            event_end+=1

        fig,axes= plt.subplots(3,1,figsize=(15,7),sharex=True,constrained_layout=True)

        for ax in axes:
            ax.grid(True)
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        
        axes[0].plot(n_chunk.index, n_chunk.iloc[:, 0],label="Nasal Flow")
        ymin, ymax = axes[0].get_ylim()

        for i in range(event_begin, event_end):

            start_e = flow_events.loc[i, 'start']
            end_e   = flow_events.loc[i, 'end']
            event_name = flow_events.loc[i, 'Events']

            mid = start_e + (end_e - start_e)/2

            if event_name == "Obstructive Apnea":
                color = 'red'
            else:
                color = 'yellow'

            axes[0].axvspan(start_e, end_e, color=color, alpha=0.4)

            axes[0].text(mid, ymax*0.8, event_name,
                        ha='center', va='top', fontsize=7)
        axes[0].legend()
        axes[0].set_ylabel("Nasal Airflow (L/min)")
        
        axes[1].plot(t_chunk.index, t_chunk.iloc[:, 0],color='orange',label="Thoracic/Abdominal Resp.")
        axes[1].set_ylabel("Resp. Amplitude")
        axes[1].legend()

        axes[2].plot(s_chunk.index, s_chunk.iloc[:, 0],color='grey',label="SpO2")
        axes[2].set_ylabel("SpO₂ (%)")
        axes[2].legend()
        axes[2].set_xlabel("Time")
        event_begin=event_end
        plt.suptitle(f"{current} to {current + window}")
        plt.setp(axes[-1].get_xticklabels(), rotation=90, ha='right')
        # plt.show()

       # leave space for suptitle + ticks
        pdf.savefig(fig)
        plt.close(fig)
        current += window


