import os
import io
import pandas as pd


def convert_csv(folder):
    files = ["nasal_airflow","sleep_profile","spo2","thoracic_movement"]
    for par in files:
            filename = f'{folder}/{par}.txt'
            with open(filename) as f:
                lines = f.readlines()
                data_index = next(
                    (i for i, line in enumerate(lines) if line.strip() == "Data:"),
                    None
                )

                if data_index is not None:
        # Data section exists → start after it
                    data_lines = lines[data_index + 1:]
                else:
                    # No "Data:" → assume file is already data
            # Remove empty lines
                    data_lines = [l for l in lines if ";" in l]
                df = pd.read_csv(
                io.StringIO("\n".join(data_lines)),
                sep=";",
                skiprows=0,
                names=["Timestamp", par])
                df.to_csv(f"{folder}/{par}.csv", index=False)

def convert_flow_events(folder):
     filename = f'{folder}/flow_events.txt'
     with open(filename) as f:
          lines = f.readlines()
          data_index = next(
                    (i for i, line in enumerate(lines) if line.strip() == "Signal Type: Impuls"),
                    None
                )
          data_lines = lines[data_index+1:]
          df = pd.read_csv(io.StringIO("\n".join(data_lines)),
            sep=";",
            skiprows=0,
            names=["Timerange","Impulse","Events","Stage"])
          df.to_csv(f"{folder}/flow_events.csv", index=False)

          
# for i in range(1,6):
#     convert_csv(f"AP0{i}",i)
#     handle_flow_events(f"AP0{i}",i)
# Read only data part