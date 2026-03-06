# Sleep Apnea Detection using 1D CNN

## Objective

This project(Assignment of SIRP internship) aims to visualise, process participants biomedical data and build a deep learning model to detect breathing irregularities during sleep using respiratory flow signals.  
The model classifies 30-second signal windows into **Normal**, **Hypopnea**, and **Obstructive Apnea** using a 1D Convolutional Neural Network.

## Dataset
- Overnight nasal flow, thoracic movements, spo2, sleep profile recordings from **5 participants**
- ~8 hours of respiratory flow signal per subject
- Events labeled as:
  - Normal
  - Hypopnea
  - Obstructive Apnea
- nasal flow and thoracic movement sampled at 32Hz frequency sample
- spo2 sampled at 4Hz

## Preprocessing
- Butterworth signal filtering to remove noise, basically any frequency out of range (0.17 Hz to 0.4 Hz)
- Timestamp segmented into **30-second windows** and **50% overlapping windows**
- Window labeling based on **≥50% overlap with annotated events**
- 2 Datasets are being return : Breathing_dataset.csv and sleep_stages_dataset.csv
    - Breathing Dataset.csv contains filtered signals for all participants with corresponding timestamps
    - sleep_stages_dataset.pkl contains 8800 participant id, time windows (30seconds), corresponding 960 filtered signals, corresponding label of breathing event (in total 5 classes: Normal, Hypopnea, Obstructive Apnea, Body Event, Mixed Event)

## Modelling 
A **1D Convolutional Neural Network (CNN)** was used for time-series classification.
Class Label were boiled down to 3 - Normal, Hypopnea, Obstructive Apnea
Architecture:
- Conv1D
- MaxPooling
- Conv1D
- GlobalAveragePooling
- Dense layer (Softmax output)

## Evaluation Metrics
**Leave-One-Participant-Out Cross-Validation** was used as the validation method
Performance Metrics in report:
- Accuracy
- Precision
- Recall
- Confusion Matrix

## Results
The model performance:
- Accuracy: 70.6%
- Average Recall: 60%
- Average Precision: 39%

The confusion matrix shows strong detection of apnea patterns (85% recall on obstructive apnea) while some borderline cases between normal and hypopnea remain challenging.

## Sources and Tools used
- Code was totally written by me, no use of AI tools in thinking and implementing code logic or modelling
- For Debugging purpose StackOverflow and official documentation of libraries were reffered
- For Research of sleep apnea and documentation of README.md and report.pdf, ChatGPT was used as helper with grammer and structure.