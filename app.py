import streamlit as st
import pandas as pd
import time
from datetime import datetime

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%Y%m%d")


df = pd.read_csv('attendance/attendance_' +date + '.csv')

st.dataframe(df.style.highlight_max(axis=0))