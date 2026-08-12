import streamlit as st
import pandas as pd
import joblib
import subprocess
import sys

st.write("Pandas OK")


st.text(subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode())
