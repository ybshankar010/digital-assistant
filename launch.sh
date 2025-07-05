#!/bin/bash
export STREAMLIT_WATCH_DISABLE_MODULES=1
export PYTHONPATH=$PYTHONPATH:/home/bhavani/Desktop/code/digital-assistant

streamlit run digital_assistant/ui/app.py
