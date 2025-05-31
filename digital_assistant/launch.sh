#!/bin/bash
export STREAMLIT_WATCH_DISABLE_MODULES=1
streamlit run digital_assistant/ui/app.py --server.fileWatcherType none
