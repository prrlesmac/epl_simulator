#!/bin/bash

# Start cron in the background
cron -f &

# Start your Python app
python src/app/app.py
