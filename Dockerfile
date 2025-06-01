# Use an official Python runtime as a parent image
FROM python:3.10.9

# Install cron
RUN apt-get update && apt-get install -y cron

# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Add crontab file
COPY crontab /etc/cron.d/simulator-cron
RUN chmod 0644 /etc/cron.d/simulator-cron
RUN crontab /etc/cron.d/simulator-cron

# Install any needed dependencies specified in requirements.txt
RUN echo "Installing dependencies..."
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .

RUN touch /var/log/cron.log

# Expose port 8000 (or any other port your FastAPI application listens on)
EXPOSE 8050

# Command to run the FastAPI application using uvicorn with auto-reload
CMD cron && python src/app/app.py
