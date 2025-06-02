FROM python:3.10.9

# Install cron and any other dependencies
RUN apt-get update && apt-get install -y cron

# Set working directory
WORKDIR /

# Copy app code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .

# Add crontab file in the cron directory
COPY crontab /etc/cron.d/simulator-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/simulator-cron
# Create the log file to be able to run tail
RUN /usr/bin/crontab /etc/cron.d/crontab

CMD ["cron", "-f"]
