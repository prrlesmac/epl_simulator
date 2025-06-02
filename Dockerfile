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
ADD crontab /etc/cron.d/simulator-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/simulator-cron
# Create the log file to be able to run tail
RUN touch /var/log/cron.log

CMD cron && tail -f /var/log/cron.log
