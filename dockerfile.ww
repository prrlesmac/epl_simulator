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

# Create log files and set permissions
RUN mkdir -p /var/log
RUN touch /var/log/simulator.log
RUN chmod 666 /var/log/simulator.log

# Copy the cron job file into the right place
COPY crontab /etc/cron.d/simulator-cron
RUN chmod 0644 /etc/cron.d/simulator-cron
# No need to run `crontab` command here for files in /etc/cron.d/

# Expose the port your app uses
EXPOSE 8050

# Use a shell script to start cron in foreground and your app concurrently
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
