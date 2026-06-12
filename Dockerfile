FROM python:3.10.9

ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /

# Copy app code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .
#RUN playwright install --with-deps