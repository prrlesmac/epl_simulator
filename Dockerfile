FROM python:3.10.9-slim

# Set working directory
WORKDIR /

# Copy app code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .
#RUN playwright install --with-deps