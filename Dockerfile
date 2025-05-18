# Use an official Python runtime as a parent image
FROM python:3.10.9

# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed dependencies specified in requirements.txt
RUN echo "Installing dependencies..."
RUN pip install --no-cache-dir -r requirements.txt
RUN python setup.py install

# Run the simulator
RUN python src/retriever/elos.py
RUN python src/retriever/elos.py
RUN python src/simulator/simulator.py

# Expose port 8000 (or any other port your FastAPI application listens on)
EXPOSE 8050

# Command to run the FastAPI application using uvicorn with auto-reload
CMD ["python", "src/app/app.py"]
