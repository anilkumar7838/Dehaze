# Use an official Python runtime as a parent image
FROM python:3.13

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (like libjpeg for image handling and other necessary packages)
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
