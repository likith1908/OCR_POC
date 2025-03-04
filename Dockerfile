# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF, Pillow, and others
RUN pip install --no-cache-dir -r requirements.txt

# Copy requirements file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directories for outputs
RUN mkdir -p output_images uploads

# Expose the port Flask will run on
EXPOSE 5000

# Set environment variable for Flask (optional, can be overridden)
ENV FLASK_APP=app.py

# Command to run the Flask app with Gunicorn (production-ready)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]