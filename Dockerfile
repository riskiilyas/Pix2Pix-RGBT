FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Install the package
RUN pip install -e .

# Create directories for data
RUN mkdir -p data/rgb data/thermal input output artifacts logs

# Set environment variables
ENV PYTHONPATH=/app

# Expose port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]