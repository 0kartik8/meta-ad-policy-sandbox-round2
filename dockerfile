# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy all your project files into the container
COPY . .

# Install dependencies directly from the new pyproject.toml
RUN pip install --no-cache-dir .

# Expose the port Uvicorn uses
EXPOSE 8000

# Start the server, pointing it to the new folder structure!
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]