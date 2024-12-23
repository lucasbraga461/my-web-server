FROM jenkins/jenkins:lts

USER root
RUN usermod -u 1001 jenkins && \
    groupmod -g 1001 jenkins

# Install Python, pip, and other necessary tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    unzip \
    groff \
    less

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -f awscliv2.zip

# Create a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --upgrade pip

# Copy your application's requirements and install them
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of your application
COPY . .

# Switch back to the jenkins user
USER jenkins

# Expose the default Jenkins port
EXPOSE 8080
