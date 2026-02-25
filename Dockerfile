FROM python:3.9-slim-bullseye

# Install OpenJDK 11 (available in Debian repos, required by PySpark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-11-jdk-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

ARG PYSPARK_VERSION=3.2.0
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION}

WORKDIR /app

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
