FROM python:3.9-slim-bullseye

# Debian 11 (bullseye) reached end of life: its packages moved to archive.debian.org,
# the default deb.debian.org mirrors return 404 for many of them.
RUN printf '%s\n' \
        'deb http://archive.debian.org/debian bullseye main' \
        'deb http://archive.debian.org/debian bullseye-updates main' \
        'deb http://archive.debian.org/debian-security bullseye-security main' \
        > /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/*

# Install OpenJDK 11 (available in Debian repos, required by PySpark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-11-jdk-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/lib/jvm/java-11-openjdk-* /usr/lib/jvm/java-11-openjdk
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk

ARG PYSPARK_VERSION=3.2.0
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION}

WORKDIR /app

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
