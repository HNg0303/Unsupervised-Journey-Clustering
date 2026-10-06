# Kafka Connect worker with the Debezium MongoDB source and JDBC sink plugins.
# Built from the official Apache Kafka image so it only needs Docker Hub and
# Maven Central (quay.io/debezium images are not required).
ARG KAFKA_IMAGE=apache/kafka:4.1.0
FROM ${KAFKA_IMAGE}

ARG DEBEZIUM_VERSION=3.6.3.Final
ARG MAVEN=https://repo1.maven.org/maven2/io/debezium

USER root
RUN mkdir -p /opt/connect-plugins \
 && cd /opt/connect-plugins \
 && for name in mongodb jdbc; do \
      wget -q "${MAVEN}/debezium-connector-${name}/${DEBEZIUM_VERSION}/debezium-connector-${name}-${DEBEZIUM_VERSION}-plugin.tar.gz" -O plugin.tgz \
      && tar -xzf plugin.tgz && rm plugin.tgz; \
    done \
 && chown -R appuser:appuser /opt/connect-plugins

COPY connect-distributed.properties /opt/connect/connect-distributed.properties
USER appuser

EXPOSE 8083
CMD ["/opt/kafka/bin/connect-distributed.sh", "/opt/connect/connect-distributed.properties"]
