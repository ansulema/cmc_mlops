FROM pytorch/torchserve:latest

# Install transformers
RUN pip install --no-cache-dir transformers

# Copy model archive
COPY model-store /home/model-server/model-store

# Copy config
COPY config.properties /home/model-server/config.properties

# Expose ports
EXPOSE 8080 8081 8082

# Run TorchServe
CMD ["torchserve", \
     "--start", \
     "--model-store", "/home/model-server/model-store", \
     "--models", "spam_classifier=spam_classifier.mar", \
     "--ts-config", "/home/model-server/config.properties", \
     "--foreground"]
