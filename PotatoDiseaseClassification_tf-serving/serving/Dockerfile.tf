FROM tensorflow/serving

# Copy the model files to the serving container
COPY models.config.a /serving/models.config.a
COPY models/ /serving/models/

EXPOSE 8501

ENTRYPOINT ["/usr/bin/tensorflow_model_server"]
CMD [ "--rest_api_port=8501", "--model_config_file=/serving/models.config.a", "--allow_version_labels_for_unavailable_models" ]
