FROM python:3.8-slim
RUN apt-get update && \
    apt-get install -y wget curl openssh-client && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pandas numpy matplotlib ipython scikit-learn seaborn xgboost hyperopt
