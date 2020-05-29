FROM python:3.8-slim
RUN pip install --no-cache-dir pandas numpy matplotlib ipython scikit-learn seaborn xgboost
