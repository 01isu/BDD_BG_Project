# Image officielle Airflow 2.9.3 sur Python 3.11
# (Airflow officiel ne supporte pas encore Python 3.12 en stable, mais ton venv local en 3.12 reste OK
#  car Airflow tourne DANS le conteneur, pas dans ton venv local)
FROM apache/airflow:2.9.3-python3.11

# On bascule en root pour installer des paquets système si besoin
USER root

# (pas de paquet système nécessaire pour notre projet)

# On revient en utilisateur airflow (obligatoire pour pip install)
USER airflow

# On copie nos dépendances Python et on les installe
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt