FROM python:3.11-slim

WORKDIR /app
COPY ./app.py /app/
COPY ./requirements.txt /app/
COPY ./data /app/data
COPY ./pages /app/pages
COPY ./models /app/models
COPY ./.streamlit /app/.streamlit

RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["python3", "-m", "streamlit", "run", "app.py"]
