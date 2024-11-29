FROM python:3.11

WORKDIR /app
COPY . .
RUN pip3 install --upgrade pip && python3 -m venv venv && . venv/bin/activate && pip3 install --upgrade pip && pip3 install -r requirements.txt

CMD ["uvicorn" , "main:app"]