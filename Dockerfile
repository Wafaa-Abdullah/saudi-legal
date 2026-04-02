FROM python:3.11
WORKDIR /code

# 👈 السطر ده هو السحر اللي هيخلي الرسايل تظهر فوراً
ENV PYTHONUNBUFFERED=1 

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]