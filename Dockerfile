FROM python:3.8-alpine
WORKDIR /app

RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev \
    libffi-dev openssl-dev\
    lapack-dev

RUN apk add make automake gcc g++ subversion python3-dev

RUN pip install --upgrade pip

COPY . /app

RUN --mount=type=cache,mode=0755,target=/root/.cache pip install -r requirements.txt
# RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--port", "8000"]