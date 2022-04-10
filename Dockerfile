# FROM python:3.8-alpine
# WORKDIR /app

# RUN apk add --no-cache --update \
#     python3 python3-dev gcc \
#     gfortran musl-dev \
#     libffi-dev openssl-dev\
#     lapack-dev

# RUN apk add make automake gcc g++ subversion python3-dev

# RUN pip install --upgrade pip

# COPY . /app

# RUN --mount=type=cache,mode=0755,target=/root/.cache pip install -r requirements.txt
# # RUN pip install -r requirements.txt
# CMD ["uvicorn", "main:app", "--port", "8000"]



FROM huanjason/scikit-learn:latest
WORKDIR /app
# RUN adduser -d  dockuser
# RUN chown dockuser:dockuser -R /app/

# RUN apk add --no-cache --update \
#     python3 python3-dev gcc \
#     gfortran musl-dev \
#     libffi-dev openssl-dev\
#     lapack-dev

# RUN apk add make automake gcc g++ subversion python3-dev

# RUN pip install --upgrade pip

COPY . /app

# RUN --mount=type=cache,mode=0755,target=/root/.cache pip install -r requirements.txt

RUN python database/database_definition.py
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app","--host", "0.0.0.0" ,"--port", "8000"]