ARG PYTHON_VERSION=3.11.14-slim

FROM python:${PYTHON_VERSION}
RUN pip install flask requests
RUN mkdir -p /data
WORKDIR /app
COPY . .
EXPOSE 5001
ENTRYPOINT [ "python" ]
CMD ["app.py"]