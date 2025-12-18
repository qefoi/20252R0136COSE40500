ARG PYTHON_VERSION=3.11.14-slim

FROM python:${PYTHON_VERSION}
RUN pip install flask requests
WORKDIR /app
COPY . .
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD ["app.py"]