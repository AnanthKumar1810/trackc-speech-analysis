FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8765
CMD ["python", "src/serve.py", "--port", "8765"]
