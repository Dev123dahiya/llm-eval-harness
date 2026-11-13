FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY evalkit ./evalkit
COPY data ./data
RUN pip install --no-cache-dir -e .
RUN useradd --create-home appuser
USER appuser
CMD ["python", "-m", "evalkit", "demo"]
