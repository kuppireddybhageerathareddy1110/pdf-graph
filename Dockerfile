FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pymupdf spacy sentence-transformers \
    networkx pyvis numpy pandas scikit-learn \
    streamlit \
    && python -m spacy download en_core_web_sm

# Note: PyTorch/PyG install is hardware-specific.
# Uncomment one of the following:
# CPU:
# RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# RUN pip install torch-geometric
# CUDA 12.1:
# RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
# RUN pip install torch-geometric

# Copy source
COPY . .

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
