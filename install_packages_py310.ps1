# Port Intelligence — Python 3.10 Package Installer
# Run this in PowerShell AFTER installing Python 3.10 from python.org
# Usage: python3.10 -m pip install ... OR py -3.10 -m pip install ...

$PY = "py -3.10"  # Change to "python3.10" if needed

Write-Host "=== Installing PyTorch with CUDA 12.8 ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128"

Write-Host "=== Installing CLIP from GitHub ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install git+https://github.com/openai/CLIP.git"

Write-Host "=== Installing TensorFlow ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install tensorflow==2.12.0"

Write-Host "=== Installing core ML packages ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install xgboost==2.1.4 lightgbm==4.6.0 scikit-learn==1.6.1 shap==0.49.1"

Write-Host "=== Installing data packages ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install pandas==2.3.3 numpy==2.0.2 scipy==1.13.1 pyarrow==18.1.0"

Write-Host "=== Installing API packages ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.5.0 pydantic-settings==2.1.0 redis==7.0.1 psycopg2-binary==2.9.9 sqlalchemy==2.0.23"

Write-Host "=== Installing NLP/AI packages ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install transformers==4.47.0 sentence-transformers==3.3.1 langchain==0.2.14 langchain-community==0.2.12 langchain-core==0.3.29 openai==1.42.0 huggingface-hub==0.26.5"

Write-Host "=== Installing visualization ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install matplotlib==3.9.2 seaborn==0.13.1 plotly==6.6.0 streamlit==1.50.0"

Write-Host "=== Installing Jupyter ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install jupyterlab==3.2.5 ipykernel==6.6.1 nbformat==5.1.3 nbconvert==6.4.0"

Write-Host "=== Installing remaining packages ===" -ForegroundColor Cyan
Invoke-Expression "$PY -m pip install ``
  aiofiles==23.1.0 aiohttp==3.9.3 alembic==1.12.1 altair==5.5.0 ``
  beautifulsoup4==4.12.2 biopython==1.83 ``
  cryptography==46.0.3 ``
  fastai==1.0.61 filelock==3.13.1 freezegun==1.4.0 ``
  gensim==4.3.1 ``
  httpx==0.25.2 ``
  imageio==2.34.1 ``
  joblib==1.2.0 ``
  keras==2.12.0 ``
  langsmith==0.2.10 librosa==0.10.0.post2 ``
  lxml==5.1.0 ``
  networkx==3.2.1 nltk==3.8.1 numba==0.60.0 ``
  opencv-python==4.5.5.62 openpyxl==3.1.2 ``
  passlib==1.7.4 pillow==10.3.0 psycopg2-binary==2.9.9 ``
  pyarrow==18.1.0 pydub==0.25.1 pytesseract==0.3.10 ``
  pytest==7.4.3 pytest-asyncio==0.21.1 ``
  python-dotenv==1.0.0 python-jose==3.3.0 python-multipart==0.0.6 ``
  PyYAML==6.0.1 ``
  redis==7.0.1 requests==2.31.0 rich==13.7.0 ruff==0.1.15 ``
  scikit-image==0.22.0 scikit-video==1.1.11 ``
  sounddevice==0.4.6 soundfile==0.12.1 SpeechRecognition==3.10.4 ``
  sympy==1.13.3 ``
  tabulate==0.9.0 timm==1.0.24 ``
  websockets==12.0 ``
  xgboost==2.1.4 xmltodict==0.13.0 yfinance==0.2.37"

Write-Host ""
Write-Host "=== Installation complete! ===" -ForegroundColor Green
Write-Host "Verify with: py -3.10 -m pip list" -ForegroundColor Yellow
