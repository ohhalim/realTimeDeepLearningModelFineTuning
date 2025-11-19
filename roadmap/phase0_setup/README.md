# Phase 0: 환경 설정 (1일, 무료)

**목표**: GPU 환경 준비 및 개발 환경 세팅
**예상 시간**: 2-3시간
**비용**: 무료

---

## ✅ 체크리스트

- [ ] Kaggle 계정 생성
- [ ] Google Colab 계정
- [ ] GitHub 저장소 생성
- [ ] 로컬 Python 환경 세팅
- [ ] 첫 GPU 코드 실행 테스트

---

## 1. Kaggle 계정 생성 (5분)

### 왜 Kaggle인가?
- ✅ **무료 GPU**: 주 30시간 (P100/T4)
- ✅ **데이터셋**: 음악 MIDI 데이터 많음
- ✅ **커뮤니티**: 질문/답변 활발
- ✅ **Notebooks**: Jupyter 환경 제공

### 단계

```bash
1. https://www.kaggle.com 접속
2. "Sign Up" 클릭
3. Google 계정으로 가입 (추천)
4. Phone verification 완료
5. Profile 설정
```

### 전화 인증 완료하기 (중요!)
```
Settings → Account → Phone Verification
→ GPU 사용 위해 필수!
```

### ✅ 확인
- [ ] Kaggle 계정 생성 완료
- [ ] 전화 인증 완료
- [ ] Profile 설정 완료

---

## 2. Google Colab 계정 (5분)

### 왜 Colab인가?
- ✅ **무료 GPU**: T4 GPU (제한적)
- ✅ **Colab Pro**: $10/월 (더 좋은 GPU)
- ✅ **Google Drive 연동**: 쉬운 파일 관리
- ✅ **바로 시작**: 설치 불필요

### 단계

```bash
1. https://colab.research.google.com 접속
2. Google 계정으로 로그인
3. "+ 새 노트" 클릭
4. GPU 활성화: Runtime → Change runtime type → GPU
```

### 첫 GPU 테스트

```python
# 새 Colab 노트북에서 실행
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# 간단한 GPU 연산 테스트
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print(f"GPU computation successful!")
```

**예상 출력**:
```
PyTorch version: 2.0.1+cu118
CUDA available: True
GPU name: Tesla T4
GPU computation successful!
```

### ✅ 확인
- [ ] Colab 접속 완료
- [ ] GPU 활성화 완료
- [ ] 첫 GPU 코드 실행 성공

---

## 3. GitHub 저장소 생성 (10분)

### 저장소 이름
```
Brad-Mehldau-AI
```

### 생성 단계

```bash
1. https://github.com 접속
2. "New repository" 클릭
3. Repository name: "Brad-Mehldau-AI"
4. Description: "AI that improvises jazz piano in Brad Mehldau's style"
5. Public 선택 (포트폴리오용)
6. Add README.md 체크
7. Add .gitignore: Python 선택
8. License: MIT 선택 (추천)
9. "Create repository" 클릭
```

### 로컬에 클론

```bash
# 터미널에서
cd ~/projects
git clone https://github.com/YOUR_USERNAME/Brad-Mehldau-AI.git
cd Brad-Mehldau-AI
```

### 초기 구조 생성

```bash
# 디렉토리 생성
mkdir -p data/{raw,processed,splits}
mkdir -p models
mkdir -p training/configs
mkdir -p evaluation
mkdir -p deployment
mkdir -p samples/{phase2,phase3,final}
mkdir -p checkpoints
mkdir -p notebooks

# .gitignore 업데이트 (중요!)
cat >> .gitignore << 'EOF'

# Data
data/raw/*.mid
data/raw/*.midi
checkpoints/*.pt
checkpoints/*.pth

# Large files
*.pt
*.pth
*.ckpt
samples/*.mid
samples/*.mp3

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
EOF

# 첫 커밋
git add .
git commit -m "Initial project structure"
git push origin main
```

### ✅ 확인
- [ ] GitHub 저장소 생성
- [ ] 로컬에 클론 완료
- [ ] 프로젝트 구조 생성
- [ ] 첫 커밋 & 푸시 완료

---

## 4. 로컬 Python 환경 (30분)

### Python 버전 확인

```bash
python3 --version
# Python 3.8 이상 필요
```

### 가상환경 생성

```bash
cd ~/projects/Brad-Mehldau-AI

# venv 생성
python3 -m venv venv

# 활성화 (Mac/Linux)
source venv/bin/activate

# 활성화 (Windows)
# venv\Scripts\activate
```

### 필수 패키지 설치

```bash
# requirements.txt 생성
cat > requirements.txt << 'EOF'
# PyTorch (CPU version for local dev)
torch>=2.0.0
torchaudio>=2.0.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
pretty_midi>=0.2.10
mido>=1.3.0

# Training
transformers>=4.30.0
accelerate>=0.20.0
datasets>=2.14.0

# Evaluation
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Deployment
gradio>=3.40.0
streamlit>=1.25.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.13.0

# Development
jupyter>=1.0.0
ipython>=8.14.0
pytest>=7.4.0
EOF

# 설치
pip install -r requirements.txt
```

**예상 시간**: 5-10분

### ✅ 확인

```python
# test_installation.py
import torch
import numpy as np
import pretty_midi
import transformers

print("✅ All packages installed successfully!")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Transformers: {transformers.__version__}")
```

```bash
python test_installation.py
```

---

## 5. Kaggle 첫 노트북 (30분)

### Kaggle에서 프로젝트 생성

```
1. Kaggle 로그인
2. Code → New Notebook 클릭
3. 제목: "Brad Mehldau AI - Phase 0 Test"
4. Accelerator: GPU T4 x2 선택
```

### 첫 테스트 코드

```python
# Cell 1: 환경 확인
import torch
import numpy as np
import sys

print("="*60)
print("Environment Check")
print("="*60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*60)
```

```python
# Cell 2: 간단한 transformer 테스트
from transformers import GPT2LMHeadModel, GPT2Config

# 작은 GPT-2 모델 생성
config = GPT2Config(
    vocab_size=256,
    n_positions=128,
    n_embd=256,
    n_layer=4,
    n_head=4,
)

model = GPT2LMHeadModel(config)
model = model.cuda()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 더미 입력으로 테스트
batch_size = 4
seq_len = 64
input_ids = torch.randint(0, 256, (batch_size, seq_len)).cuda()

with torch.no_grad():
    outputs = model(input_ids)

print(f"✅ Forward pass successful!")
print(f"Output shape: {outputs.logits.shape}")
```

```python
# Cell 3: 간단한 학습 루프 테스트
import torch.nn.functional as F

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("Testing training loop...")
for step in range(10):
    # 더미 데이터
    input_ids = torch.randint(0, 256, (batch_size, seq_len)).cuda()
    labels = torch.randint(0, 256, (batch_size, seq_len)).cuda()

    # Forward
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

print("✅ Training loop successful!")
```

### 노트북 저장

```
File → Save Version
→ "Phase 0: Environment Test"
→ Save & Run All
```

### ✅ 확인
- [ ] Kaggle 노트북 생성
- [ ] GPU 확인 성공
- [ ] 모델 forward pass 성공
- [ ] 학습 루프 성공

---

## 6. GitHub에 코드 업로드

### README.md 작성

```bash
cd ~/projects/Brad-Mehldau-AI
cat > README.md << 'EOF'
# 🎵 Brad Mehldau AI

AI that improvises jazz piano in Brad Mehldau's style using deep learning.

## 🚀 Status

- [x] Phase 0: Environment Setup
- [ ] Phase 1: Data Collection
- [ ] Phase 2: Quick Experiment
- [ ] Phase 3: Full Training
- [ ] Phase 4: Evaluation
- [ ] Phase 5: Deployment
- [ ] Phase 6: Portfolio

## 🎯 Goal

Create a deep learning model that can:
- Improvise jazz piano in Brad Mehldau's distinctive style
- Generate coherent chord progressions
- Maintain harmonic and rhythmic consistency
- Produce "listenable" jazz music

## 🛠️ Tech Stack

- PyTorch + Transformers
- GPT-2 architecture with LoRA fine-tuning
- Corruption-refinement training paradigm
- Rotary position encoding
- Multi-scale temporal attention

## 📊 Timeline

- **Week 1-2**: Data collection (MIDI files)
- **Week 3-4**: Quick experiment (first samples)
- **Week 5-8**: Full training (JazzFormer-CR)
- **Week 9-10**: Evaluation & sample generation
- **Week 11**: Deployment (Hugging Face)
- **Week 12**: Portfolio completion

## 🎧 Samples

(Coming soon after Phase 2)

## 📝 Blog

(Coming soon)

## 🤝 Contributing

This is a personal learning project, but feedback welcome!

## 📄 License

MIT

---

**Author**: [Your Name]
**Contact**: [Your Email]
**Started**: 2025-11-18
EOF
```

### 커밋 & 푸시

```bash
git add README.md requirements.txt test_installation.py
git commit -m "Phase 0 complete: Environment setup"
git push origin main
```

### ✅ 확인
- [ ] README.md 작성
- [ ] requirements.txt 추가
- [ ] GitHub에 푸시 완료

---

## 🎯 Phase 0 완료 체크리스트

### 계정
- [ ] Kaggle 계정 + 전화 인증
- [ ] Google Colab 계정
- [ ] GitHub 저장소 생성

### 환경
- [ ] 로컬 Python venv 생성
- [ ] 필수 패키지 설치
- [ ] 설치 테스트 성공

### GPU 테스트
- [ ] Colab GPU 테스트 성공
- [ ] Kaggle GPU 테스트 성공
- [ ] 간단한 모델 학습 성공

### GitHub
- [ ] 프로젝트 구조 생성
- [ ] README.md 작성
- [ ] 첫 커밋 완료

---

## 🐛 Troubleshooting

### Kaggle Phone Verification 안됨
```
→ Settings → Account → Phone Verification
→ 한국 번호: +82 10-XXXX-XXXX
→ SMS 안 오면: 이메일 support@kaggle.com에 문의
```

### Colab GPU 할당 안됨
```
→ Runtime → Change runtime type → GPU
→ "GPU 0/0" 메시지: 무료 할당 소진 (12시간 후 재시도)
→ 해결: Colab Pro 구독 ($10/월)
```

### PyTorch CUDA 에러
```bash
# CPU 버전 설치되었을 경우
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Git push 권한 에러
```bash
# Personal Access Token 필요
# GitHub → Settings → Developer settings → Personal access tokens
# → Generate new token (classic)
# → repo 권한 체크
# → 생성된 토큰 복사

# Push 시 비밀번호에 토큰 입력
git push origin main
Username: YOUR_USERNAME
Password: ghp_xxxxxxxxxxxxxxxxxxxx
```

---

## 📚 참고 자료

### Kaggle
- [Kaggle Docs](https://www.kaggle.com/docs)
- [GPU Quota](https://www.kaggle.com/docs/efficient-gpu-usage)

### Colab
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab Pro](https://colab.research.google.com/signup)

### PyTorch
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [CUDA Compatibility](https://pytorch.org/get-started/previous-versions/)

---

## ✅ Phase 0 완료!

축하합니다! 개발 환경 세팅이 완료되었습니다.

**다음 단계**: `../phase1_data/README.md`

데이터 수집을 시작하세요! 🚀
