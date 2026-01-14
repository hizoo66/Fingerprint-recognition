# Fingerprint Recognition Model

본 프로젝트는 Siamese 기반 딥러닝 모델을 이용하여  
두 개의 지문 이미지가 동일한 사람의 지문인지 여부를 판별하는  
지문 인증(Fingerprint Verification) 시스템을 구현한 연구용 레포지토리입니다.

---

## 📌 Project Overview

일반적인 분류 방식(사람 ID 분류)이 아닌,  
**두 지문 이미지 간 유사도를 학습하여 동일/비동일을 판별하는 구조**를 사용하였습니다.

| 입력            | 출력                       |
| --------------- | -------------------------- |
| 지문 이미지 2장 | 두 지문이 동일 인물일 확률 |

---

## 📂 Directory Structure

models/ 학습 완료된 모델 (.h5, .keras)
history/ 학습 과정 로그 및 결과
notebooks/ 모델 학습 및 실험용 Jupyter Notebook
dataset/ 지문 데이터셋
train.py 모델 학습 스크립트
eval.py FAR / FRR / Threshold 평가 코드

---

## 🧠 Model Architecture

- Backbone: CNN Feature Extractor
- Metric Learning: Siamese Network
- Loss Function: Binary Crossentropy
- Output: 0 ~ 1 사이 similarity score

---

## ⚙️ Installation

```bash
git clone https://github.com/hizoo66/Fingerprint-Recognition.git
cd Fingerprint-Recognition
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python train.py
```

---

📊 Evaluation
FAR / FRR 기반 threshold 최적화

```bash
python eval.py
```

출력예시

```bash
Best Threshold: 0.63
FAR: 0.021
FRR: 0.034
```

---

## 🔍 Inference Logic

두 지문 이미지 A, B를 입력하여 similarity score를 출력하며
threshold 이상이면 동일 지문으로 판단합니다.

---

## 📈 Result

| Metric | Value |
| ------ | ----- |
| FAR    | 2.1%  |
| FRR    | 3.4%  |

---

## 📎 Notes

- 본 레포는 모델 개발 및 연구 목적으로만 사용됩니다.
- 실제 서비스 코드는 별도의 서비스 전용 레포에서 관리합니다.

--
