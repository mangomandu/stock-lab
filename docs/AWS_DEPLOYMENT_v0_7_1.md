# AWS 배포 가이드 — v0.7.1 Static HP Full Grid (7,500 cells)

## TL;DR

```
Goal:       v0.7.1 풀그리드 7,500 cells WSL2 → AWS 이전
Instance:   c6i.32xlarge spot (128 vCPU, 256 GB RAM)
시간:        ~3.3시간
비용:        ~$5 (spot $1.5/h × 3.3h)
Setup:      처음 1.5-3h (계정 처음이면 quota 승인 대기 가능)
```

---

## Step 1: AWS 계정 + IAM (15-30 min)

### 1.1 계정 로그인
이미 로그인 완료. AWS Management Console (https://console.aws.amazon.com) 접속.

### 1.2 Region 선택
**us-east-1 (Virginia)** 추천 — Spot 가격 가장 cheap. 우측 상단에서 변경.

### 1.3 IAM User 생성 (선택, 이미 있으면 skip)
- IAM → Users → Create user
- Name: `claude-lab` (또는 원하는 이름)
- Permissions: `AmazonEC2FullAccess`, `IAMFullAccess` (편의 — production이면 더 narrow)
- Access keys 발급 → CSV 다운로드 (분실 시 재발급 필요)

### 1.4 SSH key pair 생성
- EC2 → Key Pairs → Create key pair
- Name: `claude-lab-key`
- Type: ED25519, Format: .pem
- 다운로드된 .pem 파일 안전한 곳에 저장:
  ```bash
  mv ~/Downloads/claude-lab-key.pem ~/.ssh/
  chmod 400 ~/.ssh/claude-lab-key.pem
  ```

---

## Step 2: Spot Quota 확인 (5 min ~ 수일)

c6i.32xlarge spot 인스턴스는 128 vCPU 사용. 신규 계정은 "Spot Instance Requests for All Standard Instances" quota가 0일 수 있음.

### 2.1 Quota 확인
- Service Quotas → AWS services → Amazon EC2
- 검색: "Running On-Demand Standard"
- "All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests"
- 현재 quota 확인. **128 미만이면 Request increase**.

### 2.2 Quota Request (필요 시)
- Request quota increase → 128 (또는 256으로 여유)
- 사유: "Quantitative finance backtesting research, 1-time grid search"
- **승인 시간**: 즉시 ~ 1-2일 (보통 자동 승인)

### 2.3 대안 (quota 안 나오면)
- **c6i.16xlarge** (64 vCPU): 시간 2배 (~6.5h), 비용 동일 ($5)
- **c6i.8xlarge** (32 vCPU): 시간 4배 (~13h), 비용 동일

---

## Step 3: Spot 인스턴스 launch (10 min)

### 3.1 Launch Template 만들거나 직접 launch:

EC2 → Instances → Launch Instance:
- **Name**: `stock-lab-v071`
- **AMI**: Amazon Linux 2023 (free tier eligible)
- **Instance type**: `c6i.32xlarge` (128 vCPU, 256GB RAM)
- **Key pair**: claude-lab-key
- **Network**: 기본 VPC, public IP
- **Storage**: 50 GB gp3 (default 8GB는 부족)
- **Advanced** → **Spot instances**: ✓ Request spot
  - Maximum price: $1.50/h (현재 spot ~$1.0-1.5)
  - Interruption behavior: Stop (재시작 가능)

→ Launch

### 3.2 인스턴스 확인
- 1-2분 후 "Running" 상태
- Public IP 주소 메모

### 3.3 SSH 접속
```bash
ssh -i ~/.ssh/claude-lab-key.pem ec2-user@<PUBLIC_IP>
```

---

## Step 4: 환경 setup (15 min)

### 4.1 Python + 필수 패키지
```bash
# Python 3.11 설치
sudo dnf install -y python3.11 python3.11-pip git

# 또는 system Python 3.9 사용 가능
python3 --version

# Pip 업그레이드
python3 -m pip install --upgrade pip

# Required packages
pip3 install --user pandas numpy scikit-learn scipy
```

### 4.2 Code 업로드 (rsync)

**로컬 (WSL2)에서**:
```bash
# 코드 업로드 (data 제외 first, 그 다음 별도)
rsync -avz --progress \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='data/master_sp500/' \
  --exclude='results/' \
  -e "ssh -i ~/.ssh/claude-lab-key.pem" \
  /home/dlfnek/stock_lab/ \
  ec2-user@<PUBLIC_IP>:~/stock_lab/

# Master_sp500 데이터 별도 업로드 (~수백 MB)
rsync -avz --progress \
  -e "ssh -i ~/.ssh/claude-lab-key.pem" \
  /home/dlfnek/stock_lab/data/master_sp500/ \
  ec2-user@<PUBLIC_IP>:~/stock_lab/data/master_sp500/
```

업로드 시간: ~5-10min (한국 → us-east-1, ~수백 MB)

---

## Step 5: v0.7.1 worker 작성 + 실행

### 5.1 Worker 작성
v0.7.1 worker는 v0.7.0 worker와 거의 동일하지만:
- 22 model cells → **7,500 cells (6 axes joint)**
- Multi-process design은 동일 (Pool initializer)
- 128 vCPU 활용 → `--workers 128`
- v0.7.0 best model을 lock하고 6 axes만 vary

**파일명**: `tests/test_v0_7_1_full_grid.py` (구체 구현은 v0.7.0 결과 받은 후 작성 권장)

### 5.2 실행 (tmux 권장)
```bash
# tmux로 launch — SSH 끊겨도 계속 실행
tmux new -s bakeoff
cd ~/stock_lab

# 실행 (3.3시간 예상)
python3 tests/test_v0_7_1_full_grid.py --workers 128 2>&1 | tee /tmp/v0_7_1.log

# tmux detach: Ctrl+B then D
# tmux 재접속: tmux attach -t bakeoff
```

### 5.3 진행 모니터링
```bash
# 다른 SSH 세션에서:
tail -f /tmp/v0_7_1.log

# 또는:
sqlite3 ~/stock_lab/results/v0_7_1_checkpoint.db \
  "SELECT COUNT(*) FROM results;"
```

---

## Step 6: 결과 download (5 min)

### 6.1 결과 파일 download
**로컬 (WSL2)에서**:
```bash
rsync -avz --progress \
  -e "ssh -i ~/.ssh/claude-lab-key.pem" \
  ec2-user@<PUBLIC_IP>:~/stock_lab/results/v0_7_1_*.json \
  /home/dlfnek/stock_lab/results/

rsync -avz --progress \
  -e "ssh -i ~/.ssh/claude-lab-key.pem" \
  ec2-user@<PUBLIC_IP>:~/stock_lab/results/v0_7_1_*.db \
  /home/dlfnek/stock_lab/results/
```

### 6.2 인스턴스 termination (중요!)
**비용 발생 방지 위해 사용 후 즉시 terminate**:
```
EC2 → Instances → Select stock-lab-v071 → Instance state → Terminate instance
```

---

## Spot Interruption 대응

Spot 인스턴스는 언제든 회수 가능. 우리 worker는 SQLite checkpoint로 자동 재개:

1. Spot 끊김 알림 받음 (2분 사전 통지)
2. 인스턴스 stop됨 (terminate가 아니라 stop이면 재시작 가능)
3. Console에서 다시 start
4. SSH 접속 → tmux 재실행 → 자동 resume (checkpoint에서)

---

## 비용 모니터링

- AWS Billing Dashboard에서 daily 확인
- Spot 가격 변동: us-east-1 c6i.32xlarge 보통 $1.0-1.5/h
- 3.3시간 + setup 1시간 = ~4.3 vCPU-hours = **$5-7 예상**

---

## Troubleshooting

### "Insufficient capacity" 에러
- Spot pool 일시 부족. region 변경 (us-east-2, us-west-2) 또는 인스턴스 타입 변경 (c5/c6a/c7i)

### "EBS volume too small"
- Storage 50GB로 늘렸는지 확인. data + checkpoint 합쳐 ~10GB 사용

### 워커가 멈춤 / 메모리 부족
- 128 process 동시 panel load = ~7.7GB 사용. 256GB RAM 충분
- 메모리 부족 시 `--workers 64`로 축소

### SSH 접속 안 됨
- Security group inbound rule: 22 port 본인 IP만 허용
- Key file permissions: `chmod 400 ~/.ssh/claude-lab-key.pem`

---

## 체크리스트

```
[ ] AWS 계정 로그인 완료
[ ] us-east-1 region 선택
[ ] IAM user + access keys 발급 (또는 root 사용)
[ ] SSH key pair 생성 + 다운로드 + chmod 400
[ ] Spot quota 128 vCPU 이상 확보 (또는 c6i.16xlarge 사용)
[ ] c6i.32xlarge spot launch
[ ] Public IP 메모, SSH 접속 확인
[ ] Python + pandas/numpy/sklearn 설치
[ ] Code rsync (제외: data, results, __pycache__)
[ ] Master_sp500 데이터 rsync
[ ] tests/test_v0_7_1_full_grid.py 작성 (v0.7.0 결과 받은 후)
[ ] tmux launch + python3 실행
[ ] 진행 모니터링
[ ] 완료 후 결과 download
[ ] 인스턴스 terminate (중요!)
[ ] 비용 청구 확인
```

---

## 다음 단계 (v0.7.1 끝나면)

- v0.7.1 결과 분석 (analyze_v0_7_1.py)
- v0.7.0 best model + v0.7.1 best HP 통합 → final config
- v0.7.2 (Hysteresis) 코딩 — 운용 시 WSL2 병렬 가능
- v0.7.3 (Refresh) — 또 AWS 권장 (1.5-2h, $3-4)
