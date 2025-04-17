# COSE304_TP_G7
repository for group project of 2025-spring-COSE304

# CycleGAN 프로젝트 Docker 사용 가이드

이 저장소는 CycleGAN 구현을 위한 도커 환경을 제공합니다. 팀원 모두가 동일한 개발 환경에서 작업할 수 있도록 도와줍니다.

## 목차
- [시작하기](#시작하기)
- [로컬에서 Docker 사용하기](#로컬에서-docker-사용하기)
- [사전 빌드된 Docker 이미지 사용하기](#사전-빌드된-docker-이미지-사용하기)
- [개발 시 유용한 Docker 명령어](#개발-시-유용한-docker-명령어)
- [프로젝트 구조](#프로젝트-구조)

## 시작하기

### 사전 요구사항
- [Docker 설치](https://www.docker.com/get-started)
- Git 설치

### 저장소 클론하기
```bash
git clone <repository-url>
cd cyclegan-project
```

## 로컬에서 Docker 사용하기

### Docker 이미지 빌드하기
```bash
docker build -t cyclegan-project .
```

### Docker 컨테이너 실행하기
기본 실행:
```bash
docker run -it --name cyclegan-container cyclegan-project
```

로컬 디렉토리를 마운트하여 실행 (코드 변경사항이 실시간으로 반영됨):
```bash
docker run -it -v $(pwd):/app --name cyclegan-container cyclegan-project
```

GPU를 사용하여 실행 (NVIDIA Docker가 설치되어 있어야 함):
```bash
docker run -it --gpus all -v $(pwd):/app --name cyclegan-container cyclegan-project
```

### Docker Compose 사용하기
```bash
docker-compose up
```

## 사전 빌드된 Docker 이미지 사용하기

팀원들은 직접 이미지를 빌드하지 않고 미리 빌드된 이미지를 다운로드하여 사용할 수 있습니다.

### Docker Hub에서 이미지 다운로드
```bash
docker pull yourusername/cyclegan-project:latest
docker run -it -v $(pwd):/app yourusername/cyclegan-project
```

### GitHub Packages에서 이미지 다운로드 (선택 사항)
```bash
docker pull ghcr.io/username/cyclegan-project:latest
docker run -it -v $(pwd):/app ghcr.io/username/cyclegan-project
```

### GitLab Container Registry에서 이미지 다운로드 (선택 사항)
```bash
docker pull registry.gitlab.com/username/project-name:latest
docker run -it -v $(pwd):/app registry.gitlab.com/username/project-name:latest
```

## 개발 시 유용한 Docker 명령어

### 실행 중인 컨테이너 목록 보기
```bash
docker ps
```

### 모든 컨테이너 목록 보기 (중지된 컨테이너 포함)
```bash
docker ps -a
```

### 컨테이너 중지하기
```bash
docker stop cyclegan-container
```

### 컨테이너 다시 시작하기
```bash
docker start -i cyclegan-container
```

### 컨테이너 삭제하기
```bash
docker rm cyclegan-container
```

### 이미지 삭제하기
```bash
docker rmi cyclegan-project
```

### 컨테이너 내부에서 bash 실행하기
```bash
docker exec -it cyclegan-container bash
```

## 프로젝트 구조
```
cyclegan-project/
├── Dockerfile          # Docker 환경 설정
├── docker-compose.yml  # Docker Compose 설정
├── requirements.txt    # Python 패키지 의존성
├── main.py             # 메인 실행 파일
├── models/             # CycleGAN 모델 구현
│   ├── cycle_gan.py
│   ├── discriminator.py
│   └── generator.py
├── datasets/           # 데이터셋 로딩 및 전처리
│   └── dataset.py
├── utils/              # 유틸리티 함수
│   ├── image_pool.py
│   └── visualizer.py
└── README.md           # 현재 파일
```

## 사용 예시

CycleGAN 모델 학습하기:
```bash
docker run -it -v $(pwd):/app cyclegan-project python main.py --mode train --dataroot ./datasets/horse2zebra
```

학습된 모델로 이미지 변환하기:
```bash
docker run -it -v $(pwd):/app cyclegan-project python main.py --mode test --dataroot ./test_images
```

## 문제 해결

### Docker 이미지 빌드 오류
- 네트워크 연결을 확인하세요
- Docker 데몬이 실행 중인지 확인하세요
- Dockerfile에 문법 오류가 없는지 확인하세요

### GPU 사용 오류
- NVIDIA Docker가 올바르게 설치되었는지 확인하세요
- CUDA 버전 호환성을 확인하세요

문제가 지속되면 이슈를 생성해주세요.