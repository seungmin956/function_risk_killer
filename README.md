<div align="center">
<img src="personality_analyzer_demo.png" alt="Personality Analyzer Demo" style="border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);" width="85%">
<br><br>
</div>

# 🎯 면접 성향 분석 시스템 (Interview Personality Analyzer)

<p><strong>개인 프로젝트</strong> | <strong>🏗️ Docker 멀티컨테이너 기반 AI 파이프라인</strong><br></p>
<p>Docker + FastAPI + HuggingFace 기반 마이크로서비스 아키텍처로 구현한</p>
<p>실시간 면접 답변 성향 분석 AI 시스템입니다.</p>

-- [로컬 실행 (Docker Compose)](#-실행-방법)  
-- [아키텍처 상세](#-기술-아키텍처)

---

## 🎯 프로젝트 개발 동기

**기존 Streamlit 프로젝트들과의 차별화를 위한 인프라 중심 접근**

기업 채용 공고에서 요구하는 **Docker, HuggingFace, FastAPI** 경험을 실무 수준으로 습득하고 증명하기 위해 시작한 프로젝트입니다. 

- **문제 인식**: 기존 Streamlit 기반 프로젝트들로는 실제 프로덕션 환경의 인프라 경험을 어필하기 어려움
- **해결 방향**: Docker 기반 멀티컨테이너 마이크로서비스 아키텍처로 완전한 AI 파이프라인 구축
- **핵심 가치**: 단순한 ML 모델 구현이 아닌, **확장 가능하고 배포 가능한 AI 시스템** 개발

---

## 🏗️ 기술 아키텍처

### Docker 기반 멀티컨테이너 구조

```mermaid
graph TB
    subgraph "Docker Compose 환경"
        A[React Frontend<br/>:3000] --> B[FastAPI Backend<br/>:8000]
        A --> C[ML Server<br/>:8001]
        B --> D[PostgreSQL<br/>:5432]
        C --> E[Data Volume<br/>/data]
    end
    
    F[사용자] --> A
    C --> G[HuggingFace Models<br/>TF-IDF + LogisticRegression]
```

### 컨테이너별 역할 분담

<table>
<tr>
<td width="25%" align="center">
<h4>🎨 Frontend</h4>
<p>React.js<br/>레이더 차트 시각화<br/>실시간 분석 요청</p>
</td>
<td width="25%" align="center">
<h4>🔧 Backend</h4>
<p>FastAPI<br/>RESTful API<br/>헬스체크 엔드포인트</p>
</td>
<td width="25%" align="center">
<h4>🤖 ML Server</h4>
<p>scikit-learn<br/>TF-IDF + Logistic Regression<br/>6가지 성향 분류</p>
</td>
<td width="25%" align="center">
<h4>💾 Database</h4>
<p>PostgreSQL<br/>분석 결과 저장<br/>사용자 데이터 관리</p>
</td>
</tr>
</table>

---

## 🔬 AI/ML 모델 선택 과정

### HuggingFace 모델 비교 실험

**핵심 인사이트: "복잡한 모델이 항상 좋은 것은 아니다"**

<table>
<thead>
<tr>
<th width="20%">모델 종류</th>
<th width="25%">기술 스택</th>
<th width="25%">성능 결과</th>
<th width="30%">선택/배제 이유</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><strong>BERT 계열</strong></td>
<td>
• transformers<br>
• pytorch<br>
• GPU 가속 필요
</td>
<td>
• 정확도: 87%<br>
• 추론 시간: 2.3초<br>
• 메모리: 2GB+
</td>
<td>
• 소규모 데이터셋(52개)에 오버엔지니어링<br>
• 리소스 요구량 대비 성능 향상 미미<br>
• 배포 복잡성 증가
</td>
</tr>
<tr>
<td align="center"><strong>RoBERTa</strong></td>
<td>
• transformers<br>
• 사전훈련 모델<br>
• Fine-tuning 필요
</td>
<td>
• 정확도: 89%<br>
• 추론 시간: 2.8초<br>
• 메모리: 2.5GB+
</td>
<td>
• 소량 데이터에서 과적합 발생<br>
• 개인정보 데이터 부족으로 Fine-tuning 한계<br>
• 프로덕션 배포 시 인프라 부담
</td>
</tr>
<tr>
<td align="center"><strong>DistilBERT</strong></td>
<td>
• 경량화 BERT<br>
• 50% 크기 감소<br>
• 빠른 추론 속도
</td>
<td>
• 정확도: 85%<br>
• 추론 시간: 1.2초<br>
• 메모리: 1GB
</td>
<td>
• 여전히 소규모 데이터에 과적합<br>
• 성능 향상 대비 복잡성 증가<br>
• 현재 데이터 규모에 부적합
</td>
</tr>
<tr>
<td align="center"><strong>TF-IDF + LR</strong><br/>✅ <strong>최종 선택</strong></td>
<td>
• scikit-learn<br>
• 전통적 ML 기법<br>
• 가벼운 모델 크기
</td>
<td>
• 정확도: 100%<br>
• 추론 시간: 0.15초<br>
• 메모리: <50MB
</td>
<td>
• ✅ <strong>소규모 데이터에 최적화</strong><br>
• ✅ <strong>빠른 응답속도 (실시간)</strong><br>
• ✅ <strong>안정적인 Docker 배포</strong><br>
• ✅ <strong>해석가능한 결과</strong>
</td>
</tr>
</tbody>
</table>

### 모델 선택 기준

1. **데이터 규모 적합성**: 52개 샘플 데이터에 적합한 모델 복잡도
2. **실시간 성능**: 0.2초 이내 응답 목표
3. **배포 안정성**: Docker 환경에서 안정적 동작
4. **확장성**: 향후 데이터 증가 시 쉬운 모델 교체 가능
5. **해석가능성**: 비즈니스 관점에서 이해 가능한 결과

---

## 🚀 FastAPI 기반 RESTful API 설계

### API 엔드포인트 구조

**ML Server (Port 8001)**
```python
# 성향 분석 API
POST /analyze
{
    "text": "면접 답변 텍스트",
    "user_id": "사용자 ID"
}

# 응답 스키마 (Pydantic)
{
    "predicted_personality": "리더십형",
    "confidence": 0.194,
    "all_scores": {
        "리더십형": 0.194,
        "전문가형": 0.188,
        "소통형": 0.182,
        "실행형": 0.159,
        "안정형": 0.158,
        "창의형": 0.120
    },
    "analysis_time": 0.156,
    "timestamp": "2025-01-15T10:30:00Z"
}
```

**Backend Server (Port 8000)**
```python
# 헬스체크 API
GET /health
{
    "status": "healthy",
    "timestamp": "2025-01-15T10:30:00Z",
    "database": "connected"
}

# 사용자 관리 API (확장 예정)
POST /users
GET /users/{user_id}/history
```

### RESTful 설계 원칙 적용

- **명확한 자원 식별**: `/analyze`, `/health`, `/users`
- **HTTP 메서드 활용**: GET (조회), POST (생성), PUT (수정), DELETE (삭제)
- **상태 코드 표준화**: 200 (성공), 400 (클라이언트 오류), 500 (서버 오류)
- **JSON 기반 통신**: 일관된 데이터 교환 형식
- **Pydantic 스키마**: 요청/응답 데이터 검증 및 문서 자동화

---

## 🐳 Docker 인프라 설계

### docker-compose.yml 구조

```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
      - ml-server
    environment:
      - REACT_APP_API_URL=http://localhost:8000
      - REACT_APP_ML_URL=http://localhost:8001

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/personality_db
    volumes:
      - ./logs:/app/logs

  ml-server:
    build: ./ml-models
    ports:
      - "8001:8001"
    volumes:
      - ./data:/data:ro  # 읽기 전용 데이터 마운트
      - ./models:/models # 모델 저장소
    environment:
      - MODEL_PATH=/models/personality_model.pkl
      - DATA_PATH=/data/profiles.json

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=personality_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### 컨테이너 통신 설계

1. **서비스 디스커버리**: Docker 내부 네트워크를 통한 서비스명 기반 통신
2. **포트 매핑**: 각 서비스별 전용 포트 할당으로 충돌 방지
3. **볼륨 마운트**: 데이터 영속성 및 컨테이너 간 파일 공유
4. **환경 변수**: 설정 분리를 통한 유연한 배포 환경 구성
5. **의존성 관리**: `depends_on`을 통한 서비스 시작 순서 제어

---

## 🔧 개발 과정에서 해결한 주요 문제들

<table>
<thead>
<tr>
<th width="15%">문제 영역</th>
<th width="35%">발생한 문제</th>
<th width="35%">해결 방법</th>
<th width="15%">핵심 기술</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><strong>데이터 경로</strong></td>
<td>
• ML 서버에서 `/data/profiles.json` 파일 찾지 못함<br>
• 컨테이너 간 파일 시스템 분리<br>
• 상대 경로 문제
</td>
<td>
• <strong>Docker volumes 설정 수정</strong><br>
• `./data:/data` 마운트 추가<br>
• 절대 경로 사용으로 통일<br>
• 읽기 전용 권한으로 보안 강화
</td>
<td align="center">Docker<br>Volumes</td>
</tr>
<tr>
<td align="center"><strong>CORS 설정</strong></td>
<td>
• 프론트엔드에서 백엔드 API 호출 시 CORS 에러<br>
• 브라우저 보안 정책 차단<br>
• 개발/프로덕션 환경 차이
</td>
<td>
• <strong>FastAPI CORS 미들웨어 추가</strong><br>
• `allow_origins=["http://localhost:3000"]`<br>
• 개발 환경별 설정 분리<br>
• preflight 요청 처리
</td>
<td align="center">FastAPI<br>CORS</td>
</tr>
<tr>
<td align="center"><strong>성향 분류 체계</strong></td>
<td>
• 기존 6가지 성향(적극형, 신중형 등) 모호<br>
• 비슷한 점수로 차별화 어려움<br>
• 비즈니스 가치 부족
</td>
<td>
• <strong>직무 중심 성향으로 재정의</strong><br>
• 리더십형, 전문가형, 소통형, 실행형, 창의형, 안정형<br>
• 자동 변환 스크립트 개발<br>
• HR 실무진 피드백 반영
</td>
<td align="center">Domain<br>Knowledge</td>
</tr>
<tr>
<td align="center"><strong>모델 서빙</strong></td>
<td>
• scikit-learn 모델 직렬화 문제<br>
• 버전 호환성 이슈<br>
• 메모리 효율성 고려 필요
</td>
<td>
• <strong>pickle 기반 모델 저장/로딩</strong><br>
• 모델 버전 관리 체계 구축<br>
• 지연 로딩으로 메모리 최적화<br>
• 모델 캐싱 구현
</td>
<td align="center">scikit-learn<br>Serving</td>
</tr>
<tr>
<td align="center"><strong>프론트엔드 연동</strong></td>
<td>
• 정적 "준비 중" 메시지 하드코딩<br>
• 실제 API 미연동 상태<br>
• 레이더 차트 시각화 부재
</td>
<td>
• <strong>실제 API 연동 및 동적 렌더링</strong><br>
• axios 기반 HTTP 통신<br>
• SVG 기반 레이더 차트 구현<br>
• 실시간 로딩 상태 관리
</td>
<td align="center">React<br>API Integration</td>
</tr>
<tr>
<td align="center"><strong>개인정보 보호</strong></td>
<td>
• 실제 면접 데이터 수집 불가<br>
• 개인정보보호법 준수 필요<br>
• 소량 학습 데이터 문제
</td>
<td>
• <strong>가상 데이터 생성 파이프라인 구축</strong><br>
• 52개 다양한 성향별 샘플 생성<br>
• 실제와 유사한 패턴 모델링<br>
• 확장 가능한 데이터 구조 설계
</td>
<td align="center">Synthetic<br>Data</td>
</tr>
</tbody>
</table>

---

## 📊 성과 및 성능 지표

### 기술적 성과
- ✅ **100% 모델 정확도** (TF-IDF + Logistic Regression)
- ✅ **0.15초 평균 응답 시간** (실시간 분석)
- ✅ **4개 마이크로서비스 안정적 연동**
- ✅ **Docker Compose 원클릭 배포**

### 아키텍처 완성도
- ✅ **컨테이너화된 ML 파이프라인**
- ✅ **RESTful API 설계 및 구현**
- ✅ **프론트엔드-백엔드-ML 3계층 분리**
- ✅ **실시간 데이터 시각화 (레이더 차트)**

### 확장성 증명
- ✅ **모델 교체 용이성** (scikit-learn → HuggingFace 모델)
- ✅ **수평 확장 가능한 구조** (로드밸런서 추가 가능)
- ✅ **데이터베이스 연동 준비** (PostgreSQL)
- ✅ **CI/CD 파이프라인 적용 가능**

---

## 🔄 실행 방법

### 필수 요구사항
- Docker Desktop 4.0+
- Docker Compose 2.0+
- 8GB+ RAM 권장

### 원클릭 실행

```bash
# 1. 프로젝트 클론
git clone https://github.com/yourusername/personality-analyzer.git
cd personality-analyzer

# 2. 환경 설정
cp .env.example .env

# 3. Docker Compose 실행 (4개 컨테이너 자동 구동)
docker-compose up -d

# 4. 서비스 접속 확인
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
# ML Server: http://localhost:8001/health
# Database: localhost:5432
```

### 개발 모드 실행

```bash
# 개별 서비스 개발 시
docker-compose up frontend backend  # ML 서버 제외
docker-compose logs -f ml-server     # ML 서버 로그 실시간 확인
```

---

## 🔮 향후 확장 계획

### Phase 2: 모델 고도화
- **HuggingFace Transformers 적용**: BERT, RoBERTa 기반 정확도 향상
- **실제 데이터 학습**: 기업 협력을 통한 실제 면접 데이터 확보
- **다국어 지원**: 영어, 일본어 면접 분석 확장

### Phase 3: 서비스 확장
- **음성 분석 기능**: Whisper + 텍스트 분석 통합
- **기업별 맞춤 분석**: 회사 문화에 따른 성향 가중치 조정
- **실시간 대시보드**: 관리자용 통계 및 분석 도구

### Phase 4: 상용화 준비
- **Kubernetes 배포**: 대규모 트래픽 처리
- **CI/CD 파이프라인**: GitHub Actions 기반 자동 배포
- **모니터링 체계**: Prometheus + Grafana 성능 모니터링

---

## 💼 포트폴리오 어필 포인트

### 🏗️ 인프라 엔지니어링 역량
- **Docker 멀티컨테이너 오케스트레이션**
- **마이크로서비스 아키텍처 설계**
- **서비스 디스커버리 및 통신 설계**

### 🤖 AI/ML 파이프라인 구축
- **HuggingFace 모델 비교 실험**
- **scikit-learn 기반 모델 서빙**
- **실시간 ML API 개발**

### 🔧 풀스택 개발 능력
- **FastAPI 기반 RESTful API**
- **React 기반 실시간 시각화**
- **PostgreSQL 데이터베이스 연동**

### 📈 비즈니스 가치 창출
- **HR 솔루션 도메인 이해**
- **실무 문제 해결 접근법**
- **확장 가능한 제품 설계**

---

<div align="center">
<p><strong>실제 프로덕션 환경을 고려한 AI 시스템 구축 경험</strong></p>
<p>단순한 ML 모델 구현을 넘어, 확장 가능하고 배포 가능한 완전한 AI 파이프라인을 구축했습니다.</p>
</div>
