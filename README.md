# Python Dev Tools Marketplace

Python/FastAPI 풀스택 개발을 위한 Claude Code 플러그인 마켓플레이스입니다.

## 필수 조건

- [Claude Code](https://claude.com/claude-code) 설치 및 인증 (v1.0.33 이상)
- 버전 확인: `claude --version`

## 설치 방법

### 1. 마켓플레이스 추가

Claude Code에서 다음 명령어를 실행합니다:

```shell
/plugin marketplace add ingpdw/pdw-python-dev-tools-marketplace
```

또는 Git URL로 추가:

```shell
/plugin marketplace add https://github.com/ingpdw/pdw-python-dev-tools-marketplace.git
```

### 2. 플러그인 설치

```shell
/plugin install pdw-python-dev-tools@pdw-python-dev-tools-marketplace
```

설치 범위를 지정할 수 있습니다:

```shell
# 사용자 범위 (기본값, 모든 프로젝트에서 사용)
/plugin install pdw-python-dev-tools@pdw-python-dev-tools-marketplace --scope user

# 프로젝트 범위 (팀과 공유, .claude/settings.json에 저장)
/plugin install pdw-python-dev-tools@pdw-python-dev-tools-marketplace --scope project

# 로컬 범위 (개인용, gitignored)
/plugin install pdw-python-dev-tools@pdw-python-dev-tools-marketplace --scope local
```

### 3. 설치 확인

Claude Code를 재시작한 후 `/help`를 실행하여 `pdw-python-dev-tools` 네임스페이스의 스킬이 표시되는지 확인합니다.

### 로컬 개발 테스트

소스에서 직접 플러그인을 로드하여 테스트할 수 있습니다:

```bash
git clone https://github.com/ingpdw/pdw-python-dev-tools-marketplace.git
claude --plugin-dir ./pdw-python-dev-tools-marketplace/plugins/pdw-python-dev-tools
```

### 팀 마켓플레이스 구성

프로젝트의 `.claude/settings.json`에 추가하면 팀원이 프로젝트를 신뢰할 때 자동으로 마켓플레이스 설치를 안내받습니다:

```json
{
  "extraKnownMarketplaces": {
    "pdw-python-dev-tools-marketplace": {
      "source": {
        "source": "github",
        "repo": "ingpdw/pdw-python-dev-tools-marketplace"
      }
    }
  },
  "enabledPlugins": {
    "pdw-python-dev-tools@pdw-python-dev-tools-marketplace": true
  }
}
```

## 포함된 플러그인

### pdw-python-dev-tools `v1.1.0`

10개의 스킬, 1개의 커맨드, 1개의 에이전트를 포함하는 Python/FastAPI 풀스택 개발 플러그인입니다.

### Skills (10개)

| Skill | Description | Assets / References |
|-------|-------------|---------------------|
| **package-managing** | Python 패키지/프로젝트 매니저 (uv 기반: init, add, sync, lock, run, 워크스페이스) | `pyproject-template.toml` |
| **asgi-server** | ASGI 서버 설정, 리소스 제한, 프로덕션 배포 (Uvicorn, Granian, Hypercorn, SSL/TLS, 프록시) | `deployment.md` |
| **app-scaffolding** | FastAPI 앱 스캐폴딩, 라우팅, 미들웨어, DI, 라이프스팬 관리 | `app-template/`, `routing-patterns.md`, `middleware.md`, `dependency-injection.md` |
| **async-patterns** | Python async/await 패턴, TaskGroup, 동시성 프리미티브, 에러 핸들링 | `concurrency.md` |
| **pydantic** | 데이터 검증, 직렬화, 모델 정의, BaseSettings, 커스텀 밸리데이터 | `validators.md` |
| **database** | SQLAlchemy async 엔진/세션, 모델 정의, CRUD 패턴, Alembic 마이그레이션, 커넥션 풀 | - |
| **auth-security** | OAuth2 + JWT 인증, 패스워드 해싱, RBAC, API Key 인증, 보안 베스트 프랙티스 | - |
| **agent-workflow** | LangChain/LangGraph 에이전트 워크플로, 도구 호출, 스트리밍, 체크포인팅 | `graph-template.py`, `langgraph-workflows.md`, `tools.md` |
| **docker-build** | Dockerfile 멀티스테이지 빌드, Docker Compose, 보안 강화, 멀티아키텍처, uv 통합 | `Dockerfile.fastapi`, `Dockerfile.dev`, `docker-compose.yml`, `.dockerignore` |
| **test-runner** | pytest 기반 테스트 실행, 커버리지, 비동기 테스트, fixture 스코프, 디버깅 | - |

### Command

- **`/pdw-python-dev-tools:scaffold-fastapi`** `<project-name> [--with-langchain] [--with-docker] [--with-db postgres|sqlite]`
  - FastAPI 프로젝트 전체 스캐폴딩 (9단계 워크플로)
  - uv 초기화 → 앱 구조 생성 → Pydantic 모델 → DB 레이어 → LangChain → Docker → 검증 → 요약

### Agent

- **fastapi-reviewer** - Python/FastAPI 코드 리뷰 전문 에이전트
  - async 정확성, FastAPI 패턴, Pydantic 모델, DI, 보안, 에러 핸들링, 테스트, Docker, 프로젝트 구조 검증

## 사용 예시

### 프로젝트 스캐폴딩

```shell
/pdw-python-dev-tools:scaffold-fastapi my-api --with-langchain --with-docker --with-db postgres
```

### 스킬 자동 호출

Claude에게 관련 작업을 요청하면 스킬이 자동으로 활용됩니다:

- "FastAPI 프로젝트를 만들어줘" → **app-scaffolding** 스킬 활성화
- "uv로 의존성 추가해줘" → **package-managing** 스킬 활성화
- "Pydantic 모델 만들어줘" → **pydantic** 스킬 활성화
- "데이터베이스 모델 만들어줘" → **database** 스킬 활성화
- "로그인 기능 추가해줘" → **auth-security** 스킬 활성화
- "Docker로 배포 설정해줘" → **docker-build** 스킬 활성화
- "테스트 작성해줘" → **test-runner** 스킬 활성화
- "LangGraph 에이전트 만들어줘" → **agent-workflow** 스킬 활성화

## 스킬 간 참조 관계

```
package-managing ──→ asgi-server ──→ docker-build
       │                  │                │
       ▼                  ▼                ▼
app-scaffolding ──→ async-patterns    test-runner
       │                                   │
       ▼                                   ▼
   pydantic ──→ agent-workflow         database
       │                                   │
       ▼                                   ▼
  auth-security ←─────────────────── database
```

- **app-scaffolding** → pydantic (요청/응답 모델), async-patterns (비동기 핸들러)
- **asgi-server** → docker-build (배포), package-managing (의존성 관리)
- **database** → pydantic (스키마), app-scaffolding (DI), async-patterns (비동기 세션)
- **auth-security** → database (유저 모델), pydantic (토큰 스키마), app-scaffolding (미들웨어)
- **agent-workflow** → pydantic (구조화된 출력), async-patterns (동시성)
- **docker-build** → asgi-server (서버 설정), package-managing (uv 통합)
- **test-runner** → async-patterns (비동기 테스트), database (DB 테스트 fixture)

## 프로젝트 구조

```
.
├── .claude-plugin/
│   └── marketplace.json                   # 마켓플레이스 메타데이터
├── plugins/
│   └── pdw-python-dev-tools/              # 메인 플러그인
│       ├── .claude-plugin/
│       │   └── plugin.json                # 플러그인 매니페스트
│       ├── commands/
│       │   └── scaffold-fastapi.md        # /scaffold-fastapi 커맨드
│       ├── agents/
│       │   └── fastapi-reviewer.md        # 코드 리뷰 에이전트
│       └── skills/
│           ├── package-managing/
│           │   ├── SKILL.md
│           │   └── assets/
│           │       └── pyproject-template.toml
│           ├── asgi-server/
│           │   ├── SKILL.md
│           │   └── references/
│           │       └── deployment.md
│           ├── app-scaffolding/
│           │   ├── SKILL.md
│           │   ├── assets/
│           │   │   └── app-template/
│           │   │       ├── main.py
│           │   │       ├── config.py
│           │   │       ├── dependencies.py
│           │   │       └── routers/
│           │   │           └── __init__.py
│           │   └── references/
│           │       ├── routing-patterns.md
│           │       ├── middleware.md
│           │       └── dependency-injection.md
│           ├── async-patterns/
│           │   ├── SKILL.md
│           │   └── references/
│           │       └── concurrency.md
│           ├── pydantic/
│           │   ├── SKILL.md
│           │   └── references/
│           │       └── validators.md
│           ├── agent-workflow/
│           │   ├── SKILL.md
│           │   ├── assets/
│           │   │   └── graph-template.py
│           │   └── references/
│           │       ├── langgraph-workflows.md
│           │       └── tools.md
│           ├── database/
│           │   └── SKILL.md
│           ├── auth-security/
│           │   └── SKILL.md
│           ├── docker-build/
│           │   ├── SKILL.md
│           │   └── assets/
│           │       ├── Dockerfile.fastapi
│           │       ├── Dockerfile.dev
│           │       ├── docker-compose.yml
│           │       └── .dockerignore
│           └── test-runner/
│               └── SKILL.md
└── README.md
```

## 플러그인 관리

### 업데이트

```shell
/plugin marketplace update
/plugin update pdw-python-dev-tools@pdw-python-dev-tools-marketplace
```

### 비활성화 / 활성화

```shell
/plugin disable pdw-python-dev-tools@pdw-python-dev-tools-marketplace
/plugin enable pdw-python-dev-tools@pdw-python-dev-tools-marketplace
```

### 제거

```shell
/plugin uninstall pdw-python-dev-tools@pdw-python-dev-tools-marketplace
```

## 검증

마켓플레이스 구성을 검증하려면:

```bash
claude plugin validate .
```

또는 Claude Code 내에서:

```shell
/plugin validate .
```

## 문제 해결

| 문제 | 해결 방법 |
|------|----------|
| 플러그인이 로드되지 않음 | `claude --debug`로 로딩 오류 확인 |
| 스킬이 나타나지 않음 | Claude Code 재시작 후 `/help` 확인 |
| 설치 실패 | `/plugin validate .`로 마켓플레이스 JSON 검증 |
| 인증 오류 (비공개 저장소) | `gh auth status`로 GitHub 인증 확인 |

## License

MIT
