# Claude Python Marketplace

Python/FastAPI 풀스택 개발을 위한 Claude Code 플러그인 마켓플레이스입니다.

## 포함된 플러그인

### pdw-python-dev-tools `v1.0.0`

8개의 스킬, 1개의 커맨드, 1개의 에이전트를 포함하는 Python/FastAPI 풀스택 개발 플러그인입니다.

#### Skills (8개)

| Skill | Description | Assets / References |
|-------|-------------|---------------------|
| **package-managing** | Python 패키지/프로젝트 매니저 (uv 기반: init, add, sync, lock, run, 워크스페이스) | `pyproject-template.toml` |
| **asgi-server** | ASGI 서버 설정 및 프로덕션 배포 (Uvicorn, Granian, Hypercorn, SSL/TLS, 프록시) | `deployment.md` |
| **app-scaffolding** | FastAPI 앱 스캐폴딩, 라우팅, 미들웨어, DI, 라이프스팬 관리 | `app-template/`, `routing-patterns.md`, `middleware.md`, `dependency-injection.md` |
| **async-patterns** | Python async/await 패턴, TaskGroup, 동시성 프리미티브, 에러 핸들링 | `concurrency.md` |
| **pydantic** | 데이터 검증, 직렬화, 모델 정의, BaseSettings, 커스텀 밸리데이터 | `validators.md` |
| **agent-workflow** | LangChain/LangGraph 에이전트 워크플로, 도구 호출, 스트리밍, 체크포인팅 | `graph-template.py`, `langgraph-workflows.md`, `tools.md` |
| **docker-build** | Dockerfile 멀티스테이지 빌드, Docker Compose, 보안 강화, uv 통합 | `Dockerfile.fastapi`, `Dockerfile.dev`, `docker-compose.yml`, `.dockerignore` |
| **test-runner** | pytest 기반 테스트 실행, 커버리지, 비동기 테스트, 파라미터화, 마커 | - |

#### Command

- **`/scaffold-fastapi`** `<project-name> [--with-langchain] [--with-docker] [--with-db postgres|sqlite]`
  - FastAPI 프로젝트 전체 스캐폴딩 (9단계 워크플로)
  - uv 초기화 → 앱 구조 생성 → Pydantic 모델 → DB 레이어 → LangChain → Docker → 검증 → 요약

#### Agent

- **fastapi-reviewer** - Python/FastAPI 코드 리뷰 전문 에이전트
  - async 정확성, FastAPI 패턴, Pydantic 모델, DI, 보안, 에러 핸들링, 테스트, Docker, 프로젝트 구조 검증

## 설치 방법

### 마켓플레이스 등록

`~/.claude/plugins/known_marketplaces.json`에 다음을 추가:

```json
{
  "pdw-python-dev-toolss-marketplace": {
    "source": {
      "source": "github",
      "repo": "ingpdw/pdw-python-dev-toolss-marketplace"
    }
  }
}
```

### 플러그인 설치

```
/plugin install pdw-python-dev-tools@pdw-python-dev-toolss-marketplace
```

### 로컬 테스트

```bash
git clone https://github.com/ingpdw/pdw-python-dev-toolss-marketplace.git
claude --plugin-dir ./pdw-python-dev-toolss-marketplace/plugins/pdw-python-dev-tools
```

## 프로젝트 구조

```
.
├── .claude-plugin/
│   └── marketplace.json                    # 마켓플레이스 메타데이터
├── plugins/
│   └── pdw-python-dev-tools/                # 메인 플러그인
│       ├── .claude-plugin/
│       │   └── plugin.json                 # 플러그인 설정 (v1.0.0)
│       ├── commands/
│       │   └── scaffold-fastapi.md         # /scaffold-fastapi 커맨드
│       ├── agents/
│       │   └── fastapi-reviewer.md         # 코드 리뷰 에이전트
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

## 스킬 간 참조 관계

```
package-managing ──→ asgi-server ──→ docker-build
       │                  │                │
       ▼                  ▼                ▼
app-scaffolding ──→ async-patterns    test-runner
       │
       ▼
   pydantic ──→ agent-workflow
```

- **app-scaffolding** → pydantic (요청/응답 모델), async-patterns (비동기 핸들러)
- **asgi-server** → docker-build (배포), package-managing (의존성 관리)
- **agent-workflow** → pydantic (구조화된 출력), async-patterns (동시성)
- **docker-build** → asgi-server (서버 설정), package-managing (uv 통합)
- **test-runner** → async-patterns (비동기 테스트), app-scaffolding (FastAPI 테스트)

## License

MIT
