# Claude Python FastAPI Marketplace

Python/FastAPI 풀스택 개발을 위한 Claude Code 플러그인 마켓플레이스입니다.

## 포함된 플러그인

### python-fastapi-stack

7개의 스킬, 1개의 커맨드, 1개의 에이전트를 포함하는 Python/FastAPI 풀스택 개발 플러그인입니다.

#### Skills (7개)

| Skill | Description |
|-------|-------------|
| **uv** | Python 패키지/프로젝트 매니저 (init, add, sync, lock, run) |
| **uvicorn** | ASGI 서버 설정 및 프로덕션 배포 |
| **fastapi** | FastAPI 앱 스캐폴딩, 라우팅, 미들웨어, DI |
| **async-patterns** | Python async/await 패턴 및 동시성 |
| **pydantic** | 데이터 검증, 직렬화, 모델 정의 |
| **langchain-langgraph** | LangChain/LangGraph 에이전트 워크플로 |
| **docker-build** | Dockerfile, 멀티스테이지 빌드, Docker Compose |

#### Command

- `/scaffold-fastapi` - FastAPI 프로젝트 전체 스캐폴딩 (uv + FastAPI + Pydantic + Docker + 선택적 LangChain)

#### Agent

- **fastapi-reviewer** - Python/FastAPI 코드 리뷰 전문 에이전트 (async 정확성, 보안, Pydantic 패턴 검증)

## 설치 방법

### 마켓플레이스 등록

`~/.claude/plugins/known_marketplaces.json`에 다음을 추가:

```json
{
  "claude-python-fastapi-marketplace": {
    "source": {
      "source": "github",
      "repo": "ingpdw/claude-python-fastapi-marketplace"
    }
  }
}
```

### 플러그인 설치

```
/plugin install python-fastapi-stack@claude-python-fastapi-marketplace
```

### 로컬 테스트

```bash
git clone https://github.com/ingpdw/claude-python-fastapi-marketplace.git
claude --plugin-dir ./claude-python-fastapi-marketplace/plugins/python-fastapi-stack
```

## 프로젝트 구조

```
.
├── .claude-plugin/
│   └── marketplace.json              # 마켓플레이스 메타데이터
├── plugins/
│   └── python-fastapi-stack/         # 메인 플러그인
│       ├── .claude-plugin/
│       │   └── plugin.json
│       ├── commands/
│       │   └── scaffold-fastapi.md   # /scaffold-fastapi
│       ├── agents/
│       │   └── fastapi-reviewer.md   # 코드 리뷰 에이전트
│       └── skills/
│           ├── uv/                   # uv 패키지 매니저
│           ├── uvicorn/              # ASGI 서버
│           ├── fastapi/              # FastAPI 프레임워크
│           ├── async-patterns/       # 비동기 패턴
│           ├── pydantic/             # 데이터 검증
│           ├── langchain-langgraph/  # AI 에이전트
│           └── docker-build/         # Docker 컨테이너화
└── README.md
```

## License

MIT
