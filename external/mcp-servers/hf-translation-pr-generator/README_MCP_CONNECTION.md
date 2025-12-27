# Claude Code MCP 연결 가이드

## 🔌 MCP 서버 연결 설정

### 1. Claude Desktop 설정 파일 위치
```
# macOS
~/Library/Application Support/Claude/claude_desktop_config.json

# Windows  
%APPDATA%/Claude/claude_desktop_config.json

# Linux
~/.config/Claude/claude_desktop_config.json
```

### 2. 설정 파일 내용 추가

기존 설정 파일에 다음 내용을 추가하거나, 없다면 새로 생성:

```json
{
  "mcpServers": {
    "hf-translation-pr-generator": {
      "command": "python",
      "args": [
        "/Users/mjjwa/Documents/GitHub/hyeonseo/hf_translation_hub/external/mcp-servers/hf-translation-pr-generator/app.py"
      ],
      "env": {
        "GITHUB_TOKEN": "your-actual-github-token-here",
        "PORT": "7864"
      }
    }
  }
}
```

### 3. GitHub Token 설정

1. GitHub에서 Personal Access Token 생성
   - Settings → Developer settings → Personal access tokens
   - 권한: `repo`, `workflow` 선택

2. 설정 파일의 `GITHUB_TOKEN`에 실제 토큰 입력

### 4. Claude Desktop 재시작

설정 변경 후 Claude Desktop을 완전히 종료하고 다시 시작

## 🛠️ 사용 가능한 MCP Tools

연결 후 다음 5개 도구를 사용할 수 있습니다:

### 1. validate_pr_config
```
GitHub 설정 검증
입력: owner, repo_name, project
```

### 2. search_reference_pr  
```
참조 PR 검색
입력: target_language, context
```

### 3. analyze_translation
```
번역 컨텐츠 분석
입력: filepath, translated_content, target_language, project
```

### 4. generate_pr_draft
```
PR 드래프트 생성
입력: filepath, translated_content, target_language, reference_pr_url, project
```

### 5. create_github_pr
```
GitHub PR 생성 (시뮬레이션)
입력: owner, repo_name, filepath, translated_content, target_language, reference_pr_url, project, pr_title, pr_description
```

## 📋 사용 예시

Claude Code에서 다음과 같이 사용:

```
hf-translation-pr-generator의 validate_pr_config를 사용해서 
myusername/transformers 저장소가 transformers 프로젝트에 대해 
유효한지 확인해줘
```

## ⚠️ 주의사항

1. **보안**: GitHub token을 설정 파일에 저장할 때 주의
2. **포트**: 7864 포트가 사용 중이면 다른 포트로 변경
3. **경로**: Python과 파일 경로가 정확한지 확인
4. **재시작**: 설정 변경 후 반드시 Claude Desktop 재시작

## 🔍 연결 확인

Claude Code에서 다음 명령으로 연결 상태 확인:

```
MCP 서버가 연결되었나요? 사용 가능한 도구들을 알려주세요.
```