# MCP Tools Specification for HuggingFace Translation PR Generator

## Overview
이 MCP 서버는 approve_handler를 대체하여 GitHub PR 생성 프로세스를 처리합니다. 기존의 LLM completion을 내부에서 수행하는 대신, MCP client(Claude)가 직접 LLM completion을 수행하도록 설계되었습니다.

## Architecture
- **MCP Server**: GitHub API 호출, 설정 검증, 메타데이터 생성
- **MCP Client (Claude)**: Reference PR 분석, PR 제목/설명 생성, 번역 품질 검증

## MCP Tools

### 1. validate_pr_config
**Endpoint**: `pr_validate_config`
**Purpose**: GitHub PR 생성에 필요한 설정 검증

**Input Schema**:
```json
{
  "owner": "string", 
  "repo_name": "string",
  "project": "string"
}
```

**Environment Variables Required**:
- `GITHUB_TOKEN`: GitHub Personal Access Token

**Output Schema**:
```json
{
  "status": "success|error",
  "data": {
    "is_valid": "boolean",
    "github_config": {
      "token_valid": "boolean",
      "repo_access": "boolean",
      "fork_exists": "boolean",
      "permissions": ["read", "write", "admin"]
    },
    "project_config": {
      "base_repo_url": "string",
      "docs_path": "string",
      "supported_languages": ["string"]
    },
    "missing_requirements": ["string"],
    "recommendations": ["string"]
  }
}
```

### 2. search_reference_pr
**Endpoint**: `pr_search_reference`
**Purpose**: 번역 관련 참조 PR 검색 (LLM completion 없이 GitHub API만 사용)

**Input Schema**:
```json
{
  "target_language": "string",
  "context": "string"
}
```

**Output Schema**:
```json
{
  "status": "success|error",
  "data": {
    "reference_prs": [
      {
        "url": "string",
        "title": "string",
        "description": "string",
        "files_changed": ["string"],
        "language": "string",
        "score": "number",
        "created_at": "string"
      }
    ],
    "search_metadata": {
      "total_found": "number",
      "search_criteria": "object",
      "search_time": "string"
    },
    "suggestion": "Use MCP client to analyze these PRs and select the best reference"
  }
}
```

### 3. analyze_translation
**Endpoint**: `pr_analyze_translation`
**Purpose**: 번역된 컨텐츠 분석 및 메타데이터 생성

**Input Schema**:
```json
{
  "filepath": "string",
  "translated_content": "string", 
  "target_language": "string",
  "project": "string"
}
```

**Output Schema**:
```json
{
  "status": "success|error",
  "data": {
    "file_analysis": {
      "original_path": "string",
      "target_path": "string", 
      "file_type": "string",
      "size_comparison": {
        "original_size": "number",
        "translated_size": "number", 
        "size_ratio": "number"
      }
    },
    "content_analysis": {
      "markdown_structure": {
        "headers_count": "number",
        "code_blocks_count": "number",
        "links_count": "number",
        "tables_count": "number"
      },
      "translation_quality": {
        "completeness": "number",
        "formatting_preserved": "boolean",
        "code_integrity": "boolean"
      }
    },
    "pr_metadata": {
      "suggested_branch_name": "string",
      "file_category": "string",
      "priority": "high|medium|low"
    },
    "context_for_llm": {
      "file_description": "string",
      "translation_guidelines": ["string"],
      "formatting_notes": ["string"]
    }
  }
}
```

### 4. generate_pr_draft
**Endpoint**: `pr_generate_draft`
**Purpose**: PR 생성을 위한 기본 구조 및 메타데이터 준비

**Input Schema**:
```json
{
  "filepath": "string",
  "translated_content": "string",
  "target_language": "string", 
  "reference_pr_url": "string",
  "project": "string"
}
```

**Output Schema**:
```json
{
  "status": "success|error",
  "data": {
    "pr_structure": {
      "branch_name": "string",
      "target_file_path": "string",
      "base_branch": "string"
    },
    "file_changes": [
      {
        "action": "create|modify",
        "path": "string",
        "content": "string"
      }
    ],
    "reference_analysis": {
      "reference_pr_url": "string",
      "title_pattern": "string",
      "description_template": "string",
      "common_elements": ["string"]
    },
    "llm_prompts": {
      "title_generation_prompt": "string",
      "description_generation_prompt": "string",
      "context": "object"
    },
    "toctree_updates": {
      "required": "boolean",
      "target_files": ["string"],
      "updates": ["object"]
    }
  }
}
```

### 5. create_github_pr
**Endpoint**: `pr_create_github_pr`  
**Purpose**: 실제 GitHub PR 생성 및 파일 업로드

**Input Schema**:
```json
{
  "github_token": "string",
  "owner": "string",
  "repo_name": "string", 
  "filepath": "string",
  "translated_content": "string",
  "target_language": "string",
  "reference_pr_url": "string",
  "project": "string",
  "pr_title": "string",
  "pr_description": "string",
  "metadata": "object"
}
```

**Output Schema**:
```json
{
  "status": "success|partial_success|error",
  "data": {
    "pr_url": "string",
    "branch_name": "string", 
    "files_created": [
      {
        "path": "string",
        "status": "created|updated",
        "commit_sha": "string"
      }
    ],
    "pr_details": {
      "number": "number",
      "title": "string", 
      "description": "string",
      "state": "open|closed",
      "created_at": "string"
    },
    "toctree_status": {
      "updated": "boolean",
      "files_modified": ["string"],
      "commit_sha": "string"
    },
    "additional_info": {
      "existing_pr_updated": "boolean",
      "conflicts": ["string"],
      "warnings": ["string"]
    }
  }
}
```

## Workflow
1. **validate_pr_config** - 설정 검증
2. **search_reference_pr** - 참조 PR 후보 검색  
3. **MCP Client Analysis** - 참조 PR 분석 및 최적 선택
4. **analyze_translation** - 번역 컨텐츠 분석
5. **generate_pr_draft** - PR 구조 생성
6. **MCP Client Generation** - PR 제목/설명 생성
7. **create_github_pr** - GitHub PR 생성

## Key Differences from approve_handler
- ❌ **제거된 기능**: 내부 LLM completion (find_reference_pr_simple_stream)
- ✅ **새로운 방식**: MCP client가 참조 PR 분석 및 PR 제목/설명 생성
- ✅ **유지된 기능**: GitHub API 호출, 파일 업로드, toctree 업데이트
- ✅ **개선된 기능**: 더 상세한 메타데이터 및 컨텍스트 제공