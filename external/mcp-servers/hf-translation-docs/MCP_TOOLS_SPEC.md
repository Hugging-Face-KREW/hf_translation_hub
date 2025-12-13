# HuggingFace Translation Documentation MCP Server Tools Specification

## Overview
MCP Server for HuggingFace documentation translation with streamable HTTP implementation.
Client performs translation while server handles file operations and workflow management.

## Tools Definition

### 1. search_translation_files
**Description:** Search for files that need translation in a HuggingFace project

**Input:**
```json
{
  "project": "transformers",           // Project name (transformers, diffusers, etc.)
  "target_language": "ko",             // Target language code (ko, zh, ja, etc.)
  "max_files": 10                      // Maximum number of files to return
}
```

**Output:**
```json
{
  "status": "success",
  "data": {
    "project": "transformers",
    "target_language": "ko",
    "total_found": 156,
    "files": [
      {
        "path": "docs/source/en/model_doc/bert.md",
        "size": 15420,
        "last_modified": "2024-01-15T10:30:00Z",
        "priority": "high",
        "translation_status": "missing"
      }
    ],
    "statistics": {
      "missing": 145,
      "outdated": 11,
      "up_to_date": 0
    }
  }
}
```

### 2. get_file_content
**Description:** Retrieve original file content for translation

**Input:**
```json
{
  "project": "transformers",
  "file_path": "docs/source/en/model_doc/bert.md",
  "include_metadata": true
}
```

**Output:**
```json
{
  "status": "success",
  "data": {
    "file_path": "docs/source/en/model_doc/bert.md",
    "content": "# BERT\n\n<Tip>\n\nThis is the BERT model...",
    "metadata": {
      "encoding": "utf-8",
      "size": 15420,
      "last_modified": "2024-01-15T10:30:00Z",
      "content_hash": "sha256:abc123..."
    },
    "processed_content": {
      "to_translate": "BERT\n\nThis is the BERT model...",
      "code_blocks_removed": 3,
      "tables_removed": 1
    }
  }
}
```

### 3. generate_translation_prompt
**Description:** Generate optimized translation prompt for the content

**Input:**
```json
{
  "target_language": "ko",
  "content": "# BERT\n\nThis is the BERT model...",
  "additional_instruction": "Use technical terms consistently",
  "project": "transformers",
  "file_path": "docs/source/en/model_doc/bert.md"
}
```

**Output:**
```json
{
  "status": "success",
  "data": {
    "prompt": "You are a professional technical translator...",
    "context": {
      "target_language_name": "Korean",
      "content_type": "technical_documentation",
      "domain": "machine_learning",
      "file_type": "model_documentation"
    },
    "guidelines": [
      "Preserve markdown formatting",
      "Keep technical terms in English where appropriate",
      "Maintain code block integrity"
    ]
  }
}
```

### 4. validate_translation
**Description:** Validate translated content for quality and formatting

**Input:**
```json
{
  "original_content": "# BERT\n\nThis is the BERT model...",
  "translated_content": "# BERT\n\n이것은 BERT 모델입니다...",
  "target_language": "ko",
  "file_path": "docs/source/en/model_doc/bert.md"
}
```

**Output:**
```json
{
  "status": "success",
  "data": {
    "is_valid": true,
    "quality_score": 0.95,
    "issues": [],
    "suggestions": [
      {
        "type": "terminology",
        "message": "Consider using '모델' consistently for 'model'",
        "line": 3
      }
    ],
    "formatting": {
      "markdown_valid": true,
      "links_preserved": true,
      "code_blocks_intact": true
    }
  }
}
```

### 5. save_translation_result
**Description:** Save translation result to file system

**Input:**
```json
{
  "project": "transformers",
  "original_file_path": "docs/source/en/model_doc/bert.md",
  "translated_content": "# BERT\n\n이것은 BERT 모델입니다...",
  "target_language": "ko",
  "metadata": {
    "translator": "claude-3.5-sonnet",
    "translation_date": "2024-01-20T14:30:00Z",
    "additional_instruction": "Use technical terms consistently"
  }
}
```

**Output:**
```json
{
  "status": "success",
  "data": {
    "saved_path": "/path/to/translation_result/docs/source/ko/model_doc/bert.md",
    "backup_path": "/path/to/backup/bert_20240120_143000.md",
    "file_size": 16840,
    "checksum": "sha256:def456...",
    "created_directories": ["docs/source/ko/model_doc"]
  }
}
```

### 6. create_github_pr
**Description:** Create GitHub Pull Request for translation

**Input:**
```json
{
  "github_config": {
    "token": "ghp_...",
    "owner": "user-fork",
    "repo_name": "transformers",
    "reference_pr_url": "https://github.com/huggingface/transformers/pull/12345"
  },
  "translation_data": {
    "file_path": "docs/source/en/model_doc/bert.md",
    "target_language": "ko",
    "translated_content": "# BERT\n\n이것은 BERT 모델입니다...",
    "en_title": "BERT"
  },
  "project": "transformers"
}
```

**Output:**
```json
{
  "status": "success",
  "data": {
    "pr_url": "https://github.com/user-fork/transformers/pull/123",
    "pr_number": 123,
    "branch_name": "add-korean-bert-docs",
    "commit_hash": "abc1234567890",
    "files_changed": [
      "docs/source/ko/model_doc/bert.md",
      "docs/source/ko/_toctree.yml"
    ],
    "pr_details": {
      "title": "Add Korean translation for BERT documentation",
      "body": "This PR adds Korean translation for BERT model documentation...",
      "reviewers": []
    }
  }
}
```

### 7. get_project_config
**Description:** Get project-specific configuration and settings

**Input:**
```json
{
  "project": "transformers"
}
```

**Output:**
```json
{
  "status": "success",
  "data": {
    "project": "transformers",
    "repo_url": "https://github.com/huggingface/transformers",
    "docs_path": "docs/source",
    "supported_languages": ["ko", "zh", "ja", "es", "fr"],
    "reference_pr_url": "https://github.com/huggingface/transformers/pull/12345",
    "translation_guidelines": {
      "preserve_code_blocks": true,
      "keep_english_terms": ["API", "token", "embedding"],
      "style_guide_url": "https://..."
    }
  }
}
```

## Streaming Implementation

All tools support streaming responses for better UX:

```http
GET /tools/{tool_name}
Content-Type: application/json
Accept: text/event-stream

Response:
data: {"type": "progress", "message": "Searching files...", "progress": 0.3}

data: {"type": "partial", "data": {"files": [...partial_results...]}}

data: {"type": "complete", "data": {...final_result...}}
```

## Error Handling

Standard error response format:
```json
{
  "status": "error",
  "error": {
    "code": "FILE_NOT_FOUND",
    "message": "The specified file could not be found",
    "details": {
      "file_path": "docs/source/en/model_doc/bert.md",
      "project": "transformers"
    }
  }
}
```

## Workflow Integration

1. Client calls `search_translation_files` → gets file list
2. Client calls `get_file_content` → gets original content  
3. Client calls `generate_translation_prompt` → gets optimized prompt
4. **Client performs translation using LLM** ← Key difference
5. Client calls `validate_translation` → checks quality
6. Client calls `save_translation_result` → saves result
7. Client calls `create_github_pr` → creates PR

This architecture separates concerns: MCP server handles file operations, client handles translation.