---
title: hf-translation-hub MCP Servers
emoji: 🤗
colorFrom: yellow
colorTo: yellow
pinned: false
---

<div align="center">

# 🌐 Hugging Face Translation Hub – MCP Servers
<img width="4175" height="2156" alt="image" src="https://github.com/user-attachments/assets/643135f8-93a8-49ff-afb1-5024a004ef8f" />
<img width="1042" height="670" alt="image" src="https://github.com/user-attachments/assets/607a6cbb-cf8f-411d-9321-73b865434f94" />

*Composable Model Context Protocol (MCP) servers for Hugging Face documentation translation*

![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)
![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-blue)

**Search missing docs • Translate with context • Review quality • Auto-create GitHub PRs**

> 🎯 **Created to address [Hugging Face Transformers Issue #20179](https://github.com/huggingface/transformers/issues/20179)**  
> Improving and streamlining documentation internationalization workflows  

</div>

---

## ✨ What is this?

**hf-translation-hub MCP Servers** is a collection of **Model Context Protocol (MCP)–compliant servers**
designed to automate and standardize **Hugging Face documentation translation workflows**.

Rather than providing a single, monolithic UI agent, this project exposes **translation-related capabilities as independent MCP servers**.  
These servers can be orchestrated by external AI agents to build flexible, scalable, and inspectable translation pipelines.

Inspired by the original *i18n-agent*, this project focuses on **agent-native, server-side tooling**.

The MCP servers enable agents to:

- 🔍 **Discover** untranslated or outdated documentation files
- 🤖 **Translate** technical documents with context awareness and structure preservation
- 🧪 **Review** translation quality, terminology, and formatting consistency
- 📝 **Create** GitHub pull requests automatically

---

## 🎯 Why MCP Servers?

Documentation translation is not a single action—it is a **workflow**.

Based on the project overview, the core problem identified was that traditional approaches
bundle discovery, translation, review, and PR creation into tightly coupled systems,
making them hard to reuse, automate, or adapt.

This project adopts a **multi-server MCP architecture** to solve that problem:

- Each MCP server has a **single, well-defined responsibility**
- Agents can **compose only the steps they need**
- Each step is **explicit, inspectable, and retryable**
- Workflows can evolve without rewriting the entire system

This design supports:

- 🤖 Fully autonomous translation agents
- 🧑‍💻 Human-in-the-loop translation workflows
- 📦 Batch and incremental localization pipelines
- 🔁 Continuous documentation updates

---

## 🌍 Supported Languages

<div align="center">

| Language | Code | Status |
|---------|------|--------|
| 🇰🇷 Korean | `ko` | ✅ Supported |
| 🇯🇵 Japanese | `ja` | ✅ Supported |
| 🇨🇳 Chinese | `zh` | ✅ Supported |
| 🇫🇷 French | `fr` | ✅ Supported |
| 🇩🇪 German | `de` | ✅ Supported |

*Additional languages may be added incrementally.*

</div>

---

## 🧭 Available MCP Servers

<div align="center">

| MCP Server | Role |
|-----------|------|
| **Docs Explorer MCP** | Discover documentation files and analyze translation status |
| **Translation MCP** | Translate documents while preserving markdown and code structure |
| **PR Review MCP** | Review translation quality, terminology, and structural consistency |
| **PR Creation MCP** | Create GitHub branches, commits, and pull requests |

</div>

Each MCP server is **independently deployable** and **independently usable**.

---

## 🎥 Demo Video

[Hugging Face Translation Hub MCP – End-to-End Demo](https://drive.google.com/file/d/1TqjEYHqbPbPzwAq_a45htm3wIBJ0CI4R/view?usp=sharing)  
*Full walkthrough showing discovery → translation → PR creation → review*

---

## 🚀 Quick Start (MCP Usage)

These MCP servers can be used from any MCP-compatible client, including:

- Claude Desktop (MCP)
- HuggingChat MCP
- Custom Python / Node.js agents
- n8n MCP workflows


> The example below demonstrates a typical discovery interaction with the Docs Explorer MCP.

<details>
<summary>Show MCP request example</summary>

```json
{
  "jsonrpc": "2.0",
  "id": "call-1",
  "method": "tools/call",
  "params": {
    "name": "hf_translation_docs_explorer_search_files",
    "arguments": {
      "project": "transformers",
      "lang": "ko",
      "limit": 10,
      "include_status_report": true
    }
  }
}
```

</details>

> Translation and GitHub PR creation are handled by their respective MCP servers.
> For authoritative tool names, parameters, and end-to-end examples, please refer to the project overview: [https://hugging-face-krew.github.io/translation-mcp-project-overview/](https://hugging-face-krew.github.io/translation-mcp-project-overview/)


---

## 🔁 How It Works

<img width="5836" height="5412" alt="image" src="https://github.com/user-attachments/assets/431ddf2e-ce3f-48d7-adad-3382be4bb2f4" />

Each step is optional.
Agents may stop after translation, insert custom validation logic, or replace steps as needed.

---

## 🧠 Design & Architecture

Detailed design rationale, MCP boundaries, and architectural decisions are documented in:

* **[Translation MCP Project Overview](https://hugging-face-krew.github.io/translation-mcp-project-overview/)**
  
* **[MCP Server Design & Tooling](https://hugging-face-krew.github.io/hf_translation_hub_mcp_design_and_tooling/)**

* **[MCP Server Usage Guide](https://hugging-face-krew.github.io/hf-translation-hub-mcp-server-usage-guide/)**

This README focuses on **what the servers do and how to use them**.
For the deeper design motivation and trade-offs, please refer to the links above.

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge\&logo=python\&logoColor=ffdd54)
![Gradio](https://img.shields.io/badge/gradio-FF6B35?style=for-the-badge\&logo=gradio\&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge\&logo=github\&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-blue?style=for-the-badge)

</div>

---

## 💡Use Cases
* **[n8n-Based Translation Automation](https://hugging-face-krew.github.io/hf-translation-mcp-n8n/)**

Integrate MCP servers into an n8n workflow to automate document discovery, translation, and GitHub PR creation on a schedule or event basis.  
This enables fully automated, repeatable translation pipelines without manual intervention.

---

## 🤝 Contributing

Contributions are welcome and encouraged.

**Ways to contribute:**

* Add new MCP servers or tools
* Improve translation or review logic
* Extend language support
* Improve documentation and examples

**Guidelines:**

* Keep MCP servers single-responsibility
* Avoid hidden cross-server side effects
* Preserve tool input/output contracts



