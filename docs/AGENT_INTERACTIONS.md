# 🔄 Agent Interaction Diagram

---

# ✅ Final Architecture: 9 Agents

### ✅ Complete List of Agents

| № | Agent                    | Port | Purpose                                        |
|---|--------------------------|------|------------------------------------------------|
| 1 | PlannerAgent             | 8301 | Task planning, decomposition, Jira integration |
| 2 | ProgressAgent            | 8302 | Progress and velocity analysis                 |
| 3 | RisksAgent               | 8303 | Risk analysis                                  |
| 4 | DigestAgent              | 8304 | Daily / weekly summaries                       |
| 5 | ArchitectureIntelligence | 8305 | Multimodal diagram & architecture analysis     |
| 6 | HealthMonitorAgent       | 8306 | Health checks + circuit breaker                |
| 7 | MetricsAgent             | 8307 | Prometheus / business metrics                  |
| 8 | MarathonAgent            | 8308 | Long-running tasks with self-correction        |
| 9 | CodeExecutionAgent       | 8309 | Autonomous code generation, testing & debug    |

---

# 🔄 Architecture Diagram (9 Agents)

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        n8n Orchestrator                                                            │
│                                 (Workflow Automation Layer)                                                        │
└─────┬────────────┬──────  ────┬───────────┬───────────┬──────────────┬───────────┬─────────────────────────────────┘
      │            │            │           │           │              │           │           │           │
 ┌────▼────┐  ┌────▼────┐   ────▼────┐ ┌────▼────┐ ┌────▼───────┐ ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ ─────▼───── 
 │ Planner │  │ Progress│  │  Risks  │ │  Digest │ │Architecture│ │  Health │ │ Metrics │ │ Marathon│ │  CodeEx │
 │ :8301   │  │ :8302   │  │ :8303   │ │ :8304   │ │Intelligence│ │ :8306   │ │ :8307   │ │ :8308   │ │ :8309   │
 │         │  │         │  │         │ │         │ │  :8305     │ │         │ │         │ │         │ │         │
 └────┬────┘  └────┬────┘  └────┬────┘ └────┬────┘ └────┬───────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
      │            │            │           │           │              │           │           │           │
      └─────────────────────────┴───────────┴───────────┴──────────────┴───────────┴───────────┴───────────┘
                                                        │
                                        ┌───────────────▼───────────────┐
                                        │          Shared Layer         │
                                        │───────────────────────────────│
                                        │ • MCP Protocol                │
                                        │ • LLM Client (Gemini)         │
                                        │ • Jira Client                 │
                                        │ • Vision Provider             │
                                        │ • Error Handler               │
                                        │ • Circuit Breaker             │
                                        │ • Metrics Core                │
                                        └───────────────────────────────┘
```
---

# 🧠 Role of Each Agent (Detailed Description)

## 1️⃣ Planner Agent (8301)
**Purpose:** Intelligent task decomposition and planning.  
**Responsibilities:**
- Accepts high-level task description
- Generates detailed subtasks with reasoning chain
- Predictive estimation based on historical data
- Risk-aware planning
- Optional Jira task creation (epics + subtasks)
**Used when:** New feature request, GitHub issue, workflow start.

## 2️⃣ Progress Agent (8302)
**Purpose:** Project progress and velocity tracking.  
**Functions:**
- Analyze Jira issues and commits
- Calculate velocity and completion rate
- Classify project status (excellent / good / at_risk / critical)

## 3️⃣ Risks Agent (8303)
**Purpose:** Security and technical risk assessment.  
**Analyzes:**
- Security vulnerabilities
- Performance risks
- Technical debt
- Dependency issues  
**Outputs:** Prioritized risks + mitigation strategies.

## 4️⃣ Digest Agent (8304)
**Purpose:** Human-readable report generation.  
**Aggregates:**
- Progress, risks, achievements
- Blockers and mood  
**Formats:** Daily/weekly digests for Slack/Email.

## 5️⃣ Architecture Intelligence Agent (8305)
**Purpose:** Multimodal analysis of diagrams and screenshots.  
**Capabilities:**
- Architecture diagram understanding (Gemini Vision)
- Detect bottlenecks, anti-patterns, scalability issues
- Provide improvement recommendations
- Analyze UI mockups and infrastructure schemes

## 6️⃣ Health Monitor Agent (8306)
**Purpose:** System health and resilience.  
**Functions:**
- Periodic health checks of all agents
- Circuit breaker management
- Anomaly detection
- Alert recommendations

## 7️⃣ Metrics Agent (8307)
**Purpose:** Observability and monitoring.  
**Provides:**
- Prometheus-compatible `/metrics` endpoint
- Request counts, latency, error rates
- Per-agent and system-wide metrics

## 8️⃣ Marathon Agent (8308)
**Purpose:** Long-running complex tasks with autonomy.  
**Features:**
- Maintains long context across multiple LLM calls
- Self-correction loops
- Adaptive planning and re-planning
- Handles research, multi-step reasoning, prolonged analysis

## 9️⃣ Code Execution Agent (8309)
**Purpose:** Autonomous code generation and verification (Vibe Engineering).  
**Capabilities:**
- Generate production-ready code + test cases
- Execute tests in isolated sandbox
- Autonomous debugging loop (self-correction)
- Code quality and security review
- Production-readiness assessment

---

# 🔁 End-to-End Scenario (Full Workflow)

```text
GitHub / User / Manual Trigger
│
▼
n8n Workflow Starts
│
├─▶ PlannerAgent
│   └─ Task decomposition + predictive planning + Jira issues
│
├─▶ RisksAgent
│   └─ Risk evaluation of planned changes
│
├─▶ ArchitectureIntelligence
│   └─ Diagram/architecture review (multimodal)
│
├─▶ CodeExecutionAgent
│   └─ Generate & autonomously test implementation code
│
├─▶ MarathonAgent (if needed)
│   └─ Long-running research or complex coordination
│
├─▶ ProgressAgent
│   └─ Current velocity + completion status
│
├─▶ DigestAgent
│   └─ Final human-readable summary/report
│
├─▶ MetricsAgent
│   └─ Record workflow metrics
│
└─▶ HealthMonitorAgent (background/parallel)
    ├─ Continuous agent health pings
    ├─ Circuit breaker status
    └─ System-wide alerts if needed
```

---

# 🧩 Architectural Advantages

### ✅ SOLID + Micro-agent Design
- Each agent has single responsibility
- Independent deployment and scaling
- Easy testing and maintenance

### ✅ Observability-First Approach
- Dedicated Metrics and Health agents
- No monitoring clutter in business logic

### ✅ Fault Isolation & Resilience
- Circuit breakers per agent
- One failing agent doesn’t crash the system

### ✅ Unified Contract (MCP Protocol)
- All agents expose `/mcp` with identical schema
- Transparent orchestration in n8n

### ✅ Advanced Gemini Usage
- Text reasoning + Vision (multimodal)
- Long context (Marathon)
- Autonomous loops (Code Execution)

This architecture is production-ready, resilient, and fully leverages Gemini capabilities across the entire DevOps lifecycle.