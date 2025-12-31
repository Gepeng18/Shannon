# 《任务、会话、代理、工作流：Shannon的AI协作四重奏》

> **专栏语录**：在AI系统的江湖中，任务、会话、代理、工作流这四个概念如同四位绝世高手，各有绝技却又相辅相成。理解它们之间的关系，就如同掌握了AI系统的内功心法。本文将带你深入Shannon的源码，揭秘这四位高手是如何通过精心设计的协议和数据流，实现完美的协作。

## 第一章：一个问题引发的AI革命

### 从"Hello World"到"深度思考"

让我们从一个看似简单的问题开始："帮我分析一下苹果公司的最新财报，重点关注AI投资情况。"

在传统AI系统中，这个问题可能会这样处理：

**这块代码展示了什么？**

这段代码展示了从"Hello World"到"深度思考"的核心实现。背景是：现代AI系统需要处理复杂的业务逻辑和技术挑战，这个代码示例演示了具体的解决方案和技术实现。

这段代码的目的是说明如何通过编程实现特定的功能需求和技术架构。

```python
# 传统AI系统的处理方式
def handle_user_query(query: str) -> str:
    # 1. 直接调用大模型
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )

    # 2. 返回结果
    return response.choices[0].message.content

# 调用
result = handle_user_query("分析苹果财报AI投资")
print(result)
```

**但是Shannon的处理方式完全不同**：

**这块代码展示了什么？**

这段代码展示了Shannon的核心协作流程，包括任务包装、会话增强、代理编排和工作流执行四个阶段。背景是：复杂AI任务需要多层次的处理，从简单的查询理解到复杂的多代理协作，Shannon通过分层架构实现了从简单到复杂的平滑扩展。

这段代码的目的是说明Shannon如何将传统的一次性AI调用转换为结构化的多阶段协作过程。

```go
// Shannon的多阶段协作处理
func ProcessComplexQuery(query TaskInput) (*TaskResult, error) {
    // 阶段1：任务包装 - 将用户问题转换为结构化任务
    task := NewTask(query)

    // 阶段2：会话增强 - 注入历史上下文和用户偏好
    enrichedTask := sessionManager.EnrichWithContext(task)

    // 阶段3：代理编排 - 分解为多个子任务，分配给不同代理
    workflow := agentOrchestrator.CreateWorkflow(enrichedTask)

    // 阶段4：工作流执行 - 协调多个代理的并行执行
    result := workflowEngine.Execute(workflow)

    return result, nil
}
```

**Shannon vs 传统AI：一场观念的革命**

| 维度 | 传统AI系统 | Shannon系统 |
|------|-----------|------------|
| **问题建模** | 字符串输入 | 结构化任务对象 |
| **上下文处理** | 单轮对话 | 持久化会话状态 |
| **执行模式** | 单模型调用 | 多代理协作 |
| **结果质量** | 一次生成 | 迭代优化 |
| **可观测性** | 黑盒调用 | 完整执行追踪 |

**为什么Shannon要如此复杂？** 因为当AI系统面对现实世界的复杂问题时，简单的"大模型调用"已经不够了。我们需要：

1. **任务抽象**：将模糊的用户意图转换为精确的执行计划
2. **会话连续性**：记住用户的偏好、历史对话和上下文
3. **代理分工**：不同专业能力的AI代理协同工作
4. **工作流编排**：协调复杂任务的执行顺序和依赖关系

本文将通过深度源码分析，揭示Shannon如何实现这四个核心概念的完美协作。

## 第二章：任务（Task）- AI系统的"问题翻译官"

### 任务抽象：从模糊意图到精确执行

在Shannon的设计哲学中，**任务不是简单的字符串，而是AI系统对用户意图的精确建模**。这种设计源于一个深刻的洞察：用户说的话和计算机理解的指令之间，存在巨大的语义鸿沟。

#### TaskInput的设计哲学：26个字段背后的故事

**这块代码展示了什么？**

这段代码定义了TaskInput结构体，包含26个字段用于描述AI任务的完整规范。背景是：AI任务处理需要考虑用户意图、上下文环境、安全约束、资源限制等多个维度，一个简单的字符串无法承载这些复杂信息。

这段代码的目的是说明如何通过结构化数据模型精确表达AI任务的所有方面。

```go
// go/orchestrator/internal/workflows/types.go

// TaskInput - Shannon任务的完整输入规范，定义了AI系统处理用户请求的所有参数
// 这个结构体是Shannon架构的核心抽象，将用户模糊意图转换为系统可精确执行的任务定义
type TaskInput struct {
    // ========== 核心身份标识 - 任务的基本定位信息 ==========
    // 以下四个字段是任务的必填核心标识，构成了任务的唯一身份和执行上下文
    Query     string `json:"query"`     // 用户原始查询文本 - 核心业务输入，经过长度验证(1-10000字符)
    UserID    string `json:"user_id"`    // 用户唯一标识 - 用于多租户隔离和个性化，格式如"usr_123456"
    TenantID  string `json:"tenant_id"`  // 租户标识 - 数据隔离的关键，格式如"tn_abcdef"
    SessionID string `json:"session_id"` // 会话标识 - 保证对话连续性，格式如"ses_xyz789"

    // ========== 执行环境增强 - 运行时上下文扩展 ==========
    // Context提供额外的执行参数，支持业务层面的自定义扩展
    Context   map[string]interface{} `json:"context,omitempty"` // 运行时参数映射 - 如{"timezone": "UTC+8", "language": "zh-CN"}

    // ========== 执行策略控制 - 任务处理模式选择 ==========
    // Mode字段决定任务的执行路径，直接影响系统资源使用和响应时间
    Mode      string `json:"mode,omitempty"` // 执行模式枚举值: "simple"(直接执行), "complex"(DAG编排), "research"(深度分析)

    // ========== 模板系统集成 - 声明式任务配置 ==========
    // 模板系统允许将复杂任务逻辑预定义，提高一致性和可维护性
    TemplateName    string `json:"template_name,omitempty"`    // 模板名称 - 如"financial_analysis", "code_review"
    TemplateVersion string `json:"template_version,omitempty"` // 模板版本 - 语义化版本如"v2.1.0"
    DisableAI       bool   `json:"disable_ai,omitempty"`       // AI禁用标志 - 为true时只执行预定义步骤，不调用LLM

    // ========== 会话连续性 - 多轮对话支持 ==========
    // 历史和会话状态是AI系统实现连续对话的关键
    History    []Message              `json:"history,omitempty"`    // 对话历史消息数组 - 包含最近N轮对话，格式[{"role": "user", "content": "..."}]
    SessionCtx map[string]interface{} `json:"session_ctx,omitempty"` // 会话持久化状态 - 如{"user_preferences": {...}, "conversation_topic": "finance"}

    // ========== 人工干预机制 - 安全和合规控制 ==========
    // 敏感任务需要人工审核，防止AI自主执行高风险操作
    RequireApproval bool `json:"require_approval,omitempty"` // 人工审核标志 - 为true时任务需等待人工批准
    ApprovalTimeout int  `json:"approval_timeout,omitempty"` // 审核超时时间(秒) - 默认300秒，过期自动拒绝

    // ========== 性能优化 - 执行效率控制 ==========
    // 这些字段允许根据任务特点进行性能调优
    BypassSingleResult bool `json:"bypass_single_result,omitempty"` // 单结果快速返回 - 为true时直接返回单个工具结果，跳过复杂编排

    // ========== 嵌套执行支持 - 复杂任务分解 ==========
    // 支持任务的递归嵌套，便于构建复杂的工作流层次结构
    ParentWorkflowID string `json:"parent_workflow_id,omitempty"` // 父工作流ID - 子任务关联到父工作流，便于追踪和取消

    // ========== 预分解信息 - LLM提示增强 ==========
    // 这些信息帮助LLM更好地理解任务意图，减少推理错误
    SuggestedTools   []string               `json:"suggested_tools,omitempty"`   // 建议使用工具列表 - 如["web_search", "data_analysis"]
    ToolParameters   map[string]interface{} `json:"tool_parameters,omitempty"` // 工具预设参数 - 如{"web_search": {"max_results": 5}}

    // ========== 资源控制 - 成本和资源限制 ==========
    // 防止单个任务消耗过多资源或产生高额费用
    MaxTokens      int     `json:"max_tokens,omitempty"`      // 最大token消耗限制 - 默认无限制，超过则截断
    BudgetLimitUSD float64 `json:"budget_limit_usd,omitempty"` // 美元预算上限 - 监控API调用成本，超过则终止

    // ========== 质量保证 - 输出质量控制 ==========
    // 确保AI输出满足最低质量标准
    MinConfidence float64 `json:"min_confidence,omitempty"` // 最小置信度阈值 - 0.0-1.0，低于此值重新生成
    RequireCitations bool  `json:"require_citations,omitempty"` // 引用要求标志 - 为true时输出必须包含来源引用

    // ========== 执行约束 - 时间和优先级管理 ==========
    // 控制任务执行的时间特性和调度优先级
    TimeoutSeconds int `json:"timeout_seconds,omitempty"` // 总执行超时(秒) - 默认300秒，防止任务无限运行
    Priority       int `json:"priority,omitempty"`        // 执行优先级 - 1-10，数字越大优先级越高

    // ========== 可观测性 - 调试和监控支持 ==========
    // 这些字段增强系统的可观测性和问题排查能力
    DebugMode      bool   `json:"debug_mode,omitempty"`      // 调试模式开关 - 为true时输出详细执行日志
    RequestID      string `json:"request_id,omitempty"`      // 全局请求追踪ID - 用于跨服务追踪，格式如"req_abc123"
    UserAgent      string `json:"user_agent,omitempty"`      // 客户端标识 - 如"WebApp/1.2.3"或"MobileApp/2.1.0"
    ClientIP       string `json:"client_ip,omitempty"`       // 客户端IP地址 - 用于地理位置分析和安全审计
}
```

**26个字段的设计权衡：功能 vs 复杂性**

Shannon的TaskInput有整整26个字段！这个设计决策饱受争议：

**支持者观点**：
- **完整性**：覆盖了AI任务执行的所有方面
- **可扩展性**：新需求可以通过新增字段满足
- **可配置性**：每个任务都可以精确控制执行行为

**批评者观点**：
- **复杂性**：学习成本高，新手难以掌握
- **维护负担**：字段之间可能存在隐含依赖
- **API污染**：客户端需要处理过多可选参数

**Shannon的设计哲学**：与其让任务抽象过于简单（导致执行时才发现问题），不如在一开始就把所有可能性都考虑清楚。

#### 核心字段的"强制要求"分析

```go
// 这四个字段为什么是必填的？
type TaskInput struct {
    Query     string `json:"query"`     // ❌ 不能为空 - 没有查询，何来任务？
    UserID    string `json:"user_id"`    // ❌ 不能为空 - 多租户基础
    TenantID  string `json:"tenant_id"`  // ❌ 不能为空 - 数据隔离关键
    SessionID string `json:"session_id"` // ❌ 不能为空 - 会话连续性保证
    // ...
}
```

**Query字段：任务的灵魂**
```go
// 为什么Query必须存在？
func ValidateTaskInput(input *TaskInput) error {
    if strings.TrimSpace(input.Query) == "" {
        return errors.New("query cannot be empty")
    }

    // 检查查询长度
    if len(input.Query) > 10000 {
        return errors.New("query too long")
    }

    // 检查查询质量
    if !containsMeaningfulWords(input.Query) {
        return errors.New("query must contain meaningful content")
    }

    return nil
}
```

**UserID和TenantID：多租户架构的基石**

**这块代码展示了什么？**

这段代码展示了多租户数据隔离的实现，每个租户拥有独立的存储空间。背景是：SaaS系统中需要严格的数据隔离，防止不同租户的数据泄露，同时支持水平扩展和资源共享。

这段代码的目的是说明如何通过租户ID实现数据隔离和访问控制。

```go
// 多租户数据隔离的实现
type MultiTenantStore struct {
    // 每个租户有独立的存储空间
    stores map[string]Store // tenant_id -> store
}

func (mts *MultiTenantStore) GetData(userID, tenantID string) (interface{}, error) {
    // 1. 验证用户属于该租户
    if !mts.validateUserTenant(userID, tenantID) {
        return nil, errors.New("user not in tenant")
    }

    // 2. 获取租户特定的存储
    store := mts.stores[tenantID]
    if store == nil {
        return nil, errors.New("tenant not found")
    }

    // 3. 在租户空间内查询数据
    return store.Get(userID)
}
```

#### 可选字段的"艺术"：平衡灵活性与复杂性

**Mode字段：字符串 vs 枚举的争议**
```go
// 为什么选择字符串而不是枚举？
type TaskInput struct {
    Mode string `json:"mode,omitempty"` // "simple", "complex", "research"
}

// 优势：运行时扩展，无需重新编译
func ProcessTaskByMode(task *TaskInput) error {
    switch task.Mode {
    case "simple":
        return processSimple(task)
    case "complex":
        return processComplex(task)
    case "research":
        return processResearch(task)
    case "custom_ai_agent":  // 新模式，无需修改代码
        return processCustomAIAgent(task)
    default:
        return processDefault(task) // 向后兼容
    }
}
```

**历史vs枚举的权衡**：

| 方法 | 优势 | 劣势 |
|------|------|------|
| 枚举 | 类型安全、IDE支持 | 需要重新编译、扩展困难 |
| 字符串 | 运行时扩展、向后兼容 | 拼写错误、运行时才发现 |

Shannon选择了字符串，因为AI系统的进化速度远超编译周期。

#### 模板系统的哲学：声明式 vs 命令式

```go
// 模板系统：声明式配置的威力
type TaskInput struct {
    TemplateName    string `json:"template_name,omitempty"`    // "financial_analysis"
    TemplateVersion string `json:"template_version,omitempty"` // "v2.1"
    DisableAI       bool   `json:"disable_ai,omitempty"`       // 纯模板模式
}

// 模板定义示例
financialAnalysisTemplate := Template{
    Name: "financial_analysis",
    Version: "v2.1",
    Steps: []Step{
        {
            Name: "data_collection",
            Tool: "web_scraper",
            Parameters: map[string]interface{}{
                "url": "{{.Query}}",  // 从任务查询中提取URL
                "format": "json",
            },
        },
        {
            Name: "analysis",
            Tool: "ai_analyzer",
            Parameters: map[string]interface{}{
                "model": "gpt-4",
                "prompt": "分析这份财务数据：{{.data_collection.output}}",
            },
        },
    },
}
```

**模板系统的三重优势**：
1. **一致性**：相同类型任务使用相同流程
2. **可维护性**：模板更新自动应用到所有任务
3. **可审计性**：模板版本控制和变更追踪

#### 资源控制：预算与质量的平衡

```go
// 预算控制的实现
type BudgetController struct {
    limits map[string]float64 // user_id -> daily_limit_usd
}

func (bc *BudgetController) CheckAndReserve(task *TaskInput) error {
    userLimit := bc.limits[task.UserID]
    currentUsage := bc.getCurrentDayUsage(task.UserID)

    // 检查预算
    if currentUsage + task.BudgetLimitUSD > userLimit {
        return errors.New("budget exceeded")
    }

    // 预留预算
    return bc.reserveBudget(task.UserID, task.BudgetLimitUSD)
}
```

**Shannon的资源控制哲学**：在AI时代，**计算资源就是金钱**。每个任务都必须有明确的预算限制。

#### TaskResult：执行结果的完整建模

```go
// TaskResult - 任务执行的"成绩单"
type TaskResult struct {
    // 执行标识
    TaskID      string `json:"task_id"`      // 任务唯一标识
    WorkflowID  string `json:"workflow_id"`  // 工作流实例ID
    Status      string `json:"status"`       // 执行状态: completed/failed/timeout

    // 核心结果
    Result      string `json:"result"`       // 主要结果文本
    ResultType  string `json:"result_type"`  // 结果类型: text/json/html

    // 结构化输出
    StructuredOutput map[string]interface{} `json:"structured_output,omitempty"`

    // 质量指标
    Citations   []Citation `json:"citations,omitempty"` // 引用来源
    Confidence  float64    `json:"confidence,omitempty"` // 结果置信度

    // 执行统计
    TokenUsage  TokenUsage `json:"token_usage"`   // 令牌使用统计
    CostUSD     float64    `json:"cost_usd"`      // 总成本(美元)
    DurationMs  int64      `json:"duration_ms"`   // 执行耗时(毫秒)

    // 工具执行记录
    ToolCalls   []ToolCall `json:"tool_calls,omitempty"` // 工具调用历史

    // 代理协作记录
    AgentExecutions []AgentExecution `json:"agent_executions,omitempty"` // 代理执行记录

    // 时间戳
    StartedAt   time.Time `json:"started_at"`   // 开始时间
    CompletedAt time.Time `json:"completed_at"` // 完成时间

    // 错误信息
    Error       *TaskError `json:"error,omitempty"` // 错误详情

    // 扩展元数据
    Metadata    map[string]interface{} `json:"metadata,omitempty"`
}
```

**TaskResult的设计哲学**：结果不仅仅是答案，更是完整的执行记录和质量保证。

- **可观测性**：完整的执行追踪和性能指标
- **可审计性**：工具调用和代理执行的历史记录
- **可复现性**：包含所有输入参数和执行上下文
- **可扩展性**：通过Metadata支持未来扩展

**结构设计的权衡分析**：

1. **字段分组策略**：
   ```go
   // 为什么这样分组？
   // 1. 核心字段：Query, UserID, TenantID - 任务执行必需
   // 2. 上下文字段：History, SessionCtx - 连续性支持
   // 3. 控制字段：Mode, Template* - 执行策略控制
   // 4. 质量字段：RequireApproval, MinConfidence - 结果保证
   // 5. 资源字段：MaxTokens, BudgetLimitUSD - 成本控制
   // 6. 元数据字段：RequestID, UserAgent - 可观测性
   ```

2. **可选字段的设计哲学**：
   ```go
   // 使用omitempty标签的考虑：
   // 1. 向后兼容：新字段不会破坏旧客户端
   // 2. 存储效率：空值不占用存储空间
   // 3. API简洁：可选参数不污染接口
   // 4. 演进安全：字段可逐步添加而不影响现有代码
   ```

3. **类型选择的设计决策**：
   ```go
   // 字符串vs枚举的权衡：
   Mode string // 为什么不用枚举？
   // 1. 扩展性：新模式无需修改类型定义
   // 2. API友好：字符串在JSON/HTTP中更自然
   // 3. 向后兼容：未知模式不会导致解析错误
   // 4. 配置灵活：运行时可配置新模式
   ```

#### TaskResult的完整定义

**这块代码展示了什么？**

这段代码定义了TaskResult结构体，包含任务执行的完整结果信息。背景是：AI任务的输出不仅是文本结果，还包括执行状态、置信度、引用来源等丰富信息，这些信息对于评估结果质量和可追溯性至关重要。

这段代码的目的是说明如何结构化地表达AI任务的执行结果和质量指标。

```go
// TaskResult 定义任务执行的完整结果
type TaskResult struct {
    // 执行标识
    TaskID      string `json:"task_id"`      // 任务唯一标识
    WorkflowID  string `json:"workflow_id"`  // 工作流实例ID
    Status      string `json:"status"`       // 执行状态: completed/failed/timeout

    // 核心结果
    Result      string `json:"result"`       // 主要结果文本
    ResultType  string `json:"result_type"`  // 结果类型: text/json/html

    // 结构化输出
    StructuredOutput map[string]interface{} `json:"structured_output,omitempty"` // 结构化数据

    // 引用和证据
    Citations   []Citation `json:"citations,omitempty"` // 引用来源
    Confidence  float64    `json:"confidence,omitempty"` // 结果置信度

    // 执行统计
    TokenUsage  TokenUsage `json:"token_usage"`   // 令牌使用统计
    CostUSD     float64    `json:"cost_usd"`      // 总成本(美元)
    DurationMs  int64      `json:"duration_ms"`   // 执行耗时(毫秒)

    // 工具执行记录
    ToolCalls   []ToolCall `json:"tool_calls,omitempty"` // 工具调用历史

    // 代理协作记录
    AgentExecutions []AgentExecution `json:"agent_executions,omitempty"` // 代理执行记录

    // 时间戳
    StartedAt   time.Time `json:"started_at"`   // 开始时间
    CompletedAt time.Time `json:"completed_at"` // 完成时间

    // 错误信息
    Error       *TaskError `json:"error,omitempty"` // 错误详情

    // 元数据
    Metadata    map[string]interface{} `json:"metadata,omitempty"` // 扩展元数据
}
```

**结果设计的完整性考虑**：
- **可观测性**：完整的执行追踪和性能指标
- **可审计性**：工具调用和代理执行的历史记录
- **可复现性**：包含所有输入参数和执行上下文
- **可扩展性**：通过Metadata支持未来扩展

### 任务路由系统的深度实现

任务创建后进入智能路由系统，让我们分析其决策逻辑的实现。

#### OrchestratorRouter的核心架构

```go
// go/orchestrator/internal/workflows/orchestrator_router.go

// OrchestratorRouter 任务路由器 - 智能决策执行策略
type OrchestratorRouter struct {
    // 复杂度分析器
    complexityAnalyzer *ComplexityAnalyzer

    // 执行策略映射
    strategies map[string]Strategy

    // 配置管理
    config *RouterConfig

    // 指标收集
    metrics *RouterMetrics

    // 日志记录
    logger *zap.Logger
}

// RouterConfig 路由配置
type RouterConfig struct {
    // 复杂度阈值
    SimpleTaskThreshold    float64 `yaml:"simple_task_threshold"`    // 简单任务阈值
    ComplexTaskThreshold   float64 `yaml:"complex_task_threshold"`   // 复杂任务阈值

    // 默认策略
    DefaultSimpleStrategy  string  `yaml:"default_simple_strategy"`  // 默认简单策略
    DefaultComplexStrategy string  `yaml:"default_complex_strategy"` // 默认复杂策略

    // 模板路由
    TemplateRoutingEnabled bool    `yaml:"template_routing_enabled"` // 启用模板路由
    TemplateFallback       bool    `yaml:"template_fallback"`        // 模板降级

    // 性能控制
    RoutingTimeout         time.Duration `yaml:"routing_timeout"`    // 路由超时
    MaxConcurrency         int           `yaml:"max_concurrency"`     // 最大并发
}

// RouteTask 执行任务路由的核心逻辑
func (r *OrchestratorRouter) RouteTask(ctx context.Context, input TaskInput) (TaskResult, error) {
    // 1. 记录路由开始
    startTime := time.Now()
    routeID := r.generateRouteID()

    r.logger.Info("Starting task routing",
        zap.String("route_id", routeID),
        zap.String("task_id", input.RequestID),
        zap.String("user_id", input.UserID))

    // 2. 预路由验证
    if err := r.validateInput(input); err != nil {
        r.metrics.RecordRouteError("validation_error")
        return TaskResult{}, fmt.Errorf("input validation failed: %w", err)
    }

    // 3. 上下文增强
    enrichedInput, err := r.enrichContext(ctx, input)
    if err != nil {
        r.metrics.RecordRouteError("context_enrichment_error")
        return TaskResult{}, fmt.Errorf("context enrichment failed: %w", err)
    }

    // 4. 复杂度分析 - 核心决策逻辑
    complexity, err := r.complexityAnalyzer.AnalyzeComplexity(ctx, enrichedInput)
    if err != nil {
        r.metrics.RecordRouteError("complexity_analysis_error")
        // 复杂度分析失败，使用保守策略
        return r.executeConservativeStrategy(ctx, enrichedInput)
    }

    // 5. 策略选择和执行
    result, err := r.selectAndExecuteStrategy(ctx, enrichedInput, complexity)
    if err != nil {
        r.metrics.RecordRouteError("strategy_execution_error")
        return TaskResult{}, fmt.Errorf("strategy execution failed: %w", err)
    }

    // 6. 后处理和记录
    result = r.postProcessResult(result, routeID, startTime)

    r.metrics.RecordRouteSuccess(time.Since(startTime))
    return result, nil
}
```

**路由决策算法的实现**：

```go
func (r *OrchestratorRouter) selectAndExecuteStrategy(
    ctx context.Context,
    input TaskInput,
    complexity *ComplexityScore,
) (TaskResult, error) {

    var strategyName string

    // 决策树逻辑
    switch {
    case input.DisableAI:
        // 强制使用模板模式
        strategyName = "template"

    case input.TemplateName != "":
        // 用户指定模板
        strategyName = "template"

    case complexity.Score < r.config.SimpleTaskThreshold:
        // 简单任务 - 直接执行
        strategyName = r.selectSimpleStrategy(input, complexity)

    case complexity.Score < r.config.ComplexTaskThreshold:
        // 中等复杂度 - 分解执行
        strategyName = r.selectComplexStrategy(input, complexity)

    case complexity.Subtasks != nil && len(complexity.Subtasks) > 1:
        // 高复杂度 - 多代理协作
        strategyName = r.selectCollaborativeStrategy(input, complexity)

    default:
        // 研究型复杂任务
        strategyName = "research"
    }

    // 获取并执行策略
    strategy, exists := r.strategies[strategyName]
    if !exists {
        return TaskResult{}, fmt.Errorf("unknown strategy: %s", strategyName)
    }

    r.logger.Info("Selected execution strategy",
        zap.String("strategy", strategyName),
        zap.Float64("complexity_score", complexity.Score))

    return strategy.Execute(ctx, input)
}
```

#### 复杂度分析器的实现

**这块代码展示了什么？**

这段代码展示了复杂度分析器的实现，用于评估AI任务的复杂度并决定执行策略。背景是：不同复杂度的任务需要不同的处理策略，复杂度分析器通过多维度指标（如查询长度、上下文丰富度、质量要求）来量化任务复杂度。

这段代码的目的是说明如何通过算法分析将任务分类并选择合适的执行路径。

```go
// go/orchestrator/internal/workflows/complexity_analyzer.go

// ComplexityAnalyzer 复杂度分析器
type ComplexityAnalyzer struct {
    // 复杂度指标
    indicators []ComplexityIndicator

    // 机器学习模型（可选）
    mlModel MLComplexityPredictor

    // 规则引擎
    ruleEngine *RuleEngine

    // 缓存
    cache *ComplexityCache
}

// ComplexityScore 复杂度评分
type ComplexityScore struct {
    Score     float64       // 总体复杂度分数 (0-1)
    Breakdown map[string]float64 // 各维度分解

    // 分解信息
    Subtasks   []Subtask     // 子任务列表
    Reasoning  string        // 分析推理过程
    Confidence float64       // 分析置信度

    // 元数据
    AnalyzedAt time.Time     // 分析时间
    Version    string        // 分析器版本
}

// AnalyzeComplexity 核心分析逻辑
func (ca *ComplexityAnalyzer) AnalyzeComplexity(ctx context.Context, input TaskInput) (*ComplexityScore, error) {
    // 1. 缓存检查
    if cached := ca.cache.Get(input.Query); cached != nil {
        ca.metrics.RecordCacheHit()
        return cached, nil
    }

    // 2. 多维度分析
    indicators := ca.computeIndicators(input)

    // 3. 机器学习预测（如果启用）
    mlScore := float64(0)
    if ca.mlModel != nil {
        mlScore = ca.mlModel.PredictComplexity(input)
    }

    // 4. 规则引擎评估
    ruleScore := ca.ruleEngine.EvaluateRules(input, indicators)

    // 5. 综合评分
    finalScore := ca.combineScores(indicators, mlScore, ruleScore)

    // 6. 任务分解（如果需要）
    subtasks := ca.decomposeIfNeeded(input, finalScore)

    score := &ComplexityScore{
        Score:     finalScore,
        Breakdown: indicators,
        Subtasks:  subtasks,
        Reasoning: ca.generateReasoning(indicators, finalScore),
        Confidence: ca.computeConfidence(indicators),
        AnalyzedAt: time.Now(),
        Version:    ca.getVersion(),
    }

    // 7. 缓存结果
    ca.cache.Put(input.Query, score)

    return score, nil
}

// computeIndicators 计算复杂度指标
func (ca *ComplexityAnalyzer) computeIndicators(input TaskInput) map[string]float64 {
    indicators := make(map[string]float64)

    // 查询长度指标
    indicators["query_length"] = math.Min(float64(len(input.Query))/1000.0, 1.0)

    // 历史复杂度指标
    if len(input.History) > 5 {
        indicators["conversation_depth"] = 1.0
    } else {
        indicators["conversation_depth"] = float64(len(input.History)) / 5.0
    }

    // 上下文丰富度
    contextSize := len(fmt.Sprintf("%v", input.Context))
    indicators["context_richness"] = math.Min(float64(contextSize)/5000.0, 1.0)

    // 时间压力指标
    if input.TimeoutSeconds > 0 && input.TimeoutSeconds < 300 {
        indicators["time_pressure"] = 1.0 - float64(input.TimeoutSeconds)/300.0
    }

    // 质量要求指标
    qualityScore := 0.0
    if input.RequireCitations {
        qualityScore += 0.3
    }
    if input.MinConfidence > 0.8 {
        qualityScore += 0.3
    }
    if input.RequireApproval {
        qualityScore += 0.4
    }
    indicators["quality_requirements"] = qualityScore

    // 工具需求指标
    if len(input.SuggestedTools) > 3 {
        indicators["tool_complexity"] = 1.0
    } else {
        indicators["tool_complexity"] = float64(len(input.SuggestedTools)) / 3.0
    }

    return indicators
}
```

**复杂度分析的设计哲学**：

1. **多维度评估**：
   - 查询长度：长查询通常更复杂
   - 会话深度：多轮对话需要更多上下文
   - 上下文丰富度：更多上下文信息增加复杂度
   - 时间压力：紧急任务需要更简单策略
   - 质量要求：高要求任务需要更谨慎处理
   - 工具复杂度：多工具调用增加复杂度

2. **自适应学习**：
   - 缓存历史分析结果
   - 机器学习模型持续改进
   - 规则引擎动态调整

3. **可观测性**：
   - 详细的推理过程记录
   - 置信度评分
   - 性能指标收集

### 任务生命周期的完整追踪

从用户输入到最终结果的完整执行流程：

```go
// 任务生命周期追踪器
type TaskLifecycleTracker struct {
    taskID     string
    startTime  time.Time
    events     []LifecycleEvent
    mu         sync.RWMutex
}

type LifecycleEvent struct {
    Stage     string                 // 阶段：created/routed/executing/completed/failed
    Timestamp time.Time              // 时间戳
    Metadata  map[string]interface{} // 元数据
}

// 完整生命周期
func (t *TaskLifecycleTracker) TrackFullLifecycle(input TaskInput) (TaskResult, error) {
    // 阶段1: 任务创建
    t.recordEvent("created", map[string]interface{}{
        "query_length": len(input.Query),
        "has_history": len(input.History) > 0,
    })

    // 阶段2: 输入验证
    if err := validateTaskInput(input); err != nil {
        t.recordEvent("validation_failed", map[string]interface{}{
            "error": err.Error(),
        })
        return TaskResult{}, err
    }

    // 阶段3: 上下文增强
    enrichedInput, err := enrichTaskContext(input)
    if err != nil {
        t.recordEvent("enrichment_failed", map[string]interface{}{
            "error": err.Error(),
        })
        return TaskResult{}, err
    }
    t.recordEvent("enriched", map[string]interface{}{
        "context_size": len(fmt.Sprintf("%v", enrichedInput.Context)),
    })

    // 阶段4: 复杂度分析
    complexity, err := analyzeTaskComplexity(enrichedInput)
    if err != nil {
        t.recordEvent("complexity_analysis_failed", map[string]interface{}{
            "error": err.Error(),
        })
        // 降级处理
        complexity = &ComplexityScore{Score: 0.5}
    }
    t.recordEvent("complexity_analyzed", map[string]interface{}{
        "score": complexity.Score,
        "subtasks_count": len(complexity.Subtasks),
    })

    // 阶段5: 策略路由
    strategy := selectExecutionStrategy(enrichedInput, complexity)
    t.recordEvent("strategy_selected", map[string]interface{}{
        "strategy_name": strategy.Name(),
    })

    // 阶段6: 执行
    t.recordEvent("execution_started", map[string]interface{}{
        "strategy": strategy.Name(),
    })

    result, err := strategy.Execute(context.Background(), enrichedInput)
    if err != nil {
        t.recordEvent("execution_failed", map[string]interface{}{
            "error": err.Error(),
            "duration_ms": time.Since(t.startTime).Milliseconds(),
        })
        return TaskResult{}, err
    }

    // 阶段7: 后处理
    finalResult := postProcessTaskResult(result, t.events)
    t.recordEvent("completed", map[string]interface{}{
        "duration_ms": time.Since(t.startTime).Milliseconds(),
        "result_size": len(finalResult.Result),
        "tokens_used": finalResult.TokenUsage.TotalTokens,
        "cost_usd": finalResult.CostUSD,
    })

    return finalResult, nil
}
```

**生命周期设计的价值**：
1. **可观测性**：完整的事件追踪链
2. **性能分析**：各阶段耗时统计
3. **故障诊断**：失败点精确定位
4. **优化机会**：识别性能瓶颈

## 会话（Session）：记忆与连续性

### 会话系统架构的深度设计

会话是Shannon实现连续对话的核心，让我们深入分析其多层架构和内存管理机制。

#### Session数据结构的完整定义

**这块代码展示了什么？**

这段代码定义了Session结构体，包含用户会话的完整状态信息。背景是：AI对话需要保持上下文连续性，会话管理不仅要存储对话历史，还要管理代理状态、资源统计、安全控制等多个维度的数据。

这段代码的目的是说明如何通过结构化设计实现复杂的会话状态管理。

```go
// go/orchestrator/internal/session/types.go

// Session 表示用户会话的完整状态
type Session struct {
    // 身份标识 - 会话唯一性
    ID        string    `json:"id" redis:"id"`        // UUID格式的会话ID
    UserID    string    `json:"user_id" redis:"user_id"` // 用户标识
    TenantID  string    `json:"tenant_id" redis:"tenant_id"` // 租户标识

    // 时间管理 - 会话生命周期
    CreatedAt time.Time `json:"created_at" redis:"created_at"` // 创建时间
    UpdatedAt time.Time `json:"updated_at" redis:"updated_at"` // 最后更新时间
    ExpiresAt time.Time `json:"expires_at" redis:"expires_at"` // 过期时间
    LastActivity time.Time `json:"last_activity" redis:"last_activity"` // 最后活动时间

    // 核心数据 - 会话状态
    Metadata  map[string]interface{} `json:"metadata" redis:"metadata"` // 元数据存储
    Context   map[string]interface{} `json:"context" redis:"context"`   // 持久化上下文
    History   []Message `json:"history" redis:"history"`               // 消息历史

    // 代理状态 - 多代理协作
    AgentStates map[string]*AgentState `json:"agent_states" redis:"agent_states"`

    // 资源统计 - 成本控制
    TotalTokensUsed int     `json:"total_tokens_used" redis:"total_tokens_used"` // 总令牌使用
    TotalCostUSD    float64 `json:"total_cost_usd" redis:"total_cost_usd"`       // 总成本(美元)

    // 性能指标 - 可观测性
    RequestCount    int     `json:"request_count" redis:"request_count"`       // 请求次数
    AverageLatency  float64 `json:"average_latency" redis:"average_latency"`   // 平均延迟(ms)

    // 安全控制 - 访问限制
    IPWhitelist     []string `json:"ip_whitelist" redis:"ip_whitelist"`     // IP白名单
    RateLimit       int      `json:"rate_limit" redis:"rate_limit"`         // 速率限制(RPM)

    // 版本控制 - 数据一致性
    Version         int64    `json:"version" redis:"version"`               // 乐观锁版本号
    SchemaVersion   string   `json:"schema_version" redis:"schema_version"` // 架构版本
}

// AgentState 代理在会话中的状态
type AgentState struct {
    AgentID       string                 `json:"agent_id"`       // 代理标识
    AgentName     string                 `json:"agent_name"`     // 代理显示名
    LastActive    time.Time              `json:"last_active"`    // 最后活跃时间
    State         string                 `json:"state"`          // 当前状态: active/idle/error
    Memory        map[string]interface{} `json:"memory"`         // 代理记忆
    ToolsUsed     []string               `json:"tools_used"`     // 已使用的工具
    TokensUsed    int                    `json:"tokens_used"`    // 令牌使用量
    SuccessRate   float64                `json:"success_rate"`   // 成功率
    AverageLatency float64                `json:"average_latency"` // 平均延迟(ms)

    // 执行统计
    ExecutionCount int `json:"execution_count"` // 执行次数
    ErrorCount     int `json:"error_count"`     // 错误次数
    RetryCount     int `json:"retry_count"`     // 重试次数
}
```

**数据结构设计的权衡分析**：

1. **字段分组策略**：
   ```go
   // 为什么这样分组？
   // 1. 标识字段：ID, UserID, TenantID - 核心定位信息
   // 2. 时间字段：*At时间戳 - 生命周期管理
   // 3. 状态字段：Context, History - 会话核心数据
   // 4. 代理字段：AgentStates - 多代理支持
   // 5. 统计字段：TokensUsed, Cost - 资源监控
   // 6. 控制字段：RateLimit, IPWhitelist - 安全控制
   ```

2. **序列化标签的选择**：
   ```go
   // json标签 vs redis标签的区别：
   // json: 用于API响应和JSON序列化
   // redis: 用于Redis存储的字段映射
   // 允许不同的序列化策略
   CreatedAt time.Time `json:"created_at" redis:"created_at"`
   ```

3. **版本控制的设计**：
   ```go
   // 双版本控制的原因：
   // Version: 乐观锁，防止并发更新冲突
   // SchemaVersion: 架构版本，支持平滑升级
   Version int64 `redis:"version"`
   SchemaVersion string `redis:"schema_version"`
   ```

#### 会话管理器的完整实现

**这块代码展示了什么？**

这段代码展示了会话管理器的完整实现，采用多层缓存架构。背景是：会话数据需要高性能访问，同时要保证数据一致性和故障恢复，多层缓存（本地+Redis）提供了性能和可靠性的平衡。

这段代码的目的是说明如何实现高性能的分布式会话管理。

```go
// go/orchestrator/internal/session/manager.go

// Manager 会话管理器 - 多层缓存架构
type Manager struct {
    // Redis存储层
    redisClient *circuitbreaker.RedisWrapper

    // 本地缓存层 - LRU策略
    localCache  *lru.Cache[string, *Session]  // 线程安全LRU缓存
    cacheSize   int                           // 缓存最大大小

    // 并发控制
    mu          sync.RWMutex                  // 保护本地缓存

    // 配置参数
    sessionTTL      time.Duration             // 会话生存时间
    maxHistorySize  int                       // 最大历史消息数
    cleanupInterval time.Duration             // 清理间隔

    // 监控指标
    metrics        *SessionMetrics
    logger         *zap.Logger

    // 后台任务
    cleanupTicker  *time.Ticker
    cleanupDone    chan struct{}
}

// NewManager 创建会话管理器
func NewManager(config *SessionConfig, logger *zap.Logger) (*Manager, error) {
    // 1. 初始化Redis客户端
    redisClient, err := circuitbreaker.NewRedisWrapper(config.RedisAddr, config.RedisPassword)
    if err != nil {
        return nil, fmt.Errorf("failed to create Redis client: %w", err)
    }

    // 2. 初始化LRU缓存
    localCache, err := lru.New[string, *Session](config.CacheSize)
    if err != nil {
        return nil, fmt.Errorf("failed to create LRU cache: %w", err)
    }

    manager := &Manager{
        redisClient:     redisClient,
        localCache:      localCache,
        cacheSize:       config.CacheSize,
        sessionTTL:      config.SessionTTL,
        maxHistorySize:  config.MaxHistorySize,
        cleanupInterval: config.CleanupInterval,
        metrics:         NewSessionMetrics(),
        logger:          logger,
        cleanupDone:     make(chan struct{}),
    }

    // 3. 启动后台清理任务
    go manager.cleanupWorker()

    return manager, nil
}
```

**多层缓存架构的核心机制**：

```go
// Get 获取会话 - 缓存优先策略
func (m *Manager) Get(ctx context.Context, sessionID string) (*Session, error) {
    startTime := time.Now()

    // 1. 本地缓存查找
    if session, ok := m.localCache.Get(sessionID); ok {
        m.metrics.RecordCacheHit("local")
        m.metrics.RecordOperationDuration("get", time.Since(startTime))
        return session, nil
    }

    // 2. Redis查找
    session, err := m.getFromRedis(ctx, sessionID)
    if err != nil {
        m.metrics.RecordOperationError("redis_get")
        return nil, fmt.Errorf("failed to get session from Redis: %w", err)
    }

    if session != nil {
        // 3. 写入本地缓存
        m.localCache.Put(sessionID, session)
        m.metrics.RecordCacheMiss("redis")
    } else {
        m.metrics.RecordCacheMiss("not_found")
    }

    m.metrics.RecordOperationDuration("get", time.Since(startTime))
    return session, nil
}

// getFromRedis 从Redis获取会话
func (m *Manager) getFromRedis(ctx context.Context, sessionID string) (*Session, error) {
    key := m.sessionKey(sessionID)

    // 使用HGETALL获取所有字段
    values, err := m.redisClient.HGetAll(ctx, key).Result()
    if err != nil {
        return nil, err
    }

    // Redis中不存在
    if len(values) == 0 {
        return nil, nil
    }

    // 反序列化
    session := &Session{}
    if err := m.deserializeSession(values, session); err != nil {
        return nil, fmt.Errorf("failed to deserialize session: %w", err)
    }

    // 检查过期
    if time.Now().After(session.ExpiresAt) {
        // 异步清理过期会话
        go m.deleteExpiredSession(ctx, sessionID)
        return nil, nil
    }

    return session, nil
}

// Save 保存会话 - 写穿策略
func (m *Manager) Save(ctx context.Context, session *Session) error {
    startTime := time.Now()

    // 1. 更新时间戳
    now := time.Now()
    session.UpdatedAt = now
    session.LastActivity = now
    session.Version++ // 乐观锁版本递增

    // 2. 清理历史消息（防止无限增长）
    if len(session.History) > m.maxHistorySize {
        // 保留最新的消息
        keepStart := len(session.History) - m.maxHistorySize
        session.History = session.History[keepStart:]
    }

    // 3. 序列化为Redis哈希
    sessionData := m.serializeSession(session)

    // 4. 使用MULTI事务保证原子性
    pipe := m.redisClient.TxPipeline()
    key := m.sessionKey(session.ID)

    // 设置所有字段
    pipe.HMSet(ctx, key, sessionData)

    // 设置过期时间
    pipe.Expire(ctx, key, m.sessionTTL)

    // 执行事务
    _, err := pipe.Exec(ctx)
    if err != nil {
        m.metrics.RecordOperationError("redis_save")
        return fmt.Errorf("failed to save session to Redis: %w", err)
    }

    // 5. 更新本地缓存
    m.localCache.Put(session.ID, session)

    // 6. 清理其他用户的缓存（内存优化）
    m.evictOtherUsersSessions(session.UserID, session.ID)

    m.metrics.RecordOperationDuration("save", time.Since(startTime))
    return nil
}
```

**缓存策略的设计哲学**：

1. **读策略 - Cache Aside**：
   ```go
   // 为什么选择Cache Aside？
   // 1. 数据一致性：缓存不直接写，减少不一致风险
   // 2. 实现简单：读miss时从DB加载，写时更新DB和缓存
   // 3. 容错性：缓存失效不影响功能
   // 4. 性能平衡：读多写少场景的最优选择
   ```

2. **写策略 - Write Through**：
   ```go
   // 为什么选择Write Through？
   // 1. 强一致性：写操作同时更新缓存和存储
   // 2. 读取性能：读操作总能从缓存获取最新数据
   // 3. 简单性：无需复杂的缓存失效逻辑
   // 4. 适用场景：会话数据更新频繁，需要强一致性
   ```

3. **LRU淘汰算法的实现**：
   ```go
   // LRU缓存淘汰策略
   // 1. 访问时间跟踪：每次访问更新时间戳
   // 2. 容量限制：超过容量时淘汰最久未访问
   // 3. 并发安全：使用RWMutex保护
   // 4. 内存效率：避免内存无限增长
   ```

#### 后台清理和维护机制

```go
// cleanupWorker 后台清理worker
func (m *Manager) cleanupWorker() {
    ticker := time.NewTicker(m.cleanupInterval)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            m.performCleanup()
        case <-m.cleanupDone:
            return
        }
    }
}

// performCleanup 执行清理任务
func (m *Manager) performCleanup() {
    ctx := context.Background()

    // 1. 清理本地缓存中的过期会话
    m.cleanupExpiredLocalCache()

    // 2. 清理Redis中的过期会话（使用Lua脚本）
    m.cleanupExpiredRedisSessions(ctx)

    // 3. 统计和报告
    m.reportCleanupStats()
}

// cleanupExpiredLocalCache 清理本地过期缓存
func (m *Manager) cleanupExpiredLocalCache() {
    m.mu.Lock()
    defer m.mu.Unlock()

    now := time.Now()
    evicted := 0

    // 遍历所有缓存项
    m.localCache.Each(func(key string, session *Session) bool {
        if now.After(session.ExpiresAt) {
            m.localCache.Remove(key)
            evicted++
        }
        return true // 继续遍历
    })

    if evicted > 0 {
        m.logger.Info("Cleaned up expired local cache entries",
            zap.Int("evicted_count", evicted))
    }
}

// cleanupExpiredRedisSessions 清理Redis过期会话
func (m *Manager) cleanupExpiredRedisSessions(ctx context.Context) {
    // 使用Lua脚本批量清理过期会话
    luaScript := `
        local keys = redis.call('SCAN', 0, 'MATCH', ARGV[1], 'COUNT', ARGV[2])
        local cleaned = 0
        for i, key in ipairs(keys[2]) do
            local expires_at = redis.call('HGET', key, 'expires_at')
            if expires_at and tonumber(expires_at) < tonumber(ARGV[3]) then
                redis.call('DEL', key)
                cleaned = cleaned + 1
            end
        end
        return cleaned
    `

    pattern := "session:*"
    batchSize := "100"
    now := strconv.FormatInt(time.Now().Unix(), 10)

    result, err := m.redisClient.Eval(ctx, luaScript, []string{}, pattern, batchSize, now).Result()
    if err != nil {
        m.logger.Error("Failed to cleanup expired Redis sessions", zap.Error(err))
        return
    }

    cleanedCount := result.(int64)
    if cleanedCount > 0 {
        m.logger.Info("Cleaned up expired Redis sessions",
            zap.Int64("cleaned_count", cleanedCount))
    }
}
```

**清理策略的设计考虑**：

1. **分层清理**：
   - 本地缓存：内存敏感，及时清理
   - Redis存储：批量清理，减少压力
   - 异步执行：不阻塞正常操作

2. **性能优化**：
   - Lua脚本：原子性批量操作
   - 分页处理：避免长时间阻塞
   - 统计报告：监控清理效果

3. **容错设计**：
   - 错误记录：不因清理失败影响服务
   - 重试机制：失败时可重试清理

#### 会话安全和隔离机制

```go
// ValidateAccess 验证会话访问权限
func (m *Manager) ValidateAccess(ctx context.Context, sessionID string, userID string, clientIP string) error {
    session, err := m.Get(ctx, sessionID)
    if err != nil {
        return fmt.Errorf("failed to get session: %w", err)
    }
    if session == nil {
        return errors.New("session not found")
    }

    // 1. 用户权限验证
    if session.UserID != userID {
        m.metrics.RecordSecurityViolation("user_mismatch")
        return errors.New("access denied: user mismatch")
    }

    // 2. 租户隔离验证
    if session.TenantID != m.getCurrentTenant(ctx) {
        m.metrics.RecordSecurityViolation("tenant_mismatch")
        return errors.New("access denied: tenant isolation")
    }

    // 3. IP白名单验证
    if len(session.IPWhitelist) > 0 {
        if !contains(session.IPWhitelist, clientIP) {
            m.metrics.RecordSecurityViolation("ip_blocked")
            return errors.New("access denied: IP not whitelisted")
        }
    }

    // 4. 速率限制检查
    if session.RateLimit > 0 {
        if !m.checkRateLimit(sessionID, session.RateLimit) {
            m.metrics.RecordSecurityViolation("rate_limited")
            return errors.New("rate limit exceeded")
        }
    }

    return nil
}

// checkRateLimit 速率限制实现
func (m *Manager) checkRateLimit(sessionID string, limit int) bool {
    key := fmt.Sprintf("ratelimit:session:%s", sessionID)

    // 使用Redis原子操作
    count, err := m.redisClient.Incr(ctx, key).Result()
    if err != nil {
        m.logger.Error("Rate limit check failed", zap.Error(err))
        return true // 出错时允许通过
    }

    // 第一次访问时设置过期时间
    if count == 1 {
        m.redisClient.Expire(ctx, key, time.Minute)
    }

    return int(count) <= limit
}
```

**安全设计的多层防护**：
1. **身份验证**：用户和租户匹配检查
2. **访问控制**：IP白名单和速率限制
3. **审计追踪**：所有安全事件记录
4. **隔离保证**：租户间数据完全隔离

### 会话与任务的关系

任务总是发生在会话的上下文中：

```go
// 任务执行时包含会话信息
taskInput := TaskInput{
    Query:     "分析苹果财报",
    UserID:    session.UserID,
    SessionID: session.ID,
    History:   session.GetRecentHistory(10), // 最近10条消息
    SessionCtx: session.Context,              // 持久化上下文
}

// 执行完成后更新会话
session.AddMessage(ctx, Message{
    Role:      "user",
    Content:   taskInput.Query,
    Timestamp: time.Now(),
})

session.AddMessage(ctx, Message{
    Role:      "assistant",
    Content:   result.Result,
    Timestamp: time.Now(),
})
```

## 代理（Agent）：智能执行单元

### 代理命名系统的深度设计

代理命名系统是Shannon多代理架构的核心，让我们深入分析其确定性算法和设计哲学。

#### 确定性命名算法的实现

```go
// go/orchestrator/internal/agents/names.go

// 火车站名字数据集 - 提供可读性强的代理标识
var stationNames = []string{
    // 东京核心区域
    "Tokyo", "Shinjuku", "Shibuya", "Ikebukuro", "Ueno",
    // 横滨线
    "Yokohama", "Kawasaki", "Shinagawa", "Osaki", "Oimachi",
    // 山手线
    "Akihabara", "Asakusa", "Ebisu", "Harajuku", "Meiji",
    // 其他主要站点
    "Narita", "Haneda", "Otaru", "Sapporo", "Nagoya",
    "Kyoto", "Osaka", "Kobe", "Hiroshima", "Fukuoka",
    // 扩展集 - 确保足够的选择空间
    "Sendai", "Niigata", "Kanazawa", "Nagano", "Shizuoka",
    "Gifu", "Toyama", "Fukui", "Yamanashi", "Nagano",
}

// NameGenerator 代理名字生成器
type NameGenerator struct {
    names []string
    mu    sync.RWMutex
}

// NewNameGenerator 创建名字生成器
func NewNameGenerator() *NameGenerator {
    return &NameGenerator{
        names: stationNames,
        // 支持运行时扩展名字集
    }
}

// GetAgentName 生成确定性代理名字
func (ng *NameGenerator) GetAgentName(workflowID string, index int) string {
    ng.mu.RLock()
    defer ng.mu.RUnlock()

    if len(ng.names) == 0 {
        // 降级到数字命名
        return fmt.Sprintf("agent-%d", index)
    }

    // 1. 使用FNV-1a哈希算法计算workflowID的哈希值
    hash := ng.fnv32a(workflowID)

    // 2. 结合index生成最终索引
    // 确保相同的(workflowID, index)总是生成相同的名字
    finalIndex := (int(hash) + index) % len(ng.names)

    return ng.names[finalIndex]
}

// fnv32a FNV-1a 32位哈希算法实现
func (ng *NameGenerator) fnv32a(input string) uint32 {
    const (
        offset32 uint32 = 2166136261
        prime32  uint32 = 16777619
    )

    hash := offset32
    for _, char := range input {
        hash ^= uint32(char)
        hash *= prime32
    }

    return hash
}

// AddCustomNames 运行时添加自定义名字
func (ng *NameGenerator) AddCustomNames(names []string) {
    ng.mu.Lock()
    defer ng.mu.Unlock()

    // 去重并添加
    nameSet := make(map[string]bool)
    for _, name := range ng.names {
        nameSet[name] = true
    }

    for _, name := range names {
        if !nameSet[name] {
            ng.names = append(ng.names, name)
            nameSet[name] = true
        }
    }
}
```

**命名算法的设计哲学**：

1. **确定性保证**：
   ```go
   // FNV-1a哈希算法特性：
   // 1. 确定性：相同输入总是相同输出
   // 2. 雪崩效应：输入微小变化导致输出大变化
   // 3. 均匀分布：哈希值在32位空间均匀分布
   // 4. 高性能：单次遍历，O(n)时间复杂度
   ```

2. **碰撞处理策略**：
   ```go
   // 取模运算处理哈希冲突：
   finalIndex := (int(hash) + index) % len(ng.names)
   // + index确保不同index即使哈希相同也能得到不同名字
   ```

3. **可扩展性设计**：
   ```go
   // 运行时添加名字的能力
   // 支持不同租户或场景的定制化命名
   func (ng *NameGenerator) AddCustomNames(names []string)
   ```

#### 代理工厂模式的实现

**这块代码展示了什么？**

这段代码展示了代理工厂模式的实现，负责创建和管理代理实例。背景是：AI代理具有复杂的初始化逻辑（工具集、记忆系统、提示编译等），工厂模式将这些复杂性封装起来，提供统一的代理创建接口。

这段代码的目的是说明如何通过工厂模式管理AI代理的生命周期和配置。

```go
// go/orchestrator/internal/agents/factory.go

// AgentFactory 代理工厂 - 创建和管理代理实例
type AgentFactory struct {
    // 代理配置映射
    agentConfigs map[string]*AgentConfig

    // 代理模板缓存
    agentTemplates map[string]*AgentTemplate

    // 依赖注入
    llmClient     *llm.Client
    toolRegistry  *tools.Registry
    memoryManager *memory.Manager

    // 监控指标
    metrics *AgentMetrics

    // 并发控制
    mu sync.RWMutex
}

// AgentConfig 代理配置
type AgentConfig struct {
    ID          string                 `yaml:"id"`          // 代理唯一标识
    Name        string                 `yaml:"name"`        // 显示名称
    Description string                 `yaml:"description"` // 功能描述

    // 能力配置
    Capabilities []string `yaml:"capabilities"` // 能力标签

    // LLM配置
    Model       string  `yaml:"model"`       // 默认模型
    Temperature float64 `yaml:"temperature"` // 温度参数
    MaxTokens   int     `yaml:"max_tokens"`  // 最大令牌数

    // 工具配置
    AllowedTools    []string `yaml:"allowed_tools"`    // 允许的工具
    RequiredTools   []string `yaml:"required_tools"`   // 必需的工具
    ToolPreferences map[string]int `yaml:"tool_preferences"` // 工具偏好权重

    // 记忆配置
    MemoryConfig *MemoryConfig `yaml:"memory_config"`

    // 行为配置
    Personality   string                 `yaml:"personality"`   // 个性设定
    SystemPrompt  string                 `yaml:"system_prompt"` // 系统提示
    ResponseStyle string                 `yaml:"response_style"` // 响应风格

    // 限制配置
    MaxExecutionsPerHour int `yaml:"max_executions_per_hour"` // 每小时最大执行次数
    TimeoutSeconds       int `yaml:"timeout_seconds"`         // 执行超时时间
}

// CreateAgent 创建代理实例
func (af *AgentFactory) CreateAgent(ctx context.Context, config *AgentConfig, sessionID string) (*Agent, error) {
    // 1. 验证配置
    if err := af.validateConfig(config); err != nil {
        return nil, fmt.Errorf("invalid agent config: %w", err)
    }

    // 2. 创建代理实例
    agent := &Agent{
        ID:          config.ID,
        Name:        config.Name,
        Description: config.Description,
        SessionID:   sessionID,
        CreatedAt:   time.Now(),
        Config:      config,
    }

    // 3. 初始化LLM客户端
    llmConfig := af.buildLLMConfig(config)
    agent.llmClient = af.llmClient.WithConfig(llmConfig)

    // 4. 配置工具集
    toolSet, err := af.buildToolSet(config)
    if err != nil {
        return nil, fmt.Errorf("failed to build tool set: %w", err)
    }
    agent.toolSet = toolSet

    // 5. 初始化记忆系统
    memory, err := af.memoryManager.CreateMemory(ctx, sessionID, config.ID, config.MemoryConfig)
    if err != nil {
        return nil, fmt.Errorf("failed to create memory: %w", err)
    }
    agent.memory = memory

    // 6. 编译系统提示
    systemPrompt, err := af.compileSystemPrompt(config)
    if err != nil {
        return nil, fmt.Errorf("failed to compile system prompt: %w", err)
    }
    agent.systemPrompt = systemPrompt

    // 7. 初始化指标收集
    agent.metrics = af.metrics.NewAgentMetrics(config.ID)

    af.logger.Info("Agent created successfully",
        zap.String("agent_id", config.ID),
        zap.String("session_id", sessionID))

    return agent, nil
}

// buildToolSet 构建代理的工具集
func (af *AgentFactory) buildToolSet(config *AgentConfig) (*ToolSet, error) {
    toolSet := &ToolSet{}

    // 1. 添加必需工具
    for _, toolName := range config.RequiredTools {
        tool, exists := af.toolRegistry.GetTool(toolName)
        if !exists {
            return nil, fmt.Errorf("required tool not found: %s", toolName)
        }
        toolSet.AddTool(tool, ToolPriorityRequired)
    }

    // 2. 添加允许工具（带权重）
    for _, toolName := range config.AllowedTools {
        if tool, exists := af.toolRegistry.GetTool(toolName); exists {
            priority := ToolPriorityNormal
            if weight, hasWeight := config.ToolPreferences[toolName]; hasWeight {
                priority = ToolPriority(weight)
            }
            toolSet.AddTool(tool, priority)
        }
    }

    // 3. 验证工具兼容性
    if err := toolSet.ValidateCompatibility(); err != nil {
        return nil, fmt.Errorf("tool compatibility validation failed: %w", err)
    }

    return toolSet, nil
}

// compileSystemPrompt 编译系统提示
func (af *AgentFactory) compileSystemPrompt(config *AgentConfig) (string, error) {
    // 1. 基础模板
    template := af.getBaseTemplate(config.Personality)

    // 2. 能力注入
    capabilities := strings.Join(config.Capabilities, ", ")
    template = strings.ReplaceAll(template, "{{capabilities}}", capabilities)

    // 3. 工具描述注入
    toolDescriptions := af.generateToolDescriptions(config)
    template = strings.ReplaceAll(template, "{{tools}}", toolDescriptions)

    // 4. 行为风格注入
    template = strings.ReplaceAll(template, "{{style}}", config.ResponseStyle)

    // 5. 自定义提示合并
    if config.SystemPrompt != "" {
        template = config.SystemPrompt + "\n\n" + template
    }

    return template, nil
}
```

**工厂模式的设计优势**：

1. **配置驱动**：
   - YAML配置完全定义代理行为
   - 支持运行时重新配置
   - 版本控制和回滚能力

2. **依赖注入**：
   - 清晰的依赖关系
   - 易于测试和模拟
   - 支持不同实现切换

3. **构建验证**：
   - 创建时验证配置正确性
   - 工具兼容性检查
   - 资源限制验证

#### 代理状态管理的实现

```go
// go/orchestrator/internal/session/types.go

// AgentState 代理在会话中的完整状态
type AgentState struct {
    // 基本信息
    AgentID       string    `json:"agent_id"`       // 代理唯一标识
    AgentName     string    `json:"agent_name"`     // 代理显示名称
    CreatedAt     time.Time `json:"created_at"`     // 创建时间
    LastActive    time.Time `json:"last_active"`    // 最后活跃时间
    State         string    `json:"state"`          // 当前状态

    // 执行统计
    ExecutionCount int         `json:"execution_count"` // 总执行次数
    SuccessCount   int         `json:"success_count"`   // 成功次数
    ErrorCount     int         `json:"error_count"`     // 错误次数
    RetryCount     int         `json:"retry_count"`     // 重试次数

    // 性能指标
    TotalTokensUsed    int     `json:"total_tokens_used"`    // 总令牌使用
    TotalCostUSD       float64 `json:"total_cost_usd"`       // 总成本
    AverageLatency     float64 `json:"average_latency"`     // 平均延迟(ms)
    SuccessRate        float64 `json:"success_rate"`        // 成功率

    // 记忆和上下文
    Memory        map[string]interface{} `json:"memory"`         // 短期记忆
    LongTermMemory map[string]interface{} `json:"long_term_memory"` // 长期记忆
    ContextWindow []Message              `json:"context_window"` // 上下文窗口

    // 工具使用统计
    ToolsUsed         []string `json:"tools_used"`         // 已使用的工具列表
    ToolUsageStats    map[string]int `json:"tool_usage_stats"` // 工具使用统计
    PreferredTools    []string `json:"preferred_tools"`    // 偏好工具

    // 学习和适应
    LearnedPatterns   map[string]interface{} `json:"learned_patterns"`   // 学习到的模式
    AdaptationHistory []AdaptationEvent      `json:"adaptation_history"` // 适应历史

    // 健康状态
    HealthStatus      string    `json:"health_status"`      // 健康状态: healthy/degraded/unhealthy
    LastHealthCheck   time.Time `json:"last_health_check"`  // 最后健康检查时间
    HealthScore       float64   `json:"health_score"`       // 健康评分(0-1)

    // 并发控制
    ActiveTasks       int       `json:"active_tasks"`       // 活跃任务数
    QueueDepth        int       `json:"queue_depth"`        // 队列深度

    // 版本信息
    ConfigVersion     string    `json:"config_version"`     // 配置版本
    CodeVersion       string    `json:"code_version"`       // 代码版本
}

// AgentStateManager 代理状态管理器
type AgentStateManager struct {
    sessionManager *session.Manager
    metrics        *AgentMetrics
    logger         *zap.Logger
}

// UpdateAgentState 更新代理状态
func (asm *AgentStateManager) UpdateAgentState(ctx context.Context, sessionID string, agentID string, updateFn func(*AgentState) error) error {
    // 1. 获取会话
    session, err := asm.sessionManager.Get(ctx, sessionID)
    if err != nil {
        return fmt.Errorf("failed to get session: %w", err)
    }

    // 2. 获取或创建代理状态
    agentState, exists := session.AgentStates[agentID]
    if !exists {
        agentState = &AgentState{
            AgentID:       agentID,
            AgentName:     agentID, // 临时名称，后续更新
            CreatedAt:     time.Now(),
            State:         "created",
            HealthStatus:  "healthy",
            HealthScore:   1.0,
            ConfigVersion: "1.0",
        }
        session.AgentStates[agentID] = agentState
    }

    // 3. 应用更新函数
    if err := updateFn(agentState); err != nil {
        return fmt.Errorf("failed to update agent state: %w", err)
    }

    // 4. 更新元信息
    agentState.LastActive = time.Now()
    agentState.HealthScore = asm.calculateHealthScore(agentState)

    // 5. 持久化会话
    if err := asm.sessionManager.Save(ctx, session); err != nil {
        return fmt.Errorf("failed to save session: %w", err)
    }

    return nil
}

// RecordExecution 记录执行结果
func (asm *AgentStateManager) RecordExecution(ctx context.Context, sessionID string, agentID string, result *ExecutionResult) error {
    return asm.UpdateAgentState(ctx, sessionID, agentID, func(state *AgentState) error {
        // 更新执行统计
        state.ExecutionCount++
        if result.Success {
            state.SuccessCount++
        } else {
            state.ErrorCount++
        }

        // 更新性能指标
        state.TotalTokensUsed += result.TokensUsed
        state.TotalCostUSD += result.CostUSD

        // 重新计算平均延迟
        if state.ExecutionCount == 1 {
            state.AverageLatency = result.LatencyMs
        } else {
            // 指数移动平均
            alpha := 0.1
            state.AverageLatency = alpha*result.LatencyMs + (1-alpha)*state.AverageLatency
        }

        // 更新成功率
        state.SuccessRate = float64(state.SuccessCount) / float64(state.ExecutionCount)

        // 记录工具使用
        if result.ToolUsed != "" {
            state.ToolsUsed = append(state.ToolsUsed, result.ToolUsed)
            if state.ToolUsageStats == nil {
                state.ToolUsageStats = make(map[string]int)
            }
            state.ToolUsageStats[result.ToolUsed]++

            // 更新偏好工具（使用频率最高的3个）
            state.PreferredTools = asm.calculatePreferredTools(state.ToolUsageStats)
        }

        // 更新状态
        if result.Success {
            state.State = "idle"
        } else {
            state.State = "error"
        }

        return nil
    })
}

// calculateHealthScore 计算健康评分
func (asm *AgentStateManager) calculateHealthScore(state *AgentState) float64 {
    if state.ExecutionCount == 0 {
        return 1.0 // 新创建的代理满分
    }

    // 基于多个维度的健康评分
    successRateScore := state.SuccessRate              // 成功率权重
    recentActivityScore := asm.calculateActivityScore(state.LastActive) // 活跃度权重
    errorRatePenalty := 1.0 - (float64(state.ErrorCount) / float64(state.ExecutionCount)) // 错误率惩罚

    // 加权平均
    score := successRateScore*0.5 + recentActivityScore*0.3 + errorRatePenalty*0.2

    // 限制在0-1范围内
    return math.Max(0.0, math.Min(1.0, score))
}

// calculateActivityScore 计算活跃度评分
func (asm *AgentStateManager) calculateActivityScore(lastActive time.Time) float64 {
    hoursSinceActive := time.Since(lastActive).Hours()

    // 24小时内活跃：满分1.0
    // 7天内活跃：0.7
    // 30天内活跃：0.3
    // 超过30天：0.1
    switch {
    case hoursSinceActive < 24:
        return 1.0
    case hoursSinceActive < 24*7:
        return 0.7
    case hoursSinceActive < 24*30:
        return 0.3
    default:
        return 0.1
    }
}
```

**状态管理的设计哲学**：

1. **全面的状态追踪**：
   - 执行统计：成功率、延迟、成本
   - 工具使用：偏好分析、使用频率
   - 健康监控：状态评分、活跃度
   - 学习适应：模式识别、行为调整

2. **原子性更新**：
   ```go
   // 使用函数式更新保证状态一致性
   func (state *AgentState) error
   // 避免竞态条件和部分更新
   ```

3. **性能指标计算**：
   - 指数移动平均：平滑延迟计算
   - 加权健康评分：多维度综合评估
   - 动态偏好学习：基于使用频率调整

## 工作流（Workflow）：编排的艺术

### 策略模式的工作流架构

Shannon的工作流系统采用策略模式设计，让我们深入分析其架构和实现机制。

#### 策略接口的设计哲学

**这块代码展示了什么？**

这段代码定义了工作流策略接口，采用策略模式设计。背景是：不同复杂度的任务需要不同的执行策略，策略模式允许动态选择和扩展执行逻辑，同时保持统一的接口契约。

这段代码的目的是说明如何通过面向接口设计实现灵活的工作流编排。

```go
// go/orchestrator/internal/workflows/strategies/types.go

// Strategy 工作流策略接口 - 定义执行契约
type Strategy interface {
    // Name 返回策略名称
    Name() string

    // CanHandle 判断是否能处理给定的任务
    CanHandle(input TaskInput) bool

    // EstimateComplexity 预估任务复杂度
    EstimateComplexity(input TaskInput) (*ComplexityEstimate, error)

    // Execute 执行任务的核心逻辑
    Execute(ctx context.Context, input TaskInput) (TaskResult, error)

    // ValidateResult 验证执行结果
    ValidateResult(result TaskResult) error

    // GetMetadata 返回策略元信息
    GetMetadata() StrategyMetadata
}

// StrategyMetadata 策略元信息
type StrategyMetadata struct {
    Name         string            `json:"name"`
    Description  string            `json:"description"`
    Version      string            `json:"version"`
    Capabilities []string          `json:"capabilities"` // 支持的能力
    Limitations  []string          `json:"limitations"`  // 限制条件
    Performance  PerformanceProfile `json:"performance"` // 性能特征
}

// PerformanceProfile 性能特征描述
type PerformanceProfile struct {
    TypicalLatency    time.Duration `json:"typical_latency"`    // 典型延迟
    MaxConcurrency    int           `json:"max_concurrency"`    // 最大并发度
    MemoryUsage       string        `json:"memory_usage"`       // 内存使用量级
    NetworkIO         string        `json:"network_io"`         // 网络IO量级
    CostEfficiency    float64       `json:"cost_efficiency"`    // 成本效率评分
}

// ComplexityEstimate 复杂度预估
type ComplexityEstimate struct {
    Score           float64         `json:"score"`            // 复杂度分数(0-1)
    Reasoning       string          `json:"reasoning"`        // 推理过程
    RecommendedStrategy string      `json:"recommended_strategy"` // 推荐策略
    EstimatedDuration time.Duration `json:"estimated_duration"` // 预估耗时
    ResourceRequirements ResourceReq `json:"resource_requirements"` // 资源需求
}
```

**策略模式的设计优势**：

1. **可扩展性**：
   ```go
   // 新策略只需实现Strategy接口
   // 无需修改现有代码
   type CustomStrategy struct {
       /* 自定义实现 */
   }
   func (s *CustomStrategy) Execute(ctx context.Context, input TaskInput) (TaskResult, error)
   ```

2. **可测试性**：
   ```go
   // 每个策略独立测试
   // 策略间解耦便于单元测试
   func TestDAGStrategy_Execute(t *testing.T)
   ```

3. **可配置性**：
   ```go
   // 运行时选择策略
   // 支持A/B测试和渐进式部署
   strategy := strategyRegistry.GetStrategy("dag")
   ```

#### 策略注册表和路由器的实现

```go
// go/orchestrator/internal/workflows/strategy_registry.go

// StrategyRegistry 策略注册表
type StrategyRegistry struct {
    strategies map[string]Strategy
    router     *StrategyRouter
    metrics    *StrategyMetrics
    mu         sync.RWMutex
}

// Register 注册策略
func (sr *StrategyRegistry) Register(strategy Strategy) error {
    sr.mu.Lock()
    defer sr.mu.Unlock()

    name := strategy.Name()
    if _, exists := sr.strategies[name]; exists {
        return fmt.Errorf("strategy already registered: %s", name)
    }

    sr.strategies[name] = strategy

    // 注册到路由器
    sr.router.AddRoute(name, strategy)

    sr.metrics.RecordStrategyRegistration(name)
    return nil
}

// SelectStrategy 选择最适合的策略
func (sr *StrategyRegistry) SelectStrategy(ctx context.Context, input TaskInput) (Strategy, error) {
    sr.mu.RLock()
    defer sr.mu.RUnlock()

    startTime := time.Now()

    // 1. 获取所有候选策略
    candidates := sr.getCandidateStrategies(input)

    // 2. 预估每种策略的复杂度
    estimates := make(map[string]*ComplexityEstimate)
    for _, strategy := range candidates {
        estimate, err := strategy.EstimateComplexity(input)
        if err != nil {
            sr.logger.Warn("Failed to estimate complexity",
                zap.String("strategy", strategy.Name()),
                zap.Error(err))
            continue
        }
        estimates[strategy.Name()] = estimate
    }

    // 3. 基于多维度选择最佳策略
    selectedStrategy := sr.selectBestStrategy(candidates, estimates, input)

    sr.metrics.RecordStrategySelection(selectedStrategy.Name(), time.Since(startTime))

    return selectedStrategy, nil
}

// selectBestStrategy 基于多维度选择策略
func (sr *StrategyRegistry) selectBestStrategy(
    candidates []Strategy,
    estimates map[string]*ComplexityEstimate,
    input TaskInput,
) Strategy {

    if len(candidates) == 1 {
        return candidates[0]
    }

    // 多维度评分函数
    scoreStrategy := func(strategy Strategy) float64 {
        estimate := estimates[strategy.Name()]
        if estimate == nil {
            return 0.0
        }

        // 复杂度匹配度 (0-1, 越高越好)
        complexityMatch := sr.calculateComplexityMatch(estimate.Score, input)

        // 性能效率 (0-1, 越高越好)
        performanceScore := sr.calculatePerformanceScore(strategy, input)

        // 资源效率 (0-1, 越高越好)
        resourceScore := sr.calculateResourceScore(strategy, input)

        // 成功率历史 (0-1, 越高越好)
        successRate := sr.getHistoricalSuccessRate(strategy.Name())

        // 加权综合评分
        score := complexityMatch*0.4 + performanceScore*0.3 + resourceScore*0.2 + successRate*0.1

        return score
    }

    // 选择最高分的策略
    bestStrategy := candidates[0]
    bestScore := scoreStrategy(bestStrategy)

    for _, candidate := range candidates[1:] {
        score := scoreStrategy(candidate)
        if score > bestScore {
            bestStrategy = candidate
            bestScore = score
        }
    }

    return bestStrategy
}
```

**策略选择的决策算法**：

1. **复杂度匹配**：
   ```go
   // 策略应该匹配任务复杂度
   // 简单任务用简单策略，复杂任务用复杂策略
   // 避免资源浪费或能力不足
   ```

2. **性能优化**：
   ```go
   // 考虑策略的性能特征
   // 选择延迟最低、并发度最高的策略
   ```

3. **资源效率**：
   ```go
   // 评估内存使用、网络IO等资源消耗
   // 选择最符合当前系统负载的策略
   ```

4. **历史成功率**：
   ```go
   // 基于过往执行数据选择可靠的策略
   // 学习和适应系统行为变化
   ```

#### DAG策略的深度实现

**这块代码展示了什么？**

这段代码定义了DAGStrategy结构体，用于实现有向无环图工作流策略。背景是：复杂AI任务通常包含多个相互依赖的子任务，DAG（Directed Acyclic Graph）提供了完美的数学模型来表达和执行这些任务依赖关系。

这段代码的目的是说明如何通过DAG模型实现复杂任务的编排和并行执行。

```go
// go/orchestrator/internal/workflows/strategies/dag_strategy.go

// DAGStrategy 有向无环图工作流策略
type DAGStrategy struct {
    name         string
    description  string
    version      string

    // 核心组件
    decomposer   *TaskDecomposer      // 任务分解器
    scheduler    *DAGScheduler        // DAG调度器
    synthesizer  *ResultSynthesizer   // 结果合成器
    validator    *DAGValidator        // DAG验证器

    // 并发控制
    maxConcurrency int
    semaphore      chan struct{}       // 并发限制信号量

    // 监控和指标
    metrics       *DAGMetrics
    tracer        trace.Tracer
    logger        *zap.Logger
}

// Execute 执行DAG工作流
func (ds *DAGStrategy) Execute(ctx context.Context, input TaskInput) (TaskResult, error) {
    span, ctx := ds.tracer.Start(ctx, "DAGStrategy.Execute")
    defer span.End()

    startTime := time.Now()
    workflowID := ds.generateWorkflowID(input)

    ds.logger.Info("Starting DAG workflow execution",
        zap.String("workflow_id", workflowID),
        zap.String("task_id", input.RequestID))

    // 1. 任务分解 - 生成子任务DAG
    dag, err := ds.decomposer.DecomposeToDAG(ctx, input)
    if err != nil {
        ds.metrics.RecordDecompositionError()
        return TaskResult{}, fmt.Errorf("failed to decompose task: %w", err)
    }

    ds.metrics.RecordDecompositionSuccess(dag.NodeCount(), dag.EdgeCount())

    // 2. DAG验证 - 确保无环且可调度
    if err := ds.validator.ValidateDAG(dag); err != nil {
        ds.metrics.RecordValidationError()
        return TaskResult{}, fmt.Errorf("DAG validation failed: %w", err)
    }

    // 3. 拓扑排序 - 确定执行顺序
    executionPlan, err := ds.scheduler.CreateExecutionPlan(dag)
    if err != nil {
        ds.metrics.RecordSchedulingError()
        return TaskResult{}, fmt.Errorf("failed to create execution plan: %w", err)
    }

    ds.logger.Info("DAG execution plan created",
        zap.Int("total_tasks", len(executionPlan.Tasks)),
        zap.Int("parallel_groups", len(executionPlan.ParallelGroups)))

    // 4. 并行执行 - 按层并发执行
    executionCtx, cancel := context.WithTimeout(ctx, input.TimeoutSeconds*time.Second)
    defer cancel()

    results, err := ds.executeDAG(executionCtx, executionPlan, workflowID)
    if err != nil {
        ds.metrics.RecordExecutionError()
        return TaskResult{}, fmt.Errorf("DAG execution failed: %w", err)
    }

    // 5. 结果合成 - 聚合所有子任务结果
    finalResult, err := ds.synthesizer.SynthesizeResults(ctx, results, input)
    if err != nil {
        ds.metrics.RecordSynthesisError()
        return TaskResult{}, fmt.Errorf("result synthesis failed: %w", err)
    }

    // 6. 结果验证
    if err := ds.ValidateResult(finalResult); err != nil {
        ds.metrics.RecordValidationError()
        return TaskResult{}, fmt.Errorf("result validation failed: %w", err)
    }

    // 7. 完善结果元信息
    finalResult.WorkflowID = workflowID
    finalResult.Status = "completed"
    finalResult.StartedAt = startTime
    finalResult.CompletedAt = time.Now()
    finalResult.DurationMs = finalResult.CompletedAt.Sub(startTime).Milliseconds()

    ds.metrics.RecordExecutionSuccess(time.Since(startTime))
    return finalResult, nil
}

// executeDAG 执行DAG的核心逻辑
func (ds *DAGStrategy) executeDAG(ctx context.Context, plan *ExecutionPlan, workflowID string) ([]TaskResult, error) {
    // 初始化结果收集器
    resultCollector := NewResultCollector(plan.TotalTasks)

    // 按层执行
    for layerIndex, layer := range plan.ParallelGroups {
        ds.logger.Info("Executing DAG layer",
            zap.Int("layer", layerIndex),
            zap.Int("task_count", len(layer.Tasks)))

        // 创建当前层的执行任务
        var wg sync.WaitGroup
        errorChan := make(chan error, len(layer.Tasks))

        for _, task := range layer.Tasks {
            wg.Add(1)
            go func(t SubTask) {
                defer wg.Done()

                // 获取信号量（限制并发）
                select {
                case ds.semaphore <- struct{}{}:
                    defer func() { <-ds.semaphore }()
                case <-ctx.Done():
                    errorChan <- ctx.Err()
                    return
                }

                // 执行子任务
                result, err := ds.executeSubTask(ctx, t, workflowID)
                if err != nil {
                    errorChan <- fmt.Errorf("subtask %s failed: %w", t.ID, err)
                    return
                }

                // 收集结果
                resultCollector.Collect(t.ID, result)
            }(task)
        }

        // 等待当前层完成
        wg.Wait()
        close(errorChan)

        // 检查是否有错误
        if err := <-errorChan; err != nil {
            return nil, err
        }

        // 检查是否所有前置依赖都已满足
        if !resultCollector.AreDependenciesSatisfied(layer) {
            return nil, errors.New("layer dependencies not satisfied")
        }
    }

    return resultCollector.GetAllResults(), nil
}

// executeSubTask 执行单个子任务
func (ds *DAGStrategy) executeSubTask(ctx context.Context, task SubTask, workflowID string) (TaskResult, error) {
    span, ctx := ds.tracer.Start(ctx, "executeSubTask",
        trace.WithAttributes(
            attribute.String("subtask.id", task.ID),
            attribute.String("subtask.type", task.Type),
        ))
    defer span.End()

    // 1. 创建子任务输入
    subInput := TaskInput{
        Query:       task.Description,
        UserID:      task.UserID,
        TenantID:    task.TenantID,
        SessionID:   task.SessionID,
        Context:     task.Context,
        Mode:        "simple", // 子任务通常简单
        ParentWorkflowID: workflowID,
        MaxTokens:   task.MaxTokens,
        TimeoutSeconds: task.TimeoutSeconds,
    }

    // 2. 选择子任务策略（递归使用策略选择器）
    subStrategy, err := ds.strategyRegistry.SelectStrategy(ctx, subInput)
    if err != nil {
        return TaskResult{}, fmt.Errorf("failed to select strategy for subtask: %w", err)
    }

    // 3. 执行子任务
    result, err := subStrategy.Execute(ctx, subInput)
    if err != nil {
        return TaskResult{}, fmt.Errorf("subtask execution failed: %w", err)
    }

    // 4. 添加子任务标识
    result.TaskID = task.ID
    result.WorkflowID = workflowID

    return result, nil
}
```

**DAG策略的核心机制**：

1. **任务分解**：
   ```go
   // 将复杂任务分解为有依赖关系的子任务
   // 生成DAG图结构：节点=子任务，边=依赖关系
   dag := decomposer.DecomposeToDAG(input)
   ```

2. **拓扑排序**：
   ```go
   // 确定执行顺序：无依赖的任务先执行
   // 分层组织：每层任务可并行执行
   executionPlan := scheduler.CreateExecutionPlan(dag)
   ```

3. **分层并发执行**：
   ```go
   // 信号量限制总并发数
   // 按层顺序执行，保证依赖满足
   ds.semaphore <- struct{}{} // 获取执行许可
   ```

4. **结果收集和合成**：
   ```go
   // 收集所有子结果
   // 智能合成最终答案
   finalResult := synthesizer.SynthesizeResults(results)
   ```

#### 任务分解器的实现

```go
// go/orchestrator/internal/workflows/decomposition/task_decomposer.go

// TaskDecomposer 任务分解器
type TaskDecomposer struct {
    llmClient     *llm.Client
    promptManager *prompts.Manager
    validator     *DecompositionValidator
    metrics       *DecompositionMetrics
    logger        *zap.Logger
}

// DecomposeToDAG 将任务分解为DAG
func (td *TaskDecomposer) DecomposeToDAG(ctx context.Context, input TaskInput) (*DAG, error) {
    // 1. 生成分解提示
    prompt := td.buildDecompositionPrompt(input)

    // 2. 调用LLM进行任务分解
    response, err := td.llmClient.Complete(ctx, llm.CompletionRequest{
        Model:       "gpt-4", // 使用复杂推理模型
        Messages:    []llm.Message{{Role: "user", Content: prompt}},
        Temperature: 0.1,     // 低温度保证确定性
        MaxTokens:   2000,
    })
    if err != nil {
        return nil, fmt.Errorf("LLM decomposition failed: %w", err)
    }

    // 3. 解析LLM响应
    decomposition, err := td.parseDecompositionResponse(response.Content)
    if err != nil {
        return nil, fmt.Errorf("failed to parse decomposition: %w", err)
    }

    // 4. 验证分解结果
    if err := td.validator.ValidateDecomposition(decomposition); err != nil {
        return nil, fmt.Errorf("decomposition validation failed: %w", err)
    }

    // 5. 构建DAG图
    dag, err := td.buildDAGFromDecomposition(decomposition)
    if err != nil {
        return nil, fmt.Errorf("failed to build DAG: %w", err)
    }

    td.metrics.RecordDecompositionSuccess(len(decomposition.Subtasks))
    return dag, nil
}

// buildDecompositionPrompt 构建分解提示
func (td *TaskDecomposer) buildDecompositionPrompt(input TaskInput) string {
    return fmt.Sprintf(`请将以下复杂任务分解为多个相互依赖的子任务：

任务：%s

上下文信息：
%s

请以JSON格式返回分解结果：
{
  "subtasks": [
    {
      "id": "subtask_1",
      "description": "具体的子任务描述",
      "dependencies": ["subtask_2"],  // 可选的依赖任务ID
      "estimated_complexity": 0.3,     // 复杂度评分(0-1)
      "required_tools": ["web_search"], // 需要的工具
      "success_criteria": "完成的标准"  // 成功判定条件
    }
  ],
  "execution_strategy": "parallel_when_possible"
}

要求：
1. 每个子任务应该是独立的、可并行的
2. 明确标识任务间的依赖关系
3. 确保没有循环依赖
4. 任务粒度适中，既不太大也不太小`, input.Query, td.formatContext(input.Context))
}

// parseDecompositionResponse 解析LLM响应
func (td *TaskDecomposer) parseDecompositionResponse(content string) (*TaskDecomposition, error) {
    // 提取JSON内容
    jsonContent := td.extractJSONFromResponse(content)
    if jsonContent == "" {
        return nil, errors.New("no JSON content found in response")
    }

    // 解析JSON
    var decomposition TaskDecomposition
    if err := json.Unmarshal([]byte(jsonContent), &decomposition); err != nil {
        return nil, fmt.Errorf("JSON parse error: %w", err)
    }

    return &decomposition, nil
}

// buildDAGFromDecomposition 从分解结果构建DAG
func (td *TaskDecomposer) buildDAGFromDecomposition(decomp *TaskDecomposition) (*DAG, error) {
    dag := NewDAG()

    // 添加所有节点
    for _, subtask := range decomp.Subtasks {
        node := &DAGNode{
            ID:          subtask.ID,
            Description: subtask.Description,
            Task:        subtask,
            State:       NodeStatePending,
        }
        if err := dag.AddNode(node); err != nil {
            return nil, fmt.Errorf("failed to add node %s: %w", subtask.ID, err)
        }
    }

    // 添加所有边（依赖关系）
    for _, subtask := range decomp.Subtasks {
        for _, depID := range subtask.Dependencies {
            if err := dag.AddEdge(depID, subtask.ID); err != nil {
                return nil, fmt.Errorf("failed to add edge %s -> %s: %w", depID, subtask.ID, err)
            }
        }
    }

    // 验证DAG有效性
    if err := dag.Validate(); err != nil {
        return nil, fmt.Errorf("DAG validation failed: %w", err)
    }

    return dag, nil
}
```

**任务分解的设计哲学**：

1. **LLM驱动的智能分解**：
   ```go
   // 使用GPT-4进行复杂任务理解和分解
   // 结合领域知识和推理能力
   // 生成更准确合理的子任务划分
   ```

2. **依赖关系建模**：
   ```go
   // 明确建模任务间的依赖关系
   // 支持并行执行和顺序约束
   // 避免不必要的串行化
   ```

3. **验证和纠错**：
   ```go
   // 验证分解结果的合理性
   // 检查依赖关系的一致性
   // 修复潜在的循环依赖问题
   ```

#### 结果合成器的实现

```go
// go/orchestrator/internal/workflows/synthesis/result_synthesizer.go

// ResultSynthesizer 结果合成器
type ResultSynthesizer struct {
    llmClient     *llm.Client
    templateEngine *templates.Engine
    validator     *SynthesisValidator
    metrics       *SynthesisMetrics
    logger        *zap.Logger
}

// SynthesizeResults 合成最终结果
func (rs *ResultSynthesizer) SynthesizeResults(ctx context.Context, results []TaskResult, originalInput TaskInput) (TaskResult, error) {
    // 1. 过滤和排序结果
    validResults := rs.filterValidResults(results)

    // 2. 选择合成策略
    strategy := rs.selectSynthesisStrategy(validResults, originalInput)

    // 3. 执行合成
    switch strategy {
    case SynthesisStrategyConcatenate:
        return rs.concatenateResults(validResults, originalInput)
    case SynthesisStrategySummarize:
        return rs.summarizeResults(ctx, validResults, originalInput)
    case SynthesisStrategySynthesize:
        return rs.synthesizeWithLLM(ctx, validResults, originalInput)
    case SynthesisStrategyTemplate:
        return rs.applyTemplate(validResults, originalInput)
    default:
        return rs.concatenateResults(validResults, originalInput) // 默认策略
    }
}

// synthesizeWithLLM 使用LLM进行智能合成
func (rs *ResultSynthesizer) synthesizeWithLLM(ctx context.Context, results []TaskResult, originalInput TaskInput) (TaskResult, error) {
    // 1. 构建合成提示
    prompt := rs.buildSynthesisPrompt(results, originalInput)

    // 2. 调用LLM生成合成结果
    response, err := rs.llmClient.Complete(ctx, llm.CompletionRequest{
        Model:       "gpt-4",
        Messages:    []llm.Message{{Role: "user", Content: prompt}},
        Temperature: 0.2, // 平衡创造性和一致性
        MaxTokens:   1000,
    })
    if err != nil {
        return TaskResult{}, fmt.Errorf("LLM synthesis failed: %w", err)
    }

    // 3. 构建最终结果
    finalResult := TaskResult{
        TaskID:      originalInput.RequestID,
        Status:      "completed",
        Result:      response.Content,
        ResultType:  "text",
        Citations:   rs.extractCitations(results),
        Confidence:  rs.calculateOverallConfidence(results),
        ToolCalls:   rs.aggregateToolCalls(results),
        TokenUsage:  rs.sumTokenUsage(results),
        CostUSD:     rs.sumCost(results),
        StartedAt:   rs.findEarliestStart(results),
        CompletedAt: time.Now(),
    }

    // 4. 添加元信息
    finalResult.Metadata = map[string]interface{}{
        "synthesis_strategy": "llm_synthesis",
        "subtask_count":      len(results),
        "synthesis_model":    "gpt-4",
    }

    return finalResult, nil
}

// buildSynthesisPrompt 构建合成提示
func (rs *ResultSynthesizer) buildSynthesisPrompt(results []TaskResult, originalInput TaskInput) string {
    var buffer strings.Builder

    buffer.WriteString(fmt.Sprintf("请基于以下子任务的结果，合成对原始问题的完整回答：\n\n原始问题：%s\n\n", originalInput.Query))

    // 添加每个子任务的结果
    for i, result := range results {
        buffer.WriteString(fmt.Sprintf("子任务 %d：\n%s\n\n", i+1, result.Result))
    }

    buffer.WriteString(`要求：
1. 综合所有子任务的结果
2. 提供完整、连贯的回答
3. 如果有冲突的信息，请解释并给出最合理的结论
4. 保持客观中立的态度
5. 如果适用，提供具体的证据或引用来源`)

    return buffer.String()
}

// selectSynthesisStrategy 选择合成策略
func (rs *ResultSynthesizer) selectSynthesisStrategy(results []TaskResult, originalInput TaskInput) SynthesisStrategy {
    resultCount := len(results)

    // 单结果：直接返回
    if resultCount == 1 {
        return SynthesisStrategyDirect
    }

    // 多结果：根据类型和复杂度选择策略
    totalTokens := rs.sumTokens(results)
    hasCitations := rs.hasCitations(results)

    // 高复杂度且有引用：使用LLM合成
    if totalTokens > 1000 && hasCitations {
        return SynthesisStrategySynthesize
    }

    // 中等复杂度：使用摘要
    if totalTokens > 500 {
        return SynthesisStrategySummarize
    }

    // 低复杂度：直接连接
    return SynthesisStrategyConcatenate
}
```

**结果合成的设计哲学**：

1. **策略化合成**：
   ```go
   // 根据结果特征选择最适合的合成策略
   // 单结果直接返回，多结果智能合并
   // 平衡效率和质量
   ```

2. **LLM增强合成**：
   ```go
   // 使用GPT-4进行最终答案的生成
   // 解决子任务结果间的冲突和不一致
   // 提供更连贯和完整的回答
   ```

3. **元信息保留**：
   ```go
   // 聚合所有子任务的统计信息
   // 保留引用和工具调用追踪
   // 支持结果的可验证性和可审计性
   ```

## 核心概念的协作关系

### 数据流图解

```
用户查询
    ↓
  任务创建 (TaskInput)
    ↓
  会话查找/创建 (Session)
    ↓
  路由决策 (Router)
    ↓
  工作流执行 (Workflow)
    ↓
  代理分配 (Agent Assignment)
    ↓
  工具执行 (Tool Execution)
    ↓
  结果合成 (Synthesis)
    ↓
  会话更新 (Session Update)
    ↓
  返回结果
```

### 关键协作点

1. **任务 → 会话**：任务总是发生在会话上下文中
2. **任务 → 工作流**：路由器根据任务特征选择工作流策略
3. **工作流 → 代理**：工作流编排代理的执行顺序
4. **代理 → 会话**：代理状态保存在会话中
5. **代理 → 工具**：代理调用工具执行具体操作

## 实际应用中的价值

### 多轮对话的连续性

```go
// 第一轮对话
task1 := TaskInput{
    Query:     "苹果最新的AI投资情况",
    SessionID: "session-123",
}
// 系统记住上下文，了解这是关于苹果公司的查询

// 第二轮对话
task2 := TaskInput{
    Query:     "和谷歌比怎么样",
    SessionID: "session-123",
    History:   session.GetRecentHistory(5),
}
// 系统知道"它"指的是苹果的AI投资，自动补充上下文
```

### 复杂任务的分解执行

对于复杂问题，系统会自动分解：

```go
// 输入：分析苹果公司的AI战略
// 分解为：
subtask1 := Subtask{ID: "research_apple_ai", Description: "研究苹果AI投资"}
subtask2 := Subtask{ID: "analyze_competition", Description: "分析竞争对手"}
subtask3 := Subtask{ID: "synthesize_findings", Description: "合成发现"}
// 每个子任务分配给专门的代理并发执行
```

### 性能监控和成本控制

```go
// 会话跟踪全局使用情况
session.UpdateTokenUsage(tokensUsed, cost)

// 代理跟踪个体性能
agentState.TokensUsed += tokensUsed
agentState.LastActive = time.Now()

// 工作流提供执行洞察
workflow.TrackExecutionMetrics(agentResults)
```

## 设计哲学的深度思考

### 为什么需要这种复杂的设计？

1. **连续性需求**：AI对话需要记忆和上下文
2. **复杂性处理**：简单问题简单处理，复杂问题深度分解
3. **可靠性要求**：生产系统不能容忍失败
4. **可观测性**：需要完全理解和调试执行过程

### 与传统架构的对比

| 特性 | Shannon | 传统AI服务 | ChatGPT API |
|------|---------|-----------|-------------|
| 连续性 | ✅ 会话记忆 | ❌ 无状态 | ✅ 会话ID |
| 复杂任务 | ✅ 自动分解 | ❌ 单次调用 | ❌ 单次调用 |
| 可靠性 | ✅ 工作流重试 | ⚠️ 客户端处理 | ⚠️ 客户端处理 |
| 可观测性 | ✅ 完整追踪 | ❌ 黑盒 | ❌ 黑盒 |
| 成本控制 | ✅ 令牌预算 | ❌ 无限制 | ❌ 无限制 |

### 关键创新点

1. **会话驱动架构**：一切都围绕会话构建
2. **代理协作模式**：多个代理协同工作
3. **声明式工作流**：通过配置定义执行逻辑
4. **确定性重放**：生产问题可完全重现

## 总结：一个有机整体

Shannon的核心概念不是孤立的组件，而是一个**有机的整体**：

- **任务**是触发器，定义了要做什么
- **会话**是容器，提供了连续性和记忆
- **代理**是执行者，提供了智能和专业化
- **工作流**是编排者，协调整个执行过程

这种设计让Shannon不仅能处理简单的问答，还能胜任复杂的、多步骤的AI任务。从架构的角度看，这是一个**生产级AI系统的教科书范例**。

在接下来的文章中，我们将深入探索Go Orchestrator的具体实现，了解这个调度中枢是如何协调这一切的。敬请期待！
