# 《活动系统：AI代理的执行心脏》

> **专栏语录**：在AI代理的世界里，活动就像心脏的瓣膜：看似简单，却承担着系统最核心的执行责任。Shannon的活动系统不是简单的函数集合，而是用工程思维重新设计的分布式执行引擎。本文将揭秘活动系统如何将复杂的AI任务分解为可管理、可监控、可扩展的执行单元。

## 第一章：活动系统的起源危机

### 巨石函数的反模式

在Shannon早期，我们有一个叫`processUserRequest`的巨石函数：

**这块代码展示了什么？**

这段代码演示了早期Shannon系统的巨石函数问题，将认证、限流、AI推理、工具调用等所有逻辑耦合在一个函数中。背景是：传统的单体架构将所有业务逻辑塞进一个函数，导致代码难以维护、测试困难、故障难以隔离，是典型的反模式设计。

这段代码的目的是说明为什么需要活动系统来拆分和重组这些复杂的业务逻辑。

```go
// 噩梦般的巨石函数 - 早期Shannon的耻辱
func processUserRequest(ctx context.Context, userID string, query string) (*Response, error) {
    startTime := time.Now()

    // 1. 用户认证和授权
    user, err := db.GetUser(userID)
    if err != nil {
        return nil, fmt.Errorf("user lookup failed: %w", err)
    }
    if !user.HasPermission("ai_query") {
        return nil, errors.New("permission denied")
    }

    // 2. 速率限制检查
    rateLimitKey := fmt.Sprintf("rate_limit:%s", userID)
    currentRequests, err := redis.Get(rateLimitKey).Int()
    if err != nil {
        return nil, fmt.Errorf("rate limit check failed: %w", err)
    }
    if currentRequests >= 100 {
        return nil, errors.New("rate limit exceeded")
    }
    redis.Incr(rateLimitKey)

    // 3. 意图识别 - 调用AI模型
    intent, err := aiClient.ClassifyIntent(query)
    if err != nil {
        return nil, fmt.Errorf("intent classification failed: %w", err)
    }

    // 4. 工具选择和参数准备
    var toolResponse interface{}
    switch intent.Tool {
    case "web_search":
        searchQuery := intent.Parameters["query"].(string)
        toolResponse, err = webSearchClient.Search(searchQuery)
        if err != nil {
            return nil, fmt.Errorf("web search failed: %w", err)
        }
    case "code_execution":
        code := intent.Parameters["code"].(string)
        toolResponse, err = codeExecutor.Run(code)
        if err != nil {
            return nil, fmt.Errorf("code execution failed: %w", err)
        }
    case "data_analysis":
        data := intent.Parameters["data"].(string)
        toolResponse, err = dataAnalyzer.Analyze(data)
        if err != nil {
            return nil, fmt.Errorf("data analysis failed: %w", err)
        }
    }

    // 5. 结果后处理
    processedResult, err := postProcessResult(toolResponse, intent)
    if err != nil {
        return nil, fmt.Errorf("result post-processing failed: %w", err)
    }

    // 6. 缓存结果
    cacheKey := fmt.Sprintf("result:%s:%s", userID, hash(query))
    redis.Set(cacheKey, processedResult, 1*time.Hour)

    // 7. 记录指标
    metrics.RecordRequest(userID, intent.Tool, time.Since(startTime))

    return &Response{
        Result: processedResult,
        ToolUsed: intent.Tool,
        ProcessingTime: time.Since(startTime),
    }, nil
}
```

**这个巨石函数的问题**：

1. **单一职责原则违反**：一个函数做了7件不同的事
2. **错误处理地狱**：7层嵌套的错误处理，难以调试
3. **测试不可能**：无法独立测试认证、工具调用、缓存等功能
4. **扩展灾难**：加一个新工具需要修改整个函数
5. **性能瓶颈**：所有操作串行执行，无法并发
6. **监控盲区**：无法精确监控每个步骤的性能

最可怕的是，**这个函数经常崩溃**，而且每次崩溃都像俄罗斯套娃一样，需要层层剥开才能找到根本原因。

### 活动系统的诞生：分治的智慧

我们终于意识到：**复杂系统不能用复杂函数来解决，而应该用简单函数的组合来解决**。

Shannon的活动系统基于一个简单而深刻的理念：**将复杂的业务逻辑分解为独立的、可复用的、可测试的活动单元**。

```go
// 活动系统重构后的优雅
func processUserRequest(ctx context.Context, userID string, query string) (*Response, error) {
    // 创建工作流定义
    workflow := &Workflow{
        Activities: []Activity{
            {Name: "authenticate_user", Type: "auth.AuthenticateUser"},
            {Name: "check_rate_limit", Type: "ratelimit.CheckLimit"},
            {Name: "classify_intent", Type: "ai.ClassifyIntent"},
            {Name: "execute_tool", Type: "tool.ExecuteTool"},
            {Name: "post_process", Type: "processor.PostProcess"},
            {Name: "cache_result", Type: "cache.StoreResult"},
            {Name: "record_metrics", Type: "metrics.RecordMetrics"},
        },
    }

    // 执行工作流
    return workflowEngine.Execute(ctx, workflow, map[string]interface{}{
        "user_id": userID,
        "query": query,
    })
}
```

**活动系统的核心优势**：

1. **关注点分离**：每个活动只负责一件事
2. **独立测试**：每个活动都可以单独测试
3. **错误隔离**：一个活动失败不影响其他活动
4. **并发执行**：可以并行执行独立活动
5. **动态组合**：运行时决定执行哪些活动
6. **可观测性**：每个活动都有独立的监控指标

## 第二章：活动系统的架构哲学

### 活动定义：标准化的执行契约

活动不是普通的函数，而是标准化、可配置、可监控的执行单元：

```go
// go/orchestrator/internal/activities/types.go

**这块代码展示了什么？**

这段代码定义了ActivityDefinition结构体，是Shannon活动系统的核心数据结构。背景是：活动系统需要标准化地管理各种类型的执行单元，包括AI推理、工具调用、数据处理等，这个结构体提供了统一的配置和行为规范。

这段代码的目的是说明如何通过结构化定义实现活动的标准化管理和治理。

```go
/// ActivityDefinition - Shannon活动系统的核心定义结构，标准化了所有活动的配置和行为
/// 这个结构体定义了活动从注册到执行的完整生命周期规范，确保所有活动都遵循相同的接口和治理规则
type ActivityDefinition struct {
    // ========== 基本信息 - 活动的身份标识 ==========
    Name        string                 `json:"name"`        // 全局唯一标识，如"ai.intent_classification"、"tool.web_search"
    Description string                 `json:"description"` // 人类可读的描述，用于文档和调试
    Version     string                 `json:"version"`     // 语义化版本号，如"v1.2.3"，支持版本兼容性检查
    Category    string                 `json:"category"`    // 功能分类：auth(认证), ai(人工智能), tool(工具), cache(缓存), metrics(监控)

    // ========== 执行配置 - 运行时行为控制 ==========
    Handler     ActivityHandler        `json:"-"`           // 执行处理器接口，不序列化，运行时注入
    Timeout     time.Duration         `json:"timeout"`     // 单个活动执行超时，默认30秒，防止无限等待
    MaxRetries  int                   `json:"max_retries"` // 失败时的最大重试次数，默认3次

    // ========== 资源需求 - 系统资源分配 ==========
    ResourceRequirements ResourceRequirements `json:"resource_requirements"` // CPU、内存、GPU等资源规格声明

    // ========== 权限要求 - 安全访问控制 ==========
    RequiredPermissions []string      `json:"required_permissions"` // 所需权限列表，如["ai.execute", "tool.web_access"]

    // ========== 输入输出规格 - 数据契约定义 ==========
    InputSchema  JSONSchema          `json:"input_schema"`  // 输入数据结构的JSON Schema验证器
    OutputSchema JSONSchema          `json:"output_schema"` // 输出数据结构的JSON Schema验证器

    // ========== 可观测性配置 - 监控和追踪 ==========
    MetricsEnabled bool              `json:"metrics_enabled"` // 是否启用Prometheus指标收集，默认true
    TracingEnabled bool              `json:"tracing_enabled"` // 是否启用OpenTelemetry分布式追踪，默认true

    // ========== 错误处理策略 - 故障恢复机制 ==========
    RetryPolicy  RetryPolicy         `json:"retry_policy"`   // 指数退避重试策略配置
    CircuitBreaker CircuitBreakerConfig `json:"circuit_breaker"` // 熔断器配置，防止级联故障

    // ========== 元数据 - 扩展属性支持 ==========
    Tags         map[string]string    `json:"tags"`         // 标签键值对，用于活动过滤和分组，如{"env": "prod", "team": "ai"}
    Metadata     map[string]interface{} `json:"metadata"`   // 任意扩展元数据，如作者信息、文档链接等
}

/// 活动处理器接口 - 执行契约
type ActivityHandler interface {
    /// 执行活动的核心逻辑
    Execute(ctx context.Context, input interface{}) (interface{}, error)

    /// 验证输入参数
    ValidateInput(input interface{}) error

    /// 获取活动元数据
    GetMetadata() ActivityMetadata

    /// 可选：清理资源
    Cleanup() error
}

/// 资源需求规格
type ResourceRequirements struct {
    CPU     string `json:"cpu"`     // CPU需求："0.1", "500m"
    Memory  string `json:"memory"`  // 内存需求："128Mi", "1Gi"
    Disk    string `json:"disk"`    // 磁盘需求："100Mi"
    Network string `json:"network"` // 网络需求："100Mbps"

    // GPU需求（AI活动专用）
    GPUCount int    `json:"gpu_count,omitempty"`
    GPUType  string `json:"gpu_type,omitempty"` // "nvidia-tesla-t4"
}

/// 重试策略定义
type RetryPolicy struct {
    MaxAttempts     int           `json:"max_attempts"`     // 最大尝试次数
    InitialInterval time.Duration `json:"initial_interval"` // 初始重试间隔
    MaxInterval     time.Duration `json:"max_interval"`     // 最大重试间隔
    BackoffMultiplier float64    `json:"backoff_multiplier"` // 退避倍数
    RetryableErrors []string     `json:"retryable_errors"` // 可重试的错误类型
}
```

**活动定义的设计哲学**：标准化、可配置、可治理。

1. **标准化**：所有活动都遵循相同的接口和生命周期
2. **可配置**：运行时可以调整超时、重试、资源限制
3. **可治理**：统一的权限检查、监控、错误处理

### 活动注册表：运行时的插件系统

活动不是编译时固定的，而是运行时可注册的插件：

**这块代码展示了什么？**

这段代码展示了活动注册表的设计，支持运行时动态注册和管理活动。背景是：现代系统需要支持插件化扩展，活动注册表提供了分类索引、版本管理和标签系统，使得活动可以像插件一样动态加载和管理。

这段代码的目的是说明如何构建支持热插拔的活动插件系统。

```go
// 活动注册表 - 运行时插件系统
type ActivityRegistry struct {
    // 活动存储
    activities sync.Map // map[string]*ActivityDefinition

    // 分类索引
    categories map[string][]string // category -> []activityName

    // 标签索引
    tags map[string][]string // tag -> []activityName

    // 版本管理
    versions map[string][]VersionInfo // activityName -> []versions

    // 依赖关系
    dependencies map[string][]string // activityName -> []dependencies

    // 互斥关系
    mutexes map[string][]string // activityName -> []mutuallyExclusive
}

func (ar *ActivityRegistry) Register(activity *ActivityDefinition) error {
    // 1. 验证活动定义完整性
    if err := ar.validateDefinition(activity); err != nil {
        return fmt.Errorf("invalid activity definition: %w", err)
    }

    // 2. 检查命名冲突
    if _, exists := ar.activities.Load(activity.Name); exists {
        return fmt.Errorf("activity already registered: %s", activity.Name)
    }

    // 3. 注册活动
    ar.activities.Store(activity.Name, activity)

    // 4. 更新索引
    ar.updateCategoryIndex(activity)
    ar.updateTagIndex(activity)
    ar.updateVersionIndex(activity)

    // 5. 验证依赖关系
    if err := ar.validateDependencies(activity); err != nil {
        // 回滚注册
        ar.activities.Delete(activity.Name)
        return fmt.Errorf("dependency validation failed: %w", err)
    }

    ar.logger.Info("Activity registered successfully",
        zap.String("name", activity.Name),
        zap.String("version", activity.Version),
        zap.String("category", activity.Category))

    return nil
}

func (ar *ActivityRegistry) Get(name string) (*ActivityDefinition, error) {
    if activity, ok := ar.activities.Load(name); ok {
        return activity.(*ActivityDefinition), nil
    }

    // 尝试版本匹配：name:v1.2.3
    if strings.Contains(name, ":") {
        parts := strings.Split(name, ":")
        baseName, version := parts[0], parts[1]
        return ar.getByVersion(baseName, version)
    }

    return nil, fmt.Errorf("activity not found: %s", name)
}

func (ar *ActivityRegistry) FindByCategory(category string) []*ActivityDefinition {
    names, exists := ar.categories[category]
    if !exists {
        return nil
    }

    var activities []*ActivityDefinition
    for _, name := range names {
        if activity, ok := ar.activities.Load(name); ok {
            activities = append(activities, activity.(*ActivityDefinition))
        }
    }

    return activities
}

func (ar *ActivityRegistry) FindByTags(requiredTags map[string]string) []*ActivityDefinition {
    // 标签匹配逻辑：所有必需标签都匹配
    var candidates []*ActivityDefinition

    ar.activities.Range(func(key, value interface{}) bool {
        activity := value.(*ActivityDefinition)

        matches := true
        for tagKey, tagValue := range requiredTags {
            if activity.Tags[tagKey] != tagValue {
                matches = false
                break
            }
        }

        if matches {
            candidates = append(candidates, activity)
        }

        return true // 继续遍历
    })

    return candidates
}
```

**注册表的核心优势**：

1. **运行时扩展**：无需重新编译即可添加新活动
2. **版本管理**：支持多版本活动共存
3. **依赖管理**：自动处理活动间的依赖关系
4. **服务发现**：通过分类和标签快速查找活动

### 活动执行引擎：分布式的执行心脏

执行引擎是活动系统的核心，负责调度、监控、错误处理：

```go
// 活动执行引擎
type ExecutionEngine struct {
    // 注册表
    registry *ActivityRegistry

    // 执行器池
    executors *ExecutorPool

    // 调度器
    scheduler *ActivityScheduler

    // 监控器
    monitor *ExecutionMonitor

    // 错误处理器
    errorHandler *ErrorHandler

    // 并发控制
    semaphore chan struct{}

    // 指标收集器
    metrics *ExecutionMetrics
}

func (ee *ExecutionEngine) ExecuteActivity(ctx context.Context, request *ActivityRequest) (*ActivityResult, error) {
    executionID := ee.generateExecutionID()
    startTime := time.Now()

    // 创建执行上下文
    execCtx := &ExecutionContext{
        ExecutionID: executionID,
        ActivityName: request.Name,
        StartTime: startTime,
        TraceID: trace.SpanContextFromContext(ctx).TraceID().String(),
    }

    // 记录开始执行
    ee.monitor.RecordExecutionStart(execCtx)
    ee.metrics.RecordActivityStart(request.Name)

    defer func() {
        duration := time.Since(startTime)
        ee.metrics.RecordActivityDuration(request.Name, duration)
        ee.monitor.RecordExecutionEnd(execCtx, duration)
    }()

    // 1. 获取活动定义
    activity, err := ee.registry.Get(request.Name)
    if err != nil {
        ee.metrics.RecordActivityError(request.Name, "activity_not_found")
        return nil, fmt.Errorf("activity not found: %w", err)
    }

    // 2. 预检：权限、资源、输入验证
    if err := ee.preExecuteChecks(ctx, activity, request); err != nil {
        ee.metrics.RecordActivityError(request.Name, "precheck_failed")
        return nil, err
    }

    // 3. 获取执行器
    executor, err := ee.executors.GetExecutor(activity.ResourceRequirements)
    if err != nil {
        ee.metrics.RecordActivityError(request.Name, "no_executor_available")
        return nil, fmt.Errorf("no suitable executor: %w", err)
    }

    // 4. 执行活动（带重试）
    result, err := ee.executeWithRetry(ctx, executor, activity, request, execCtx)
    if err != nil {
        ee.errorHandler.HandleExecutionError(ctx, activity, request, err, execCtx)
        return nil, err
    }

    // 5. 后处理：结果验证、缓存、通知
    finalResult, err := ee.postExecuteProcessing(ctx, activity, result, execCtx)
    if err != nil {
        ee.metrics.RecordActivityError(request.Name, "postprocessing_failed")
        return nil, err
    }

    ee.metrics.RecordActivitySuccess(request.Name)
    return finalResult, nil
}

func (ee *ExecutionEngine) executeWithRetry(
    ctx context.Context,
    executor ActivityExecutor,
    activity *ActivityDefinition,
    request *ActivityRequest,
    execCtx *ExecutionContext,
) (*ActivityResult, error) {

    var lastErr error
    maxRetries := activity.MaxRetries

    for attempt := 0; attempt <= maxRetries; attempt++ {
        if attempt > 0 {
            // 重试前的延迟
            delay := ee.calculateRetryDelay(activity.RetryPolicy, attempt)
            select {
            case <-time.After(delay):
            case <-ctx.Done():
                return nil, ctx.Err()
            }
        }

        // 执行尝试
        result, err := executor.Execute(ctx, activity, request.Input)
        if err == nil {
            // 执行成功
            if attempt > 0 {
                ee.metrics.RecordActivityRetrySuccess(activity.Name, attempt)
            }
            return result, nil
        }

        lastErr = err

        // 检查是否可重试
        if !ee.isRetryableError(err, activity.RetryPolicy) {
            break
        }

        ee.metrics.RecordActivityRetry(activity.Name, attempt, err)
        execCtx.RetryCount = attempt + 1
    }

    return nil, fmt.Errorf("activity execution failed after %d attempts: %w", maxRetries+1, lastErr)
}

func (ee *ExecutionEngine) calculateRetryDelay(policy RetryPolicy, attempt int) time.Duration {
    delay := time.Duration(float64(policy.InitialInterval) * math.Pow(policy.BackoffMultiplier, float64(attempt-1)))

    if delay > policy.MaxInterval {
        delay = policy.MaxInterval
    }

    // 添加随机抖动，避免惊群效应
    jitter := time.Duration(rand.Float64() * float64(delay) * 0.1)
    return delay + jitter
}
```

## 第三章：核心活动实现剖析

### 认证活动：安全的第一道防线

```go
// 认证活动实现
type AuthenticateUserActivity struct {
    userStore    UserStore
    tokenService TokenService
    metrics      *ActivityMetrics
}

type AuthenticateUserInput struct {
    Token     string `json:"token"`
    UserAgent string `json:"user_agent,omitempty"`
    ClientIP  string `json:"client_ip,omitempty"`
}

type AuthenticateUserOutput struct {
    UserID      string            `json:"user_id"`
    UserInfo    UserInfo          `json:"user_info"`
    Permissions []string          `json:"permissions"`
    Metadata    map[string]interface{} `json:"metadata"`
}

func (a *AuthenticateUserActivity) Execute(ctx context.Context, input interface{}) (interface{}, error) {
    req, ok := input.(*AuthenticateUserInput)
    if !ok {
        return nil, errors.New("invalid input type")
    }

    startTime := time.Now()
    defer func() {
        a.metrics.RecordExecutionTime("authenticate_user", time.Since(startTime))
    }()

    // 1. 验证token格式
    if !a.tokenService.IsValidFormat(req.Token) {
        a.metrics.RecordError("authenticate_user", "invalid_token_format")
        return nil, &AuthenticationError{Type: "invalid_format"}
    }

    // 2. 验证token并获取用户信息
    claims, err := a.tokenService.ValidateToken(req.Token)
    if err != nil {
        a.metrics.RecordError("authenticate_user", "token_validation_failed")
        return nil, &AuthenticationError{Type: "invalid_token", Cause: err}
    }

    // 3. 获取用户完整信息
    user, err := a.userStore.GetUserByID(claims.UserID)
    if err != nil {
        a.metrics.RecordError("authenticate_user", "user_lookup_failed")
        return nil, &AuthenticationError{Type: "user_not_found", Cause: err}
    }

    // 4. 检查用户状态
    if user.Status != "active" {
        a.metrics.RecordError("authenticate_user", "user_inactive")
        return nil, &AuthenticationError{Type: "user_inactive", Status: user.Status}
    }

    // 5. 获取用户权限
    permissions, err := a.userStore.GetUserPermissions(user.ID)
    if err != nil {
        a.metrics.RecordError("authenticate_user", "permission_lookup_failed")
        // 权限获取失败不阻断认证，但记录警告
        a.logger.Warn("Failed to get user permissions", zap.Error(err))
        permissions = []string{} // 默认空权限
    }

    // 6. 记录认证成功
    a.metrics.RecordSuccess("authenticate_user")
    a.auditLogger.LogAuthenticationSuccess(user.ID, req.ClientIP, req.UserAgent)

    return &AuthenticateUserOutput{
        UserID:      user.ID,
        UserInfo:    user.Info,
        Permissions: permissions,
        Metadata: map[string]interface{}{
            "authenticated_at": time.Now(),
            "token_issued_at": claims.IssuedAt,
            "token_expires_at": claims.ExpiresAt,
        },
    }, nil
}

func (a *AuthenticateUserActivity) ValidateInput(input interface{}) error {
    req, ok := input.(*AuthenticateUserInput)
    if !ok {
        return errors.New("input must be AuthenticateUserInput")
    }

    if req.Token == "" {
        return errors.New("token is required")
    }

    if len(req.Token) < 10 {
        return errors.New("token too short")
    }

    return nil
}
```

### AI推理活动：智能的核心

```go
// AI推理活动实现
type AIInferenceActivity struct {
    llmService   LLMService
    costTracker  *CostTracker
    cache        *InferenceCache
    rateLimiter  *RateLimiter
    metrics      *ActivityMetrics
}

type AIInferenceInput struct {
    Model       string                 `json:"model"`       // 模型名称
    Messages    []ChatMessage          `json:"messages"`    // 对话消息
    Parameters  InferenceParameters    `json:"parameters"`  // 推理参数
    UserID      string                 `json:"user_id"`      // 用户ID
    RequestID   string                 `json:"request_id"`   // 请求ID
}

type InferenceParameters struct {
    Temperature float64 `json:"temperature,omitempty"` // 温度参数
    MaxTokens   int     `json:"max_tokens,omitempty"`   // 最大token数
    TopP        float64 `json:"top_p,omitempty"`        // Top-p采样
    FrequencyPenalty float64 `json:"frequency_penalty,omitempty"`
    PresencePenalty  float64 `json:"presence_penalty,omitempty"`
}

func (a *AIInferenceActivity) Execute(ctx context.Context, input interface{}) (interface{}, error) {
    req, ok := input.(*AIInferenceInput)
    if !ok {
        return nil, errors.New("invalid input type")
    }

    // 1. 速率限制检查
    if !a.rateLimiter.Allow(req.UserID, 1) {
        return nil, &RateLimitError{RetryAfter: a.rateLimiter.GetRetryAfter(req.UserID)}
    }

    // 2. 缓存检查
    cacheKey := a.generateCacheKey(req)
    if cached, found := a.cache.Get(cacheKey); found {
        a.metrics.RecordCacheHit()
        return cached, nil
    }

    // 3. 成本预估和检查
    estimatedCost := a.estimateCost(req)
    if !a.costTracker.CheckBudget(req.UserID, estimatedCost) {
        return nil, &BudgetExceededError{EstimatedCost: estimatedCost}
    }

    // 4. 执行推理
    startTime := time.Now()
    result, err := a.llmService.Inference(ctx, &LLMRequest{
        Model:      req.Model,
        Messages:   req.Messages,
        Parameters: req.Parameters,
        UserID:     req.UserID,
        RequestID:  req.RequestID,
    })
    duration := time.Since(startTime)

    if err != nil {
        a.metrics.RecordInferenceError(req.Model, err)
        return nil, fmt.Errorf("inference failed: %w", err)
    }

    // 5. 记录实际成本
    actualCost := a.calculateActualCost(result, duration)
    a.costTracker.RecordUsage(req.UserID, actualCost)

    // 6. 缓存结果
    a.cache.Set(cacheKey, result, a.getCacheTTL(result))

    // 7. 记录指标
    a.metrics.RecordInferenceSuccess(req.Model, duration, actualCost)

    return result, nil
}
```

### 工具执行活动：能力的扩展

**这块代码展示了什么？**

这段代码展示了工具执行活动的实现，支持在安全沙箱中执行外部工具。背景是：AI代理需要调用各种工具（如网页搜索、数据库查询等），但工具执行存在安全风险，这个活动提供了标准化的工具调用和安全隔离机制。

这段代码的目的是说明如何在活动系统中安全地集成和执行外部工具。

```go
// 工具执行活动实现
type ToolExecutionActivity struct {
    toolRegistry *ToolRegistry
    sandbox      *WasiSandbox
    validator    *ToolInputValidator
    metrics      *ActivityMetrics
}

func (a *ToolExecutionActivity) Execute(ctx context.Context, input interface{}) (interface{}, error) {
    req, ok := input.(*ToolExecutionInput)
    if !ok {
        return nil, errors.New("invalid input type")
    }

    // 1. 获取工具定义
    tool, err := a.toolRegistry.GetTool(req.ToolName)
    if err != nil {
        return nil, &ToolNotFoundError{ToolName: req.ToolName}
    }

    // 2. 验证输入参数
    if err := a.validator.ValidateInput(tool, req.Parameters); err != nil {
        return nil, &ToolInputValidationError{Details: err.Error()}
    }

    // 3. 检查工具权限
    if err := a.checkToolPermissions(ctx, tool, req.UserID); err != nil {
        return nil, &ToolPermissionDeniedError{ToolName: req.ToolName}
    }

    // 4. 执行工具
    startTime := time.Now()
    result, err := a.executeTool(ctx, tool, req.Parameters)
    duration := time.Since(startTime)

    if err != nil {
        a.metrics.RecordToolExecutionError(req.ToolName, err)
        return nil, &ToolExecutionError{ToolName: req.ToolName, Cause: err}
    }

    // 5. 验证输出
    if err := a.validator.ValidateOutput(tool, result); err != nil {
        a.metrics.RecordToolOutputValidationError(req.ToolName)
        return nil, &ToolOutputValidationError{Details: err.Error()}
    }

    // 6. 记录执行指标
    a.metrics.RecordToolExecutionSuccess(req.ToolName, duration)

    return result, nil
}

func (a *ToolExecutionActivity) executeTool(ctx context.Context, tool *ToolDefinition, params map[string]interface{}) (interface{}, error) {
    switch tool.Type {
    case "wasm_module":
        return a.executeWasmTool(ctx, tool, params)
    case "http_api":
        return a.executeHTTPTool(ctx, tool, params)
    case "database_query":
        return a.executeDatabaseTool(ctx, tool, params)
    case "file_operation":
        return a.executeFileTool(ctx, tool, params)
    default:
        return nil, fmt.Errorf("unsupported tool type: %s", tool.Type)
    }
}

func (a *ToolExecutionActivity) executeWasmTool(ctx context.Context, tool *ToolDefinition, params map[string]interface{}) (interface{}, error) {
    // 1. 准备输入
    input := ToolInput{
        Parameters: params,
        Context: map[string]interface{}{
            "execution_id": ctx.Value("execution_id"),
            "user_id": ctx.Value("user_id"),
        },
    }

    inputJSON, err := json.Marshal(input)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal input: %w", err)
    }

    // 2. 执行WASM模块
    result, err := a.sandbox.ExecuteWasmWithTimeout(
        ctx,
        tool.WasmBytes,
        string(inputJSON),
        tool.ExecutionTimeout,
    )
    if err != nil {
        return nil, fmt.Errorf("WASM execution failed: %w", err)
    }

    // 3. 解析输出
    var output ToolOutput
    if err := json.Unmarshal([]byte(result), &output); err != nil {
        return nil, fmt.Errorf("failed to parse output: %w", err)
    }

    return output.Result, nil
}
```

## 第四章：活动系统的运维和监控

### 活动指标和性能监控

```go
// 活动指标收集器
type ActivityMetricsCollector struct {
    // 执行指标
    ExecutionCount   *prometheus.CounterVec   // 执行次数
    ExecutionDuration *prometheus.HistogramVec // 执行耗时
    ExecutionErrors  *prometheus.CounterVec   // 执行错误

    // 资源指标
    ResourceUsage    *prometheus.GaugeVec     // 资源使用
    QueueDepth       *prometheus.GaugeVec     // 队列深度

    // 性能指标
    Throughput       *prometheus.GaugeVec     // 吞吐量
    LatencyPercentiles *prometheus.HistogramVec // 延迟百分位

    // 错误指标
    ErrorRate        *prometheus.GaugeVec     // 错误率
    RetryCount       *prometheus.CounterVec   // 重试次数

    // 自定义业务指标
    BusinessMetrics  map[string]*prometheus.MetricVec
}

func (amc *ActivityMetricsCollector) RegisterMetrics() {
    // 注册标准指标
    prometheus.MustRegister(amc.ExecutionCount)
    prometheus.MustRegister(amc.ExecutionDuration)
    prometheus.MustRegister(amc.ExecutionErrors)

    // 初始化指标向量
    amc.ExecutionCount = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "activity_execution_total",
            Help: "Total number of activity executions",
        },
        []string{"activity_name", "status", "category"},
    )

    amc.ExecutionDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "activity_execution_duration_seconds",
            Help:    "Activity execution duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"activity_name", "category"},
    )
}

func (amc *ActivityMetricsCollector) RecordExecution(activityName, category string, duration time.Duration, err error) {
    // 记录执行计数
    status := "success"
    if err != nil {
        status = "error"
    }

    amc.ExecutionCount.WithLabelValues(activityName, status, category).Inc()

    // 记录执行耗时
    amc.ExecutionDuration.WithLabelValues(activityName, category).Observe(duration.Seconds())

    if err != nil {
        amc.ExecutionErrors.WithLabelValues(activityName, category, err.Error()).Inc()
    }
}
```

### 活动系统的故障排查

**这块代码展示了什么？**

这段代码展示了活动故障排查器的实现，自动分析活动失败的原因并提供修复建议。背景是：分布式系统中的故障排查很复杂，需要综合分析日志、指标、追踪数据等，这个排查器提供了智能的根本原因分析和自动化修复建议。

这段代码的目的是说明如何构建智能的故障排查系统，提高系统的运维效率。

```go
// 活动故障排查器
type ActivityTroubleshooter struct {
    // 日志分析器
    logAnalyzer *LogAnalyzer

    // 指标分析器
    metricsAnalyzer *MetricsAnalyzer

    // 执行追踪器
    executionTracer *ExecutionTracer

    // 根本原因分析器
    rcaAnalyzer *RootCauseAnalyzer
}

func (at *ActivityTroubleshooter) TroubleshootActivityFailure(activityName string, executionID string) (*TroubleshootingReport, error) {
    report := &TroubleshootingReport{
        ActivityName: activityName,
        ExecutionID:  executionID,
        Timestamp:    time.Now(),
    }

    // 1. 收集执行日志
    logs, err := at.logAnalyzer.GetExecutionLogs(executionID)
    if err != nil {
        return nil, fmt.Errorf("failed to get execution logs: %w", err)
    }
    report.ExecutionLogs = logs

    // 2. 分析指标数据
    metrics, err := at.metricsAnalyzer.GetExecutionMetrics(executionID)
    if err != nil {
        at.logger.Warn("Failed to get execution metrics", zap.Error(err))
    }
    report.MetricsData = metrics

    // 3. 执行追踪分析
    trace, err := at.executionTracer.GetExecutionTrace(executionID)
    if err != nil {
        at.logger.Warn("Failed to get execution trace", zap.Error(err))
    }
    report.ExecutionTrace = trace

    // 4. 根本原因分析
    rootCause := at.rcaAnalyzer.AnalyzeRootCause(logs, metrics, trace)
    report.RootCause = rootCause

    // 5. 生成修复建议
    recommendations := at.generateRecommendations(rootCause, logs, metrics)
    report.Recommendations = recommendations

    return report, nil
}

func (at *ActivityTroubleshooter) generateRecommendations(rootCause *RootCause, logs []LogEntry, metrics *ExecutionMetrics) []string {
    var recommendations []string

    switch rootCause.Category {
    case "resource_exhaustion":
        recommendations = append(recommendations,
            "Increase resource limits for the activity",
            "Implement resource pooling",
            "Add circuit breaker for resource protection",
        )

    case "dependency_failure":
        recommendations = append(recommendations,
            "Implement retry logic with exponential backoff",
            "Add fallback mechanisms",
            "Monitor dependency health",
        )

    case "input_validation":
        recommendations = append(recommendations,
            "Improve input validation logic",
            "Add comprehensive input sanitization",
            "Implement schema validation",
        )

    case "timeout":
        recommendations = append(recommendations,
            "Increase timeout values",
            "Optimize activity implementation",
            "Implement async processing for long-running tasks",
        )
    }

    return recommendations
}
```

## 第五章：活动系统的实践效果

### 量化收益分析

Shannon活动系统实施后的效果：

**开发效率提升**：
- **代码行数**：减少60%（从巨石函数到组合活动）
- **新功能开发时间**：从2周缩短到2天
- **单元测试覆盖率**：从40%提升到85%

**系统可靠性改善**：
- **平均故障间隔时间（MTBF）**：从2天提升到30天
- **故障恢复时间（MTTR）**：从4小时缩短到10分钟
- **服务可用性**：从99.5%提升到99.95%

**运维效率提升**：
- **问题诊断时间**：从2小时缩短到15分钟
- **部署频率**：从每周1次提升到每日多次
- **回滚成功率**：从80%提升到100%

**业务价值**：
- **用户请求处理能力**：提升300%
- **新活动上线速度**：从1个月缩短到1周
- **系统扩展性**：支持10倍并发量增长

### 关键成功因素

1. **标准化设计**：统一的活动接口和生命周期
2. **组合式架构**：通过活动组合实现复杂功能
3. **可观测性优先**：完善的监控和故障排查体系
4. **渐进式演进**：从单体到微服务，再到活动系统的平滑迁移

### 未来展望

随着AI系统的复杂度提升，活动系统将面临新挑战：

1. **智能活动调度**：基于AI的活动执行顺序优化
2. **自适应资源管理**：根据负载自动调整资源分配
3. **跨云活动执行**：支持多云环境的活动调度
4. **实时活动协作**：活动间的实时通信和协作

活动系统证明了：**复杂系统的可管理性不是来自简单的设计，而是来自精心设计的抽象和组合机制**。

## 活动系统的深度架构设计

Shannon的活动系统不仅仅是简单的函数调用，而是一个完整的**分布式执行引擎**。让我们从架构设计开始深入剖析。

#### 活动管理器的核心架构

```go
// go/orchestrator/internal/activities/manager.go

/// 活动管理器配置
type ManagerConfig struct {
    // 执行配置
    MaxConcurrency          int           `yaml:"max_concurrency"`          // 最大并发活动数
    DefaultTimeout          time.Duration `yaml:"default_timeout"`          // 默认活动超时
    MaxRetries              int           `yaml:"max_retries"`              // 最大重试次数

    // 重试配置
    InitialRetryDelay       time.Duration `yaml:"initial_retry_delay"`      // 初始重试延迟
    MaxRetryDelay           time.Duration `yaml:"max_retry_delay"`          // 最大重试延迟
    RetryBackoffCoefficient float64       `yaml:"retry_backoff_coefficient"` // 重试退避系数

    // 资源限制
    MaxMemoryUsage          int64         `yaml:"max_memory_usage"`         // 最大内存使用
    MaxCPUTime              time.Duration `yaml:"max_cpu_time"`             // 最大CPU时间

    // 监控配置
    MetricsEnabled          bool          `yaml:"metrics_enabled"`         // 启用指标收集
    TracingEnabled          bool          `yaml:"tracing_enabled"`         // 启用分布式追踪

    // 错误处理配置
    DeadLetterEnabled       bool          `yaml:"dead_letter_enabled"`      // 启用死信队列
    DeadLetterQueueSize     int           `yaml:"dead_letter_queue_size"`   // 死信队列大小
}

/// 活动管理器主结构体
type Manager struct {
    // Temporal客户端
    temporalClient *temporal.Client

    // 活动注册表
    activityRegistry *ActivityRegistry

    // 执行引擎
    executionEngine *ExecutionEngine

    // 监控组件
    metrics         *ActivityMetrics
    tracer          trace.Tracer

    // 并发控制
    semaphore       chan struct{}        // 并发限制信号量
    workerPool      *WorkerPool          // 工作池

    // 错误处理
    errorHandler    *ErrorHandler
    deadLetterQueue *DeadLetterQueue     // 死信队列

    // 配置
    config          ManagerConfig

    // 日志
    logger          *zap.Logger

    // 启动时间
    startTime       time.Time
}

/// 活动注册表
type ActivityRegistry struct {
    activities map[string]ActivityDefinition
    mu         sync.RWMutex
}

/// 活动定义
type ActivityDefinition struct {
    Name        string
    Handler     ActivityHandler
    Options     ActivityOptions
    Metadata    ActivityMetadata
}

/// 活动处理器接口
type ActivityHandler interface {
    Execute(ctx context.Context, input interface{}) (interface{}, error)
    ValidateInput(input interface{}) error
    GetMetadata() ActivityMetadata
}

/// 活动元数据
type ActivityMetadata struct {
    Name          string            `json:"name"`
    Description   string            `json:"description"`
    Version       string            `json:"version"`
    Category      string            `json:"category"`      // agent, memory, tool, etc.
    Timeout       time.Duration     `json:"timeout"`
    MaxRetries    int               `json:"max_retries"`
    RequiredPermissions []string    `json:"required_permissions"`
    ResourceRequirements ResourceReq `json:"resource_requirements"`
}
```

**架构设计的核心权衡**：

1. **并发控制机制**：
   ```go
   // 信号量限制总并发数
   // 防止系统过载
   // 保证服务质量
   semaphore := make(chan struct{}, maxConcurrency)
   ```

2. **注册表模式**：
   ```go
   // 运行时注册活动
   // 支持热更新和版本控制
   // 插件化架构
   activityRegistry := &ActivityRegistry{
       activities: make(map[string]ActivityDefinition),
   }
   ```

3. **错误处理策略**：
   ```go
   // 分层错误处理
   // 自动重试和降级
   // 死信队列防止消息丢失
   deadLetterQueue: &DeadLetterQueue{}
   ```

#### 活动执行引擎的实现

**这块代码展示了什么？**

这段代码展示了活动执行引擎的核心实现，负责活动的调度、执行、监控和错误处理。背景是：活动系统需要可靠地执行各种类型的活动，包括预检、资源分配、重试机制等，这个引擎提供了完整的执行生命周期管理。

这段代码的目的是说明如何构建一个高可靠的活动执行引擎，支持复杂的执行逻辑和错误恢复。

```go
// go/orchestrator/internal/activities/execution_engine.go

/// 活动执行引擎
type ExecutionEngine struct {
    manager *Manager
    metrics *ExecutionMetrics
    logger  *zap.Logger
}

**这块代码展示了什么？**

这段代码展示了活动执行引擎的主入口函数，实现了完整的活动执行生命周期。背景是：活动执行需要经过多个阶段（验证、权限检查、资源分配、执行、重试等），这个函数提供了标准化的执行流程，确保所有活动都遵循相同的治理规则。

这段代码的目的是说明如何构建一个健壮的活动执行引擎，支持复杂的业务逻辑和错误处理。

```go
/// 执行活动的主入口
func (ee *ExecutionEngine) ExecuteActivity(
    ctx context.Context,
    activityName string,
    input interface{},
) (interface{}, error) {

    executionID := ee.generateExecutionID()
    startTime := time.Now()

    ee.logger.Info("Starting activity execution",
        zap.String("execution_id", executionID),
        zap.String("activity_name", activityName))

    // 1. 获取活动定义
    activityDef, err := ee.manager.activityRegistry.Get(activityName)
    if err != nil {
        ee.metrics.RecordExecutionError("activity_not_found")
        return nil, fmt.Errorf("activity not found: %s", activityName)
    }

    // 2. 验证输入
    if err := activityDef.Handler.ValidateInput(input); err != nil {
        ee.metrics.RecordExecutionError("input_validation_failed")
        return nil, fmt.Errorf("input validation failed: %w", err)
    }

    // 3. 权限检查
    if err := ee.checkPermissions(ctx, activityDef.Metadata.RequiredPermissions); err != nil {
        ee.metrics.RecordExecutionError("permission_denied")
        return nil, fmt.Errorf("permission denied: %w", err)
    }

    // 4. 资源预检
    if err := ee.checkResourceAvailability(activityDef.Metadata.ResourceRequirements); err != nil {
        ee.metrics.RecordExecutionError("resource_unavailable")
        return nil, fmt.Errorf("resource unavailable: %w", err)
    }

    // 5. 创建执行上下文
    execCtx, cancel := ee.createExecutionContext(ctx, activityDef, executionID)
    defer cancel()

    // 6. 执行活动（带重试）
    result, err := ee.executeWithRetry(execCtx, activityDef, input)

    // 7. 记录执行指标
    executionTime := time.Since(startTime)
    ee.metrics.RecordExecution(activityName, executionTime, err == nil)

    if err != nil {
        ee.logger.Error("Activity execution failed",
            zap.String("execution_id", executionID),
            zap.String("activity_name", activityName),
            zap.Error(err))

        // 8. 发送到死信队列（如果启用）
        if ee.manager.config.DeadLetterEnabled {
            ee.manager.deadLetterQueue.Enqueue(&FailedActivity{
                ExecutionID:   executionID,
                ActivityName:  activityName,
                Input:         input,
                Error:         err,
                FailedAt:      time.Now(),
                RetryCount:    ee.getRetryCountFromContext(execCtx),
            })
        }

        return nil, err
    }

    ee.logger.Info("Activity execution completed",
        zap.String("execution_id", executionID),
        zap.String("activity_name", activityName),
        zap.Duration("execution_time", executionTime))

    return result, nil
}

/// 执行活动（带重试机制）
func (ee *ExecutionEngine) executeWithRetry(
    ctx context.Context,
    activityDef *ActivityDefinition,
    input interface{},
) (interface{}, error) {

    var lastErr error
    maxRetries := activityDef.Options.MaxRetries

    for attempt := 0; attempt <= maxRetries; attempt++ {
        // 设置重试上下文
        attemptCtx := context.WithValue(ctx, "retry_attempt", attempt)

        // 执行活动
        result, err := ee.executeSingleAttempt(attemptCtx, activityDef, input)

        if err == nil {
            // 执行成功
            return result, nil
        }

        lastErr = err

        // 检查是否可以重试
        if attempt >= maxRetries || !ee.isRetryableError(err) {
            break
        }

        // 计算重试延迟
        retryDelay := ee.calculateRetryDelay(attempt, activityDef.Options)

        ee.logger.Warn("Activity execution failed, retrying",
            zap.String("activity_name", activityDef.Name),
            zap.Int("attempt", attempt+1),
            zap.Duration("retry_delay", retryDelay),
            zap.Error(err))

        // 等待重试
        select {
        case <-time.After(retryDelay):
            // 继续重试
        case <-ctx.Done():
            // 上下文取消
            return nil, ctx.Err()
        }
    }

    return nil, lastErr
}

/// 执行单次尝试
func (ee *ExecutionEngine) executeSingleAttempt(
    ctx context.Context,
    activityDef *ActivityDefinition,
    input interface{},
) (interface{}, error) {

    // 1. 创建活动特定的上下文
    activityCtx, cancel := context.WithTimeout(ctx, activityDef.Options.Timeout)
    defer cancel()

    // 2. 添加监控和追踪
    span, activityCtx := ee.manager.tracer.Start(activityCtx, activityDef.Name)
    defer span.End()

    span.SetAttributes(
        attribute.String("activity.name", activityDef.Name),
        attribute.String("activity.version", activityDef.Metadata.Version),
        attribute.String("activity.category", activityDef.Metadata.Category),
    )

    // 3. 执行活动处理器
    result, err := activityDef.Handler.Execute(activityCtx, input)

    // 4. 记录执行结果
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
    } else {
        span.SetStatus(codes.Ok, "")
    }

    return result, err
}

/// 计算重试延迟
func (ee *ExecutionEngine) calculateRetryDelay(
    attempt int,
    options ActivityOptions,
) time.Duration {

    // 指数退避：delay = initial_delay * (backoff_coefficient ^ attempt)
    delay := float64(options.InitialRetryDelay) *
             math.Pow(options.RetryBackoffCoefficient, float64(attempt))

    // 应用最大延迟限制
    if delay > float64(options.MaxRetryDelay) {
        delay = float64(options.MaxRetryDelay)
    }

    // 添加随机抖动（±25%）
    jitter := delay * 0.25 * (rand.Float64()*2 - 1)
    delay += jitter

    return time.Duration(delay)
}

/// 检查错误是否可重试
func (ee *ExecutionEngine) isRetryableError(err error) bool {
    // 网络错误可重试
    if _, ok := err.(net.Error); ok {
        return true
    }

    // 超时错误可重试
    if err == context.DeadlineExceeded {
        return true
    }

    // 5xx HTTP错误可重试
    if httpErr, ok := err.(*url.Error); ok {
        if resp, respOk := httpErr.Err.(*http.Response); respOk {
            return resp.StatusCode >= 500
        }
    }

    // 其他错误默认不可重试
    return false
}
```

**执行引擎的核心机制**：

1. **重试策略**：
   ```go
   // 指数退避重试
   // 带抖动的随机延迟
   // 最大重试次数限制
   // 可重试错误识别
   ```

2. **资源管理**：
   ```go
   // 超时控制
   // 内存限制
   // CPU时间限制
   // 并发度控制
   ```

3. **错误处理**：
   ```go
   // 分层错误分类
   // 死信队列处理
   // 详细错误追踪
   // 优雅降级
   ```

#### Agent执行活动的深度实现

```go
// go/orchestrator/internal/activities/agent_execution.go

/// Agent执行活动的配置
type AgentExecutionConfig struct {
    // 执行模式
    ExecutionMode      ExecutionMode     `yaml:"execution_mode"`      // standard/direct
    ForceToolExecution bool              `yaml:"force_tool_execution"` // 强制工具执行

    // LLM配置
    Model              string            `yaml:"model"`               // 默认模型
    Temperature        float64           `yaml:"temperature"`         // 温度参数
    MaxTokens          int               `yaml:"max_tokens"`          // 最大令牌数

    // 工具配置
    EnableToolUse      bool              `yaml:"enable_tool_use"`     // 启用工具使用
    MaxToolCalls       int               `yaml:"max_tool_calls"`      // 最大工具调用数
    ToolTimeout        time.Duration     `yaml:"tool_timeout"`        // 工具超时

    // 安全配置
    SandboxEnabled     bool              `yaml:"sandbox_enabled"`    // 启用沙箱
    ContentFiltering   bool              `yaml:"content_filtering"`  // 内容过滤

    // 性能配置
    StreamingEnabled   bool              `yaml:"streaming_enabled"`  // 启用流式响应
    CachingEnabled     bool              `yaml:"caching_enabled"`    // 启用缓存
}

/// Agent执行活动的主结构体
type AgentExecutionActivity struct {
    // 核心客户端
    agentCoreClient *agent.Client       // Rust Agent Core客户端
    llmService      *llm.Client         // LLM服务客户端

    // 工具管理
    toolRegistry    *tools.Registry     // 工具注册表
    toolExecutor    *tools.Executor     // 工具执行器

    // 会话管理
    sessionManager  *session.Manager    // 会话管理器

    // 配置
    config          AgentExecutionConfig

    // 监控
    metrics         *AgentMetrics
    tracer          trace.Tracer
    logger          *zap.Logger
}

/// 执行Agent的主要逻辑
func (aea *AgentExecutionActivity) Execute(
    ctx context.Context,
    input *AgentExecutionInput,
) (*AgentExecutionResult, error) {

    executionID := generateExecutionID()
    startTime := time.Now()

    aea.logger.Info("Starting agent execution",
        zap.String("execution_id", executionID),
        zap.String("task_id", input.TaskID),
        zap.String("session_id", input.SessionID))

    // 1. 准备执行上下文
    execCtx, err := aea.prepareExecutionContext(ctx, input, executionID)
    if err != nil {
        return nil, fmt.Errorf("failed to prepare execution context: %w", err)
    }

    // 2. 选择执行策略
    strategy := aea.selectExecutionStrategy(input)

    // 3. 执行Agent推理
    result, err := aea.executeAgent(execCtx, input, strategy)
    if err != nil {
        aea.metrics.RecordExecutionError("agent_execution_failed")
        return nil, fmt.Errorf("agent execution failed: %w", err)
    }

    // 4. 处理工具调用（如果有）
    if len(result.ToolCalls) > 0 {
        toolResults, err := aea.executeTools(execCtx, result.ToolCalls, input)
        if err != nil {
            aea.logger.Warn("Tool execution failed", zap.Error(err))
            // 工具失败不影响主流程，但记录警告
        } else {
            result.ToolResults = toolResults
        }
    }

    // 5. 后处理和验证
    finalResult, err := aea.postProcessResult(execCtx, result, input)
    if err != nil {
        return nil, fmt.Errorf("result post-processing failed: %w", err)
    }

    // 6. 更新会话状态
    if err := aea.updateSessionState(execCtx, input, finalResult); err != nil {
        aea.logger.Warn("Session state update failed", zap.Error(err))
        // 会话更新失败不影响结果返回
    }

    // 7. 记录执行指标
    executionTime := time.Since(startTime)
    aea.metrics.RecordExecution(
        input.Model,
        len(result.ToolCalls),
        executionTime,
        finalResult.TokenUsage,
    )

    finalResult.ExecutionID = executionID
    finalResult.ExecutedAt = time.Now()

    return finalResult, nil
}

/// 准备执行上下文
func (aea *AgentExecutionActivity) prepareExecutionContext(
    ctx context.Context,
    input *AgentExecutionInput,
    executionID string,
) (*AgentExecutionContext, error) {

    // 1. 获取会话上下文
    session, err := aea.sessionManager.Get(ctx, input.SessionID)
    if err != nil {
        return nil, fmt.Errorf("failed to get session: %w", err)
    }

    // 2. 构建系统提示
    systemPrompt, err := aea.buildSystemPrompt(input, session)
    if err != nil {
        return nil, fmt.Errorf("failed to build system prompt: %w", err)
    }

    // 3. 准备工具列表
    availableTools, err := aea.prepareTools(input)
    if err != nil {
        return nil, fmt.Errorf("failed to prepare tools: %w", err)
    }

    // 4. 设置执行约束
    constraints := ExecutionConstraints{
        MaxTokens:       input.MaxTokens,
        Timeout:         input.Timeout,
        Temperature:     input.Temperature,
        ForceToolUse:    input.ForceToolUse,
        StreamingEnabled: aea.config.StreamingEnabled,
    }

    return &AgentExecutionContext{
        ExecutionID:     executionID,
        Session:         session,
        SystemPrompt:    systemPrompt,
        AvailableTools:  availableTools,
        Constraints:     constraints,
        UserID:          input.UserID,
        TenantID:        input.TenantID,
    }, nil
}

/// 选择执行策略
func (aea *AgentExecutionActivity) selectExecutionStrategy(
    input *AgentExecutionInput,
) ExecutionStrategy {

    // 1. 检查强制策略
    if input.ForceDirectExecution {
        return ExecutionStrategyDirect
    }

    if input.ForceAgentCoreExecution {
        return ExecutionStrategyAgentCore
    }

    // 2. 基于输入特征选择策略
    if aea.shouldUseAgentCore(input) {
        return ExecutionStrategyAgentCore
    }

    // 3. 默认使用直接执行
    return ExecutionStrategyDirect
}

/// 判断是否应该使用Agent Core
func (aea *AgentExecutionActivity) shouldUseAgentCore(input *AgentExecutionInput) bool {
    // 复杂工具调用场景
    if len(input.SuggestedTools) > 2 {
        return true
    }

    // 需要代码执行的场景
    for _, tool := range input.SuggestedTools {
        if strings.Contains(tool, "code") || strings.Contains(tool, "execute") {
            return true
        }
    }

    // 长会话上下文
    if len(input.SessionCtx) > 10000 { // 10KB
        return true
    }

    // 复杂推理任务
    if input.TaskComplexity > 0.7 {
        return true
    }

    return false
}

/// 执行Agent推理
func (aea *AgentExecutionActivity) executeAgent(
    ctx *AgentExecutionContext,
    input *AgentExecutionInput,
    strategy ExecutionStrategy,
) (*AgentExecutionResult, error) {

    switch strategy {
    case ExecutionStrategyAgentCore:
        return aea.executeViaAgentCore(ctx, input)
    case ExecutionStrategyDirect:
        return aea.executeDirect(ctx, input)
    default:
        return nil, fmt.Errorf("unknown execution strategy: %v", strategy)
    }
}

/// 通过Agent Core执行
func (aea *AgentExecutionActivity) executeViaAgentCore(
    ctx *AgentExecutionContext,
    input *AgentExecutionInput,
) (*AgentExecutionResult, error) {

    // 构建Agent Core请求
    request := &agent.ExecuteTaskRequest{
        TaskID:       input.TaskID,
        UserPrompt:   input.UserPrompt,
        SystemPrompt: ctx.SystemPrompt,
        SessionID:    input.SessionID,
        Tools:        ctx.AvailableTools,
        Context:      input.SessionCtx,
        Constraints:  ctx.Constraints,
    }

    // 调用Agent Core
    response, err := aea.agentCoreClient.ExecuteTask(ctx, request)
    if err != nil {
        return nil, fmt.Errorf("agent core execution failed: %w", err)
    }

    // 转换结果格式
    return aea.convertAgentCoreResult(response), nil
}

/// 直接执行（通过LLM服务）
func (aea *AgentExecutionActivity) executeDirect(
    ctx *AgentExecutionContext,
    input *AgentExecutionInput,
) (*AgentExecutionResult, error) {

    // 构建LLM请求
    messages := []llm.Message{
        {Role: "system", Content: ctx.SystemPrompt},
        {Role: "user", Content: input.UserPrompt},
    }

    // 添加会话历史
    if len(input.History) > 0 {
        messages = append(messages, input.History...)
    }

    request := &llm.CompletionRequest{
        Model:       input.Model,
        Messages:    messages,
        Temperature: input.Temperature,
        MaxTokens:   input.MaxTokens,
        Tools:       ctx.AvailableTools,
        Stream:      ctx.Constraints.StreamingEnabled,
    }

    // 调用LLM服务
    response, err := aea.llmService.Complete(ctx, request)
    if err != nil {
        return nil, fmt.Errorf("LLM completion failed: %w", err)
    }

    // 解析工具调用
    toolCalls, err := aea.parseToolCalls(response.Content)
    if err != nil {
        aea.logger.Warn("Failed to parse tool calls", zap.Error(err))
    }

    return &AgentExecutionResult{
        Response:      response.Content,
        ToolCalls:     toolCalls,
        TokenUsage:    response.TokenUsage,
        ModelUsed:     response.Model,
        FinishReason:  response.FinishReason,
    }, nil
}

/// 执行工具调用
func (aea *AgentExecutionActivity) executeTools(
    ctx *AgentExecutionContext,
    toolCalls []ToolCall,
    input *AgentExecutionInput,
) ([]ToolResult, error) {

    var results []ToolResult

    // 限制并发工具执行
    semaphore := make(chan struct{}, 3) // 最多3个并发工具

    var wg sync.WaitGroup
    var mu sync.Mutex
    var errors []error

    for _, toolCall := range toolCalls {
        wg.Add(1)
        go func(tc ToolCall) {
            defer wg.Done()

            // 获取信号量
            semaphore <- struct{}{}
            defer func() { <-semaphore }()

            // 执行工具
            result, err := aea.executeSingleTool(ctx, tc, input)
            if err != nil {
                mu.Lock()
                errors = append(errors, err)
                mu.Unlock()
                return
            }

            mu.Lock()
            results = append(results, *result)
            mu.Unlock()
        }(toolCall)
    }

    wg.Wait()

    // 如果有错误，返回第一个错误
    if len(errors) > 0 {
        return results, errors[0]
    }

    return results, nil
}

/// 执行单个工具
func (aea *AgentExecutionActivity) executeSingleTool(
    ctx *AgentExecutionContext,
    toolCall ToolCall,
    input *AgentExecutionInput,
) (*ToolResult, error) {

    // 1. 获取工具定义
    tool, exists := aea.toolRegistry.GetTool(toolCall.Name)
    if !exists {
        return nil, fmt.Errorf("tool not found: %s", toolCall.Name)
    }

    // 2. 验证工具权限
    if err := aea.checkToolPermissions(ctx, tool, input); err != nil {
        return nil, fmt.Errorf("tool permission denied: %w", err)
    }

    // 3. 创建工具执行上下文
    toolCtx, cancel := context.WithTimeout(ctx, aea.config.ToolTimeout)
    defer cancel()

    // 4. 执行工具
    result, err := aea.toolExecutor.Execute(toolCtx, tool, toolCall.Parameters)
    if err != nil {
        return nil, fmt.Errorf("tool execution failed: %w", err)
    }

    return &ToolResult{
        ToolCallID:  toolCall.ID,
        ToolName:    toolCall.Name,
        Result:      result,
        ExecutedAt:  time.Now(),
        ExecutionTime: time.Since(toolCtx.(time.Time)), // 需要改进
    }, nil
}

/// 构建系统提示
func (aea *AgentExecutionActivity) buildSystemPrompt(
    input *AgentExecutionInput,
    session *session.Session,
) (string, error) {

    var prompt strings.Builder

    // 基础系统提示
    prompt.WriteString("You are an AI assistant with access to various tools. ")

    // 添加工具描述
    if len(input.SuggestedTools) > 0 {
        prompt.WriteString("You have access to the following tools:\n")
        for _, toolName := range input.SuggestedTools {
            if tool, exists := aea.toolRegistry.GetTool(toolName); exists {
                prompt.WriteString(fmt.Sprintf("- %s: %s\n",
                    tool.Name, tool.Description))
            }
        }
    }

    // 添加会话特定的指令
    if session != nil && len(session.Metadata) > 0 {
        if personality, ok := session.Metadata["personality"].(string); ok {
            prompt.WriteString(fmt.Sprintf("Your personality: %s. ", personality))
        }
    }

    // 添加安全指令
    prompt.WriteString("Always prioritize user safety and data privacy. ")
    prompt.WriteString("Do not execute dangerous operations. ")

    return prompt.String(), nil
}
```

**Agent执行活动的双重架构**：

1. **Agent Core路径**：
   ```go
   // 高性能Rust实现
   // WASI沙箱安全执行
   // 复杂工具调用支持
   // 适合生产环境
   ```

2. **直接执行路径**：
   ```go
   // 轻量级Python实现
   // LLM服务直接调用
   // 简单任务快速响应
   // 开发调试友好
   ```

3. **工具执行系统**：
   ```go
   // 并发工具执行
   // 权限验证
   // 超时控制
   // 结果聚合
   ```

这个活动系统为Shannon提供了强大的执行能力和可靠性保证。

## Agent执行活动：智能代理的核心

### 双重执行路径

Shannon的Agent执行活动支持两种模式：

```go
// go/orchestrator/internal/activities/agent.go

// 标准执行：通过Rust Agent Core
func ExecuteAgent(ctx context.Context, input AgentExecutionInput) (AgentExecutionResult, error) {
	logger := activity.GetLogger(ctx)
	return executeAgentCore(ctx, input, logger)
}

// 强制工具执行：直接调用LLM服务
func ExecuteAgentWithForcedTools(ctx context.Context, input AgentExecutionInput) (AgentExecutionResult, error) {
	// 绕过Agent Core，直接调用Python LLM服务
	return executeForcedTools(ctx, input)
}
```

### 工具选择和执行流程

Agent执行的核心逻辑包含复杂的工具管理：

```go
// 工具选择策略
func executeAgentCore(ctx context.Context, input AgentExecutionInput, logger *zap.Logger) (AgentExecutionResult, error) {
	// 1. 工具过滤和验证
	allowedByRole := filterToolsByRole(input)

	// 2. 智能工具选择
	selectedToolCalls := selectToolsForQuery(ctx, input.Query, allowedByRole, logger, input.ParentWorkflowID)

	// 3. 工具调用执行
	result := executeWithToolCalls(ctx, input, selectedToolCalls)

	// 4. 结果合成
	finalResult := synthesizeResults(result)

	return finalResult, nil
}
```

### 工具选择引擎

工具选择是Agent执行的关键环节：

```go
// 工具选择逻辑
func selectToolsForQuery(ctx context.Context, query string, availableTools []string, logger *zap.Logger, parentWorkflowID string) []map[string]interface{} {
	// 1. 构建选择请求
	payload := map[string]interface{}{
		"task":              query,
		"context":           map[string]interface{}{},
		"exclude_dangerous": true,
		"max_tools":         3,  // 限制工具数量
	}

	// 2. 调用Python LLM服务的工具选择API
	response := callToolSelectionAPI(ctx, payload)

	// 3. 解析和过滤结果
	selectedTools := parseToolSelectionResponse(response, availableTools)

	return selectedTools
}
```

这个选择过程体现了：
- **智能决策**：基于查询内容选择最合适的工具
- **安全过滤**：自动排除危险工具
- **数量限制**：防止过度使用工具
- **上下文感知**：考虑查询的复杂度和类型

## 流式执行：实时响应系统

### 流式vs同步执行

Shannon支持两种执行模式：

```go
// 流式执行：实时返回结果
func runStreaming(ctx context.Context, input AgentExecutionInput) (AgentExecutionResult, error) {
	stream, err := client.StreamExecuteTask(ctx, req)
	if err != nil {
		return AgentExecutionResult{}, err
	}

	var result AgentExecutionResult
	for {
		update, err := stream.Recv()
		if err == io.EOF {
			break
		}

		// 实时处理更新
		processStreamingUpdate(update, &result)

		// 发布事件
		publishStreamingEvent(update)
	}

	return result, nil
}

// 同步执行：一次性返回结果
func runUnary(ctx context.Context, input AgentExecutionInput) (AgentExecutionResult, error) {
	resp, err := client.ExecuteTask(ctx, req)
	return processUnaryResponse(resp), err
}
```

### 流式事件处理

流式执行提供了丰富的实时反馈：

```go
// 流式更新处理
func processStreamingUpdate(update *agentpb.ExecuteTaskResponse, result *AgentExecutionResult) {
	switch u := update.Update.(type) {
	case *agentpb.ExecuteTaskResponse_Delta:
		// 文本增量
		result.Response += u.Delta
		publishPartialResult(u.Delta)

	case *agentpb.ExecuteTaskResponse_ToolResult:
		// 工具执行结果
		toolResult := processToolResult(u.ToolResult)
		result.ToolExecutions = append(result.ToolExecutions, toolResult)
		publishToolEvent(toolResult)

	case *agentpb.ExecuteTaskResponse_Metrics:
		// 性能指标
		updateMetrics(u.Metrics, result)
	}
}
```

流式执行的优势：
- **实时反馈**：用户可以看到思考过程
- **渐进式响应**：边思考边输出
- **错误早期发现**：问题出现时立即处理
- **用户体验优化**：减少等待焦虑

## 上下文管理和数据验证

### 上下文注入和验证

Agent执行前要进行严格的上下文处理：

```go
// 上下文验证和清理
func validateContext(ctx map[string]interface{}, logger *zap.Logger) map[string]interface{} {
	validated := make(map[string]interface{})

	// 过滤内部元数据字段
	for key, value := range ctx {
		if isInternalField(key) {
			continue // 跳过内部字段
		}

		// 验证和清理值
		if cleanedValue := sanitizeContextValue(value, key, logger); cleanedValue != nil {
			validated[key] = cleanedValue
		}
	}

	return validated
}

// 递归清理上下文值
func sanitizeContextValue(value interface{}, key string, logger *zap.Logger) interface{} {
	switch v := value.(type) {
	case string:
		// 限制字符串长度
		runes := []rune(v)
		if len(runes) > 10000 {
			return string(runes[:10000])
		}
		return v

	case map[string]interface{}:
		// 递归清理嵌套对象
		sanitized := make(map[string]interface{})
		for k, nested := range v {
			if cleaned := sanitizeContextValue(nested, k, logger); cleaned != nil {
				sanitized[k] = cleaned
			}
		}
		return sanitized

	default:
		// 类型检查和清理
		return sanitizeByType(v, logger)
	}
}
```

### 会话上下文集成

Agent执行深度集成了会话管理：

```go
// 会话上下文构建
func buildSessionContext(input AgentExecutionInput) *agentpb.SessionContext {
	if input.SessionID == "" && len(input.History) == 0 {
		return nil
	}

	return &agentpb.SessionContext{
		SessionId: input.SessionID,
		History:   input.History, // 对话历史
		// Context字段包含持久化会话状态
	}
}
```

这个集成确保了：
- **连续性**：Agent记住之前的交互
- **个性化**：基于历史调整行为
- **效率**：避免重复处理
- **一致性**：所有Agent共享会话状态

## 工具执行和结果处理

### 工具调用序列化

工具调用需要精确的序列化处理：

```go
// 工具调用准备
func prepareToolCalls(selectedToolCalls []map[string]interface{}) ([]*structpb.Value, error) {
	values := make([]*structpb.Value, 0, len(selectedToolCalls))

	for _, call := range selectedToolCalls {
		// 验证工具调用结构
		sanitizedCall := sanitizeToolCall(call, logger)
		if sanitizedCall == nil {
			continue
		}

		// 转换为Protobuf格式
		callStruct, err := structpb.NewStruct(sanitizedCall)
		if err != nil {
			return nil, err
		}

		values = append(values, structpb.NewStructValue(callStruct))
	}

	return values, nil
}
```

### 工具结果处理

工具执行结果需要详细处理：

```go
// 工具结果解析
func processToolExecutionResult(tr *agentpb.ToolResult, result *AgentExecutionResult) {
	toolName := tr.GetToolId()
	output := interface{}(nil)
	if tr.Output != nil {
		output = tr.Output.AsInterface()
	}

	// 创建工具执行记录
	toolExecution := ToolExecution{
		Tool:    toolName,
		Success: tr.Status == commonpb.StatusCode_STATUS_CODE_OK,
		Output:  output,
		Error:   tr.ErrorMessage,
	}

	result.ToolExecutions = append(result.ToolExecutions, toolExecution)

	// 更新工具使用统计
	updateToolUsageStats(toolName, toolExecution.Success)
}
```

## 事件流和实时监控

### 结构化事件发射

活动执行过程中会发射丰富的事件：

```go
// 事件发射系统
const (
	StreamEventAgentThinking  = "AGENT_THINKING"
	StreamEventToolInvoked    = "TOOL_INVOKED"
	StreamEventToolObs        = "TOOL_OBSERVATION"
	StreamEventLLMPartial     = "LLM_PARTIAL"
	StreamEventLLMOutput      = "LLM_OUTPUT"
)

// 事件发射函数
func emitAgentThinkingEvent(ctx context.Context, input AgentExecutionInput) {
	wfID := determineWorkflowID(input)

	streaming.Get().Publish(wfID, streaming.Event{
		WorkflowID: wfID,
		Type:       string(StreamEventAgentThinking),
		AgentID:    input.AgentID,
		Message:    fmt.Sprintf("Thinking: %s", truncateQuery(input.Query, 100)),
		Timestamp:  time.Now(),
	})
}
```

### 性能监控和指标

活动系统提供了全面的性能监控：

```go
// 活动性能指标
func recordActivityMetrics(activityName string, duration time.Duration, success bool) {
	// 记录执行时间
	metrics.ActivityDuration.WithLabelValues(activityName).Observe(duration.Seconds())

	// 记录成功率
	if success {
		metrics.ActivitySuccess.WithLabelValues(activityName).Inc()
	} else {
		metrics.ActivityFailure.WithLabelValues(activityName).Inc()
	}

	// 记录并发数
	metrics.ActiveActivities.WithLabelValues(activityName).Set(getActiveCount(activityName))
}
```

## 错误处理和恢复机制

### 多层错误处理

活动系统实现了多层错误处理：

```go
// 活动错误处理
func handleActivityError(err error, activityName string, input interface{}) error {
	// 1. 记录错误详情
	logger.Error("Activity execution failed",
		zap.String("activity", activityName),
		zap.Error(err),
		zap.Any("input", input))

	// 2. 分类错误类型
	switch err.(type) {
	case *temporal.TimeoutError:
		// 超时错误：增加重试间隔
		return wrapTimeoutError(err)

	case *temporal.ApplicationError:
		// 应用错误：检查是否可重试
		if isRetryableApplicationError(err) {
			return err // Temporal会自动重试
		}
		return wrapNonRetryableError(err)

	default:
		// 未知错误：记录并重试
		return wrapUnknownError(err)
	}
}
```

### 熔断器集成

活动系统集成了熔断器保护：

```go
// gRPC调用保护
connWrapper := circuitbreaker.NewGRPCConnectionWrapper(addr, "agent-core", logger)
conn, err := connWrapper.DialContext(ctx, grpc.WithTransportCredentials(insecure.NewCredentials()))
if err != nil {
	// 熔断器激活时会快速失败
	return AgentExecutionResult{}, err
}
```

## 安全和资源管理

### 输入验证和清理

所有活动输入都要经过严格验证：

```go
// 输入验证
func validateAgentExecutionInput(input *AgentExecutionInput) error {
	// 1. 基本字段验证
	if input.AgentID == "" {
		return errors.New("agent_id is required")
	}

	if input.Query == "" {
		return errors.New("query is required")
	}

	// 2. 工具参数验证
	if input.ToolParameters != nil {
		if err := validateToolParameters(input.ToolParameters); err != nil {
			return fmt.Errorf("invalid tool parameters: %w", err)
		}
	}

	// 3. 上下文大小限制
	if len(input.Context) > 100 {
		return errors.New("context too large")
	}

	return nil
}
```

### 资源限制和隔离

活动执行有严格的资源限制：

```go
// 资源限制配置
agentConfig := &agentpb.AgentConfig{
	MaxIterations:  10,      // 最大推理步骤
	TimeoutSeconds: 30,      // 执行超时
	EnableSandbox:  true,    // 启用沙箱
	MemoryLimitMb:  256,     // 内存限制
}
```

## 总结：活动系统的设计哲学

Shannon的活动系统架构体现了**微服务设计原则在AI代理领域的应用**：

### 核心设计原则

1. **单一职责**：每个活动专注一个功能
2. **可观测性**：完整的监控和事件流
3. **容错性**：自动重试和熔断保护
4. **可扩展性**：轻松添加新活动类型

### 技术创新点

- **智能工具选择**：基于查询内容动态选择工具
- **流式执行**：实时反馈和渐进式响应
- **上下文管理**：安全的会话状态传递
- **事件驱动**：丰富的事件系统支持实时监控

### 对AI代理系统的意义

活动系统将复杂的AI代理执行分解为：
- **可管理的组件**：每个活动独立开发和测试
- **可扩展的架构**：新功能通过新活动添加
- **可靠的执行**：内置重试、超时、监控
- **用户友好的体验**：实时进度和错误反馈

这个架构不仅解决了技术问题，更重要的是为AI代理系统提供了**生产级的执行基础**。当AI变得越来越复杂时，这样的分层架构确保了系统仍然可维护、可扩展和可靠。

在接下来的文章中，我们将探索简单任务工作流，了解如何将这些活动编排成完整的执行流程。敬请期待！
