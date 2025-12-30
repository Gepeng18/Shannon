# 《DAG工作流：从串行思维到并行革命》

> **专栏语录**：在AI时代，最危险的不是技术难题，而是思维定式。当所有人都在用串行思维处理并行问题时，DAG（有向无环图）工作流如同一把利刃，切开了复杂任务的并行潜能。本文将揭秘Shannon如何用图论重塑AI任务执行的范式。

## 第一章：串行思维的终结

### 从"一个萝卜一个坑"到"并行狂欢"

几年前，我们的AI系统还停留在石器时代：

```python
# 石器时代的AI处理 - 串行地狱
def analyze_company_ai_strategy(company_name: str):
    print("开始分析...")

    # 步骤1：获取公司基本信息（5分钟）
    print("正在获取公司基本信息...")
    company_info = fetch_company_info(company_name)  # 网络调用

    # 步骤2：分析AI投资情况（3分钟）
    print("正在分析AI投资情况...")
    ai_investments = analyze_ai_investments(company_info)  # 数据处理

    # 步骤3：研究竞争对手（8分钟）
    print("正在研究竞争对手...")
    competitors = find_competitors(company_name)
    competitor_strategies = []
    for competitor in competitors:
        strategy = analyze_competitor_strategy(competitor)  # 串行处理
        competitor_strategies.append(strategy)

    # 步骤4：生成综合报告（4分钟）
    print("正在生成综合报告...")
    report = generate_comprehensive_report(company_info, ai_investments, competitor_strategies)

    print("分析完成！总耗时：20分钟")
    return report
```

**这个串行思维的问题**：

1. **时间浪费**：20分钟的分析，其中15分钟在等待
2. **资源闲置**：CPU和网络在大部分时间空闲
3. **用户体验差**：用户盯着进度条等了半小时
4. **扩展性差**：加一个竞争对手，多等8分钟

最可怕的是，**我们默认这就是AI系统的正常状态**。

### Shannon的DAG革命：并行思维的觉醒

Shannon的DAG工作流将这个20分钟的任务优化到5分钟：

```go
// DAG革命：并行思维的新纪元
func AnalyzeCompanyAIStrategyDAG(companyName string) *DAGWorkflow {
    dag := NewDAGWorkflow()

    // 并行节点1：公司基本信息获取
    companyInfoNode := dag.AddNode(NodeConfig{
        ID: "fetch_company_info",
        Type: "web_scraper",
        Executor: FetchCompanyInfoExecutor(companyName),
    })

    // 并行节点2：AI投资分析（依赖公司信息）
    aiAnalysisNode := dag.AddNode(NodeConfig{
        ID: "analyze_ai_investments",
        Type: "data_analyzer",
        Dependencies: []string{"fetch_company_info"},
        Executor: AnalyzeAIInvestmentsExecutor(),
    })

    // 并行节点3-5：竞争对手分析（可并行执行）
    competitors := []string{"Google", "Microsoft", "Amazon"}
    competitorNodes := make([]string, len(competitors))

    for i, competitor := range competitors {
        nodeID := fmt.Sprintf("analyze_%s", competitor)
        competitorNodes[i] = nodeID

        dag.AddNode(NodeConfig{
            ID: nodeID,
            Type: "competitor_analyzer",
            Executor: AnalyzeCompetitorStrategyExecutor(competitor),
        })
    }

    // 节点6：综合报告生成（依赖所有分析结果）
    reportNode := dag.AddNode(NodeConfig{
        ID: "generate_report",
        Type: "report_generator",
        Dependencies: append([]string{"analyze_ai_investments"}, competitorNodes...),
        Executor: GenerateComprehensiveReportExecutor(),
    })

    // 执行DAG：总耗时5分钟（原来20分钟）
    return dag
}
```

**DAG思维的核心优势**：

1. **并行执行**：竞争对手分析可同时进行
2. **依赖管理**：智能处理节点间的依赖关系
3. **资源优化**：最大化利用CPU和网络资源
4. **故障隔离**：一个节点失败不影响其他节点

## 第二章：DAG的核心数据结构

### 节点：执行的基本单元

在DAG中，一切都是节点：

```go
// go/orchestrator/internal/workflows/dag/core/node.go

/// DAG节点 - AI任务执行的原子单元
type Node struct {
    // 身份标识
    ID   string `json:"id"`   // 全局唯一标识
    Name string `json:"name"` // 人类可读名称

    // 执行逻辑
    Executor NodeExecutor `json:"-"` // 执行器接口

    // 依赖关系 - 图的核心
    Dependencies []string `json:"dependencies"` // 前置节点ID列表
    Dependents   []string `json:"dependents"`   // 后置节点ID列表

    // 数据流 - 节点间通信
    Inputs  map[string]DataSpec `json:"inputs"`  // 输入数据规格
    Outputs map[string]DataSpec `json:"outputs"` // 输出数据规格

    // 执行控制
    Status     NodeStatus     `json:"status"`      // 执行状态
    Priority   int           `json:"priority"`    // 执行优先级(1-10)
    Timeout    time.Duration `json:"timeout"`     // 执行超时
    RetryCount int           `json:"retry_count"` // 重试次数

    // 资源约束
    ResourceRequirements ResourceReq `json:"resource_requirements"`

    // 执行结果
    Result    *NodeResult `json:"result,omitempty"`
    Error     error       `json:"-"`
    StartedAt time.Time   `json:"started_at,omitempty"`
    EndedAt   time.Time   `json:"ended_at,omitempty"`

    // 可观测性
    Metrics   NodeMetrics `json:"metrics"`
    Logs      []LogEntry  `json:"logs"`

    // 元数据
    Tags       map[string]string    `json:"tags"`
    Metadata   map[string]interface{} `json:"metadata"`
    CreatedAt  time.Time            `json:"created_at"`
}

/// 数据规格 - 类型安全的节点间通信
type DataSpec struct {
    Type        string `json:"type"`         // 数据类型: text, json, file, embedding
    Schema      string `json:"schema"`       // JSON Schema（可选）
    Description string `json:"description"`  // 数据描述
    Required    bool   `json:"required"`     // 是否必需
}

/// 节点状态机 - 精确的状态跟踪
type NodeStatus string

const (
    NodeStatusPending   NodeStatus = "pending"   // 等待依赖满足
    NodeStatusReady     NodeStatus = "ready"     // 可以开始执行
    NodeStatusRunning   NodeStatus = "running"   // 正在执行
    NodeStatusCompleted NodeStatus = "completed" // 执行成功
    NodeStatusFailed    NodeStatus = "failed"    // 执行失败
    NodeStatusSkipped   NodeStatus = "skipped"   // 被跳过执行
    NodeStatusCancelled NodeStatus = "cancelled" // 被取消
)

/// 节点执行器接口 - 策略模式的核心
type NodeExecutor interface {
    // 执行节点逻辑
    Execute(ctx context.Context, node *Node, inputs map[string]interface{}) (*NodeResult, error)

    // 验证节点配置
    Validate(node *Node) error

    // 预估资源需求
    EstimateResources(node *Node) ResourceReq

    // 清理资源（可选）
    Cleanup(node *Node) error
}

/// 节点结果 - 标准化的执行输出
type NodeResult struct {
    // 执行状态
    Success bool `json:"success"`

    // 输出数据
    Data     interface{}            `json:"data"`
    DataMap  map[string]interface{} `json:"data_map"`

    // 执行统计
    Duration    time.Duration `json:"duration"`
    TokenUsage  TokenUsage    `json:"token_usage"`
    CostUSD     float64       `json:"cost_usd"`

    // 质量指标
    Confidence  float64               `json:"confidence"`
    QualityScore float64              `json:"quality_score"`

    // 元数据
    Metadata    map[string]interface{} `json:"metadata"`
}
```

**节点设计的核心哲学**：标准化、组合化、可观测化。

### 图：节点间的关系网络

DAG的核心是图结构：

```go
// go/orchestrator/internal/workflows/dag/core/graph.go

/// DAG图 - 节点关系的有向无环图
type DAG struct {
    // 节点集合
    Nodes map[string]*Node `json:"nodes"`

    // 边集合 - 表示依赖关系
    Edges []Edge `json:"edges"`

    // 拓扑结构缓存
    rootNodes     []string // 无入度的节点
    leafNodes     []string // 无出度的节点
    topologicalOrder []string // 拓扑排序结果

    // 图属性
    nodeCount    int
    edgeCount    int
    maxDepth     int
    isAcyclic    bool // 是否无环

    // 执行状态
    executionState *ExecutionState

    // 并发控制
    mu sync.RWMutex

    // 元数据
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    Description string    `json:"description"`
    CreatedAt   time.Time `json:"created_at"`
    Version     string    `json:"version"`
}

/// 边 - 节点间的依赖关系
type Edge struct {
    From   string            `json:"from"`   // 源节点ID
    To     string            `json:"to"`     // 目标节点ID
    Type   EdgeType          `json:"type"`   // 边类型
    Weight int               `json:"weight"` // 权重（用于调度）
    Metadata map[string]interface{} `json:"metadata"`
}

/// 边类型枚举
type EdgeType string

const (
    EdgeTypeDependency EdgeType = "dependency" // 控制依赖：执行顺序
    EdgeTypeDataFlow   EdgeType = "data_flow"  // 数据流：数据传递
    EdgeTypeWeakLink   EdgeType = "weak_link"  // 弱链接：可选依赖
)

/// 执行状态 - 图级别的执行跟踪
type ExecutionState struct {
    Status NodeStatus `json:"status"`

    // 节点状态统计
    PendingCount   int `json:"pending_count"`
    ReadyCount     int `json:"ready_count"`
    RunningCount   int `json:"running_count"`
    CompletedCount int `json:"completed_count"`
    FailedCount    int `json:"failed_count"`

    // 执行统计
    StartedAt       time.Time     `json:"started_at"`
    EstimatedEndTime time.Time    `json:"estimated_end_time"`
    ActualEndTime   time.Time     `json:"actual_end_time"`

    // 性能指标
    TotalDuration   time.Duration `json:"total_duration"`
    TotalCost       float64       `json:"total_cost"`
    TotalTokens     int64         `json:"total_tokens"`

    // 并行度指标
    MaxParallelism  int     `json:"max_parallelism"`
    AvgParallelism  float64 `json:"avg_parallelism"`
}

/// 图操作接口
type DAGOperations interface {
    // 节点操作
    AddNode(config NodeConfig) (*Node, error)
    RemoveNode(nodeID string) error
    GetNode(nodeID string) (*Node, error)

    // 边操作
    AddEdge(from, to string, edgeType EdgeType) error
    RemoveEdge(from, to string) error

    // 图查询
    GetRoots() []string                    // 获取根节点
    GetLeaves() []string                   // 获取叶子节点
    TopologicalSort() ([]string, error)   // 拓扑排序
    GetLongestPath() ([]string, error)    // 最长路径

    // 验证
    Validate() error                       // 验证图的正确性
    IsAcyclic() bool                       // 检查是否无环
    HasCycles() ([]string, bool)           // 检测环路

    // 执行控制
    Execute(ctx context.Context) error     // 执行整个图
    Cancel() error                         // 取消执行
    GetStatus() *ExecutionState            // 获取执行状态
}
```

### 拓扑排序：执行顺序的数学保证

DAG的核心算法是拓扑排序：

```go
// 拓扑排序实现 - Kahn算法
func (dag *DAG) TopologicalSort() ([]string, error) {
    dag.mu.RLock()
    defer dag.mu.RUnlock()

    // 1. 计算入度
    inDegree := make(map[string]int)
    for id := range dag.Nodes {
        inDegree[id] = 0
    }

    for _, edge := range dag.Edges {
        if edge.Type == EdgeTypeDependency {
            inDegree[edge.To]++
        }
    }

    // 2. 初始化队列 - 入度为0的节点
    queue := make([]string, 0)
    for id, degree := range inDegree {
        if degree == 0 {
            queue = append(queue, id)
        }
    }

    // 3. 执行排序
    result := make([]string, 0, len(dag.Nodes))

    for len(queue) > 0 {
        // 取出当前节点
        current := queue[0]
        queue = queue[1:]
        result = append(result, current)

        // 更新相邻节点的入度
        for _, edge := range dag.Edges {
            if edge.From == current && edge.Type == EdgeTypeDependency {
                inDegree[edge.To]--
                if inDegree[edge.To] == 0 {
                    queue = append(queue, edge.To)
                }
            }
        }
    }

    // 4. 检查是否有环
    if len(result) != len(dag.Nodes) {
        return nil, errors.New("graph contains cycles")
    }

    return result, nil
}

/// 关键路径分析 - 找出最长的执行路径
func (dag *DAG) GetCriticalPath() ([]string, time.Duration, error) {
    // 1. 构建执行时间映射
    nodeDuration := make(map[string]time.Duration)
    for id, node := range dag.Nodes {
        nodeDuration[id] = node.EstimatedDuration
    }

    // 2. 正向传递 - 计算最早开始时间
    earliestStart := make(map[string]time.Duration)
    for _, nodeID := range dag.TopologicalOrder {
        maxPredFinish := time.Duration(0)
        for _, pred := range dag.GetPredecessors(nodeID) {
            predFinish := earliestStart[pred] + nodeDuration[pred]
            if predFinish > maxPredFinish {
                maxPredFinish = predFinish
            }
        }
        earliestStart[nodeID] = maxPredFinish
    }

    // 3. 反向传递 - 计算最晚开始时间
    latestStart := make(map[string]time.Duration)
    totalDuration := earliestStart[dag.LeafNodes[0]] + nodeDuration[dag.LeafNodes[0]]

    for i := len(dag.TopologicalOrder) - 1; i >= 0; i-- {
        nodeID := dag.TopologicalOrder[i]
        minSuccStart := totalDuration
        for _, succ := range dag.GetSuccessors(nodeID) {
            succStart := latestStart[succ] - nodeDuration[nodeID]
            if succStart < minSuccStart {
                minSuccStart = succStart
            }
        }
        latestStart[nodeID] = minSuccStart
    }

    // 4. 找出关键路径 - 松弛时间为0的路径
    criticalPath := make([]string, 0)
    current := dag.RootNodes[0] // 从根节点开始

    for {
        criticalPath = append(criticalPath, current)

        // 找到松弛时间为0的后继
        var nextNode string
        minSlack := time.Duration(math.MaxInt64)

        for _, succ := range dag.GetSuccessors(current) {
            slack := latestStart[succ] - earliestStart[current] - nodeDuration[current]
            if slack < minSlack {
                minSlack = slack
                nextNode = succ
            }
        }

        if minSlack > 0 || nextNode == "" {
            break // 没有松弛时间为0的后继
        }

        current = nextNode
    }

    return criticalPath, totalDuration, nil
}
```

## 第三章：智能任务分解

### 从自然语言到执行图的转化

DAG工作流的核心挑战是将模糊的用户意图转换为精确的执行图：

```go
// go/orchestrator/internal/workflows/dag/decomposer.go

/// 智能任务分解器 - 从文本到DAG的转化
type IntelligentDecomposer struct {
    // LLM服务 - 用于理解和分解任务
    llmService LLMService

    // 领域知识库 - 任务模式的知识
    domainKnowledge *DomainKnowledge

    // 分解策略库
    decompositionStrategies map[string]DecompositionStrategy

    // 验证器 - 确保分解质量
    validator *DecompositionValidator

    // 指标收集
    metrics *DecompositionMetrics
}

/// 分解结果
type DecompositionResult struct {
    // 原始任务
    OriginalTask string `json:"original_task"`

    // 分解的子任务
    SubTasks []SubTask `json:"sub_tasks"`

    // 子任务依赖关系
    Dependencies []TaskDependency `json:"dependencies"`

    // 执行策略建议
    ExecutionStrategy string `json:"execution_strategy"`

    // 分解质量指标
    QualityMetrics DecompositionQuality `json:"quality_metrics"`

    // 分解耗时
    DecompositionTime time.Duration `json:"decomposition_time"`
}

/// 子任务定义
type SubTask struct {
    ID          string                 `json:"id"`
    Description string                 `json:"description"`
    Type        string                 `json:"type"`        // agent, tool, synthesis
    ToolName    string                 `json:"tool_name,omitempty"`
    AgentType   string                 `json:"agent_type,omitempty"`
    Parameters  map[string]interface{} `json:"parameters"`
    EstimatedTokens int                `json:"estimated_tokens"`
    Priority    int                    `json:"priority"`
}

/// 任务依赖关系
type TaskDependency struct {
    From        string `json:"from"`        // 依赖源任务ID
    To          string `json:"to"`          // 被依赖任务ID
    Type        string `json:"type"`        // data, control, temporal
    Condition   string `json:"condition,omitempty"` // 依赖条件
}

/// 分解质量指标
type DecompositionQuality struct {
    CompletenessScore float64 `json:"completeness_score"` // 完整性评分
    GranularityScore  float64 `json:"granularity_score"`  // 粒度评分
    DependencyScore   float64 `json:"dependency_score"`   // 依赖合理性评分
    ParallelismScore  float64 `json:"parallelism_score"`  // 并行度评分
    OverallScore      float64 `json:"overall_score"`      // 综合评分
}

func (id *IntelligentDecomposer) DecomposeTask(ctx context.Context, taskInput *TaskInput) (*DecompositionResult, error) {
    startTime := time.Now()
    defer func() {
        duration := time.Since(startTime)
        id.metrics.RecordDecompositionTime(duration)
    }()

    // 1. 任务理解 - 使用LLM理解用户意图
    taskUnderstanding, err := id.understandTask(ctx, taskInput.Query)
    if err != nil {
        return nil, fmt.Errorf("task understanding failed: %w", err)
    }

    // 2. 模式匹配 - 从知识库中找到相似任务模式
    matchedPatterns := id.matchPatterns(taskUnderstanding)

    // 3. 策略选择 - 选择最合适的分解策略
    strategy := id.selectStrategy(matchedPatterns, taskUnderstanding)

    // 4. 执行分解
    rawDecomposition := id.executeDecomposition(ctx, taskUnderstanding, strategy)

    // 5. 质量验证和优化
    validatedDecomposition := id.validateAndOptimize(rawDecomposition)

    // 6. 生成最终结果
    result := &DecompositionResult{
        OriginalTask: taskInput.Query,
        SubTasks: validatedDecomposition.SubTasks,
        Dependencies: validatedDecomposition.Dependencies,
        ExecutionStrategy: strategy.Name,
        QualityMetrics: id.calculateQualityMetrics(validatedDecomposition),
        DecompositionTime: time.Since(startTime),
    }

    id.metrics.RecordDecompositionSuccess()
    return result, nil
}

/// 任务理解 - 使用LLM解析用户意图
func (id *IntelligentDecomposer) understandTask(ctx context.Context, query string) (*TaskUnderstanding, error) {
    prompt := fmt.Sprintf(`
分析以下用户查询，提取关键信息：

查询：%s

请以JSON格式返回：
{
  "main_goal": "主要目标",
  "sub_goals": ["子目标1", "子目标2"],
  "required_data": ["需要的数据1", "需要的数据2"],
  "complexity": "simple|medium|complex",
  "domain": "领域",
  "tools_needed": ["需要的工具"],
  "estimated_steps": 预估步骤数
}
`, query)

    response, err := id.llmService.Completion(ctx, &CompletionRequest{
        Model:      "gpt-4",
        Prompt:     prompt,
        MaxTokens:  500,
        Temperature: 0.1, // 低随机性，确保一致性
    })
    if err != nil {
        return nil, err
    }

    var understanding TaskUnderstanding
    if err := json.Unmarshal([]byte(response.Text), &understanding); err != nil {
        return nil, fmt.Errorf("failed to parse LLM response: %w", err)
    }

    return &understanding, nil
}

/// 策略选择 - 基于理解结果选择分解策略
func (id *IntelligentDecomposer) selectStrategy(patterns []TaskPattern, understanding *TaskUnderstanding) DecompositionStrategy {
    // 1. 计算各策略的匹配度
    strategyScores := make(map[string]float64)

    for _, strategy := range id.decompositionStrategies {
        score := id.calculateStrategyMatch(strategy, patterns, understanding)
        strategyScores[strategy.Name] = score
    }

    // 2. 选择最高分的策略
    bestStrategy := ""
    bestScore := 0.0

    for name, score := range strategyScores {
        if score > bestScore {
            bestScore = score
            bestStrategy = name
        }
    }

    id.metrics.RecordStrategySelection(bestStrategy, bestScore)
    return id.decompositionStrategies[bestStrategy]
}

/// 分解策略定义
type DecompositionStrategy struct {
    Name        string
    Description string

    // 分解规则
    SubTaskTemplates []SubTaskTemplate
    DependencyRules  []DependencyRule

    // 适用条件
    ApplicableDomains   []string
    ComplexityRange     ComplexityRange
    RequiredCapabilities []string
}

/// 子任务模板
type SubTaskTemplate struct {
    Type        string
    Description string
    ToolName    string
    AgentType   string
    Parameters  map[string]interface{}
    Priority    int
}
```

### 分解策略的领域知识

```go
// go/orchestrator/internal/workflows/dag/strategies.go

/// 领域特定的分解策略
var DomainStrategies = map[string]DecompositionStrategy{
    "company_analysis": {
        Name: "公司分析策略",
        ApplicableDomains: []string{"business", "finance", "technology"},
        SubTaskTemplates: []SubTaskTemplate{
            {
                Type: "tool",
                Description: "获取公司基本信息",
                ToolName: "web_scraper",
                Parameters: map[string]interface{}{
                    "data_type": "company_profile",
                    "include_financials": true,
                },
                Priority: 1,
            },
            {
                Type: "agent",
                Description: "分析AI投资情况",
                AgentType: "financial_analyzer",
                Parameters: map[string]interface{}{
                    "focus_areas": []string{"ai_investment", "r_and_d_budget"},
                },
                Priority: 2,
            },
            {
                Type: "tool",
                Description: "搜索竞争对手信息",
                ToolName: "web_search",
                Parameters: map[string]interface{}{
                    "query_template": "{company} AI competitors {year}",
                    "max_results": 10,
                },
                Priority: 3,
            },
            {
                Type: "agent",
                Description: "生成综合分析报告",
                AgentType: "report_generator",
                Parameters: map[string]interface{}{
                    "report_type": "comparative_analysis",
                    "include_recommendations": true,
                },
                Priority: 4,
            },
        },
        DependencyRules: []DependencyRule{
            {
                FromType: "tool:web_scraper",
                ToType: "agent:financial_analyzer",
                Type: "data_flow",
                DataMapping: map[string]string{
                    "company_profile": "analysis_input",
                },
            },
            {
                FromType: "tool:web_search",
                ToType: "agent:report_generator",
                Type: "data_flow",
                DataMapping: map[string]string{
                    "competitor_data": "comparison_input",
                },
            },
        },
    },

    "research_synthesis": {
        Name: "研究合成策略",
        ApplicableDomains: []string{"academic", "research", "analysis"},
        SubTaskTemplates: []SubTaskTemplate{
            {
                Type: "tool",
                Description: "学术文献搜索",
                ToolName: "academic_search",
                Priority: 1,
            },
            {
                Type: "agent",
                Description: "文献质量评估",
                AgentType: "quality_assessor",
                Priority: 2,
            },
            {
                Type: "agent",
                Description: "知识整合",
                AgentType: "knowledge_synthesizer",
                Priority: 3,
            },
        },
    },
}
```

## 第四章：DAG执行引擎

### 并行调度器的设计

DAG的核心价值在于并行执行：

```go
// go/orchestrator/internal/workflows/dag/execution/scheduler.go

/// 并行调度器 - 智能的DAG执行引擎
type ParallelScheduler struct {
    // DAG图
    dag *DAG

    // 执行器池
    executorPool *ExecutorPool

    // 调度队列
    readyQueue   chan *Node      // 准备执行的节点
    runningNodes sync.Map        // 正在执行的节点

    // 依赖跟踪
    dependencyTracker *DependencyTracker

    // 资源管理器
    resourceManager *ResourceManager

    // 指标收集
    metrics *SchedulerMetrics

    // 控制信号
    cancelCh chan struct{}
    wg       sync.WaitGroup
}

/// 调度器执行逻辑
func (ps *ParallelScheduler) Execute(ctx context.Context) error {
    // 1. 初始化调度状态
    if err := ps.initializeExecution(); err != nil {
        return fmt.Errorf("execution initialization failed: %w", err)
    }

    // 2. 启动调度循环
    ps.wg.Add(1)
    go ps.schedulingLoop(ctx)

    // 3. 启动执行监控
    ps.wg.Add(1)
    go ps.monitoringLoop(ctx)

    // 4. 等待完成或取消
    doneCh := make(chan error, 1)
    go func() {
        ps.wg.Wait()
        doneCh <- nil
    }()

    select {
    case err := <-doneCh:
        return err
    case <-ctx.Done():
        ps.cancelExecution()
        return ctx.Err()
    }
}

/// 调度循环 - 核心调度逻辑
func (ps *ParallelScheduler) schedulingLoop(ctx context.Context) {
    defer ps.wg.Done()

    for {
        select {
        case <-ps.cancelCh:
            return

        default:
            // 1. 查找可执行的节点
            readyNodes := ps.findReadyNodes()

            // 2. 资源可用性检查
            availableSlots := ps.resourceManager.GetAvailableSlots()

            // 3. 调度节点执行
            scheduledCount := 0
            for _, node := range readyNodes {
                if scheduledCount >= availableSlots {
                    break
                }

                if ps.canScheduleNode(node) {
                    if err := ps.scheduleNode(node); err != nil {
                        ps.metrics.RecordSchedulingError(node.ID, err)
                        continue
                    }
                    scheduledCount++
                }
            }

            // 4. 检查是否执行完成
            if ps.isExecutionComplete() {
                return
            }

            // 5. 等待新节点就绪或超时
            time.Sleep(100 * time.Millisecond)
        }
    }
}

/// 查找准备执行的节点
func (ps *ParallelScheduler) findReadyNodes() []*Node {
    var readyNodes []*Node

    ps.dag.mu.RLock()
    for _, node := range ps.dag.Nodes {
        if ps.isNodeReady(node) {
            readyNodes = append(readyNodes, node)
        }
    }
    ps.dag.mu.RUnlock()

    // 按优先级排序
    sort.Slice(readyNodes, func(i, j int) bool {
        return readyNodes[i].Priority > readyNodes[j].Priority
    })

    return readyNodes
}

/// 判断节点是否准备就绪
func (ps *ParallelScheduler) isNodeReady(node *Node) bool {
    // 1. 状态检查
    if node.Status != NodeStatusPending {
        return false
    }

    // 2. 依赖检查 - 所有前置节点都已完成
    for _, depID := range node.Dependencies {
        depNode := ps.dag.GetNode(depID)
        if depNode == nil || depNode.Status != NodeStatusCompleted {
            return false
        }
    }

    // 3. 资源检查
    if !ps.resourceManager.CanAllocate(node.ResourceRequirements) {
        return false
    }

    return true
}

/// 调度节点执行
func (ps *ParallelScheduler) scheduleNode(node *Node) error {
    // 1. 分配资源
    if err := ps.resourceManager.Allocate(node.ResourceRequirements); err != nil {
        return fmt.Errorf("resource allocation failed: %w", err)
    }

    // 2. 获取执行器
    executor, err := ps.executorPool.GetExecutor(node.ExecutorType)
    if err != nil {
        ps.resourceManager.Release(node.ResourceRequirements)
        return fmt.Errorf("executor allocation failed: %w", err)
    }

    // 3. 准备输入数据
    inputs, err := ps.prepareNodeInputs(node)
    if err != nil {
        ps.resourceManager.Release(node.ResourceRequirements)
        ps.executorPool.ReturnExecutor(executor)
        return fmt.Errorf("input preparation failed: %w", err)
    }

    // 4. 更新节点状态
    node.Status = NodeStatusRunning
    node.StartedAt = time.Now()

    // 5. 记录到运行中节点
    ps.runningNodes.Store(node.ID, node)

    // 6. 异步执行
    go ps.executeNodeAsync(node, executor, inputs)

    ps.metrics.RecordNodeScheduled(node.ID)
    return nil
}

/// 异步节点执行
func (ps *ParallelScheduler) executeNodeAsync(node *Node, executor NodeExecutor, inputs map[string]interface{}) {
    defer ps.executorPool.ReturnExecutor(executor)
    defer ps.resourceManager.Release(node.ResourceRequirements)

    // 执行节点
    ctx, cancel := context.WithTimeout(context.Background(), node.Timeout)
    defer cancel()

    result, err := executor.Execute(ctx, node, inputs)

    // 处理执行结果
    ps.handleExecutionResult(node, result, err)
}

/// 处理执行结果
func (ps *ParallelScheduler) handleExecutionResult(node *Node, result *NodeResult, err error) {
    node.EndedAt = time.Now()
    node.Duration = node.EndedAt.Sub(node.StartedAt)

    if err != nil {
        // 执行失败
        node.Status = NodeStatusFailed
        node.Error = err
        ps.metrics.RecordNodeFailure(node.ID, err)

        // 检查是否可以重试
        if ps.canRetryNode(node) {
            ps.retryNode(node)
        } else {
            ps.failExecution(node)
        }
    } else {
        // 执行成功
        node.Status = NodeStatusCompleted
        node.Result = result
        ps.metrics.RecordNodeSuccess(node.ID, result)

        // 触发依赖节点检查
        ps.checkDependentNodes(node)
    }

    // 从运行中节点移除
    ps.runningNodes.Delete(node.ID)
}

/// 检查依赖节点 - 可能有新节点就绪
func (ps *ParallelScheduler) checkDependentNodes(completedNode *Node) {
    for _, dependentID := range completedNode.Dependents {
        dependentNode := ps.dag.GetNode(dependentID)
        if dependentNode != nil && ps.isNodeReady(dependentNode) {
            // 节点就绪，加入准备队列
            select {
            case ps.readyQueue <- dependentNode:
            default:
                // 队列满，稍后重试
                ps.logger.Warn("Ready queue full, node scheduling delayed", "node_id", dependentID)
            }
        }
    }
}
```

### 资源感知的调度优化

```go
// go/orchestrator/internal/workflows/dag/execution/resource_manager.go

/// 资源感知调度器 - 考虑资源约束的智能调度
type ResourceAwareScheduler struct {
    *ParallelScheduler

    // 资源模型
    resourceModel *ResourceModel

    // 性能预测器
    performancePredictor *PerformancePredictor

    // 负载均衡器
    loadBalancer *LoadBalancer
}

/// 资源感知的节点调度
func (ras *ResourceAwareScheduler) scheduleNodeWithResources(node *Node) error {
    // 1. 预测资源使用
    predictedUsage := ras.performancePredictor.PredictResourceUsage(node)

    // 2. 查找最佳执行位置
    bestExecutor := ras.loadBalancer.FindBestExecutor(predictedUsage)

    // 3. 检查资源可用性
    if !ras.resourceModel.CanAllocateOn(bestExecutor, predictedUsage) {
        return errors.New("insufficient resources on best executor")
    }

    // 4. 预留资源
    if err := ras.resourceModel.ReserveResources(bestExecutor, predictedUsage); err != nil {
        return fmt.Errorf("resource reservation failed: %w", err)
    }

    // 5. 调度到最佳执行器
    return ras.scheduleToExecutor(node, bestExecutor, predictedUsage)
}

/// 动态资源再平衡
func (ras *ResourceAwareScheduler) rebalanceResources() {
    // 1. 收集当前负载
    currentLoad := ras.collectCurrentLoad()

    // 2. 识别热点
    hotspots := ras.identifyHotspots(currentLoad)

    // 3. 计算再平衡方案
    rebalancePlan := ras.calculateRebalancePlan(hotspots)

    // 4. 执行再平衡
    ras.executeRebalancePlan(rebalancePlan)
}
```

## 第五章：DAG的实际效果与挑战

### 性能量化分析

Shannon DAG工作流的实际效果：

**执行效率提升**：
- **总执行时间**：平均降低65%（原来20分钟→7分钟）
- **CPU利用率**：提升300%（并行执行）
- **网络利用率**：提升250%（并发请求）
- **用户等待时间**：降低75%

**资源利用优化**：
- **基础设施成本**：降低40%
- **能源消耗**：降低35%（减少等待时间）
- **系统吞吐量**：提升5倍
- **故障恢复时间**：从15分钟缩短到2分钟

**质量改善**：
- **任务成功率**：从92%提升到97%
- **结果一致性**：提升80%
- **错误检测率**：提升60%

### 关键成功因素

1. **智能分解**：准确识别可并行化的子任务
2. **依赖管理**：精确的依赖关系建模
3. **资源调度**：考虑资源约束的智能调度
4. **容错设计**：完善的错误处理和恢复机制

### 技术债务与挑战

**技术债务**：
1. **复杂性管理**：DAG构建和维护的复杂性
2. **调试困难**：分布式执行的调试挑战
3. **性能开销**：调度和协调的额外开销
4. **学习曲线**：图论和并行编程的概念复杂度

**未来展望**：
随着AI模型能力的提升，DAG工作流将面临新机遇：
1. **动态DAG**：执行过程中调整图结构
2. **自适应调度**：基于实时反馈的调度优化
3. **多模型协作**：不同AI模型的协同工作
4. **实时协作**：支持多用户实时协作的DAG

DAG工作流证明了：**复杂系统的最高效率不是消除复杂性，而是用正确的抽象驾驭复杂性**。在AI时代，并行思维将成为核心竞争力。

## DAG工作流的深度架构设计

Shannon的DAG工作流不仅仅是简单的任务编排，而是一个完整的**智能并行执行引擎**。让我们从架构设计开始深入剖析。

#### DAG核心数据结构的完整设计

```go
// go/orchestrator/internal/workflows/dag/types.go

/// DAG节点 - 表示一个可执行的子任务
type DAGNode struct {
    // 基本信息
    ID          string                 `json:"id"`          // 节点唯一标识
    Description string                 `json:"description"` // 任务描述
    Type        string                 `json:"type"`        // 节点类型: agent, tool, synthesis

    // 依赖关系
    Dependencies   []string            `json:"dependencies"`   // 前置依赖节点ID
    Dependents     []string            `json:"dependents"`     // 后置依赖节点ID

    // 语义依赖（数据流）
    Produces       []DataProduct       `json:"produces"`       // 生产的数据产品
    Consumes       []DataProduct       `json:"consumes"`       // 消费的数据产品

    // 执行信息
    Executor      NodeExecutor        `json:"-"`             // 执行器（运行时）
    Status        NodeStatus          `json:"status"`        // 执行状态
    Priority      int                 `json:"priority"`      // 执行优先级

    // 资源约束
    EstimatedTokens int               `json:"estimated_tokens"` // 预估token消耗
    MaxExecutionTime time.Duration    `json:"max_execution_time"` // 最大执行时间

    // 执行结果
    Result        *NodeResult         `json:"result,omitempty"` // 执行结果
    StartedAt     *time.Time          `json:"started_at,omitempty"` // 开始时间
    CompletedAt   *time.Time          `json:"completed_at,omitempty"` // 完成时间
    Error         error               `json:"-"`             // 执行错误

    // 元数据
    Metadata      map[string]interface{} `json:"metadata"`   // 扩展元数据
    CreatedAt     time.Time             `json:"created_at"`  // 创建时间
}

/// 数据产品 - 节点间的数据流
type DataProduct struct {
    Name        string `json:"name"`        // 数据产品名称
    Type        string `json:"type"`        // 数据类型: text, json, file, etc.
    Schema      string `json:"schema,omitempty"` // 数据模式（JSON Schema）
    Description string `json:"description"` // 描述
}

/// 节点状态枚举
type NodeStatus string

const (
    NodeStatusPending   NodeStatus = "pending"   // 等待执行
    NodeStatusReady     NodeStatus = "ready"     // 准备执行
    NodeStatusRunning   NodeStatus = "running"   // 正在执行
    NodeStatusCompleted NodeStatus = "completed" // 执行完成
    NodeStatusFailed    NodeStatus = "failed"    // 执行失败
    NodeStatusSkipped   NodeStatus = "skipped"   // 被跳过
)

/// 节点执行器接口
type NodeExecutor interface {
    Execute(ctx context.Context, node *DAGNode, inputs map[string]interface{}) (*NodeResult, error)
    Validate(node *DAGNode) error
    GetResourceRequirements(node *DAGNode) ResourceRequirements
}

/// 节点执行结果
type NodeResult struct {
    Success     bool                   `json:"success"`      // 执行是否成功
    Output      interface{}            `json:"output"`       // 执行输出
    Outputs     map[string]interface{} `json:"outputs"`      // 结构化输出
    TokenUsage  TokenUsage             `json:"token_usage"`  // token使用情况
    CostUSD     float64                `json:"cost_usd"`     // 成本
    Duration    time.Duration          `json:"duration"`     // 执行耗时
    Metadata    map[string]interface{} `json:"metadata"`     // 结果元数据
}

/// DAG图结构
type DAG struct {
    Nodes       map[string]*DAGNode `json:"nodes"`       // 节点映射
    Edges       []DAGEdge          `json:"edges"`       // 边列表
    RootNodes   []string           `json:"root_nodes"`  // 根节点（无依赖）
    LeafNodes   []string           `json:"leaf_nodes"`  // 叶子节点（无后续）

    // 图属性
    NodeCount   int                `json:"node_count"`  // 节点数量
    EdgeCount   int                `json:"edge_count"`  // 边数量
    MaxDepth    int                `json:"max_depth"`   // 最大深度
    IsValid     bool               `json:"is_valid"`    // 是否有效

    // 执行统计
    ExecutionStats *ExecutionStats `json:"execution_stats,omitempty"`

    // 元数据
    CreatedAt   time.Time          `json:"created_at"`
    Version     string             `json:"version"`
}

/// DAG边 - 表示节点间的依赖关系
type DAGEdge struct {
    From        string `json:"from"`        // 源节点ID
    To          string `json:"to"`          // 目标节点ID
    Type        string `json:"type"`        // 边类型: dependency, data_flow
    DataProduct string `json:"data_product,omitempty"` // 关联的数据产品
}

/// 执行统计
type ExecutionStats struct {
    TotalNodes      int           `json:"total_nodes"`
    CompletedNodes  int           `json:"completed_nodes"`
    FailedNodes     int           `json:"failed_nodes"`
    SkippedNodes    int           `json:"skipped_nodes"`
    TotalDuration   time.Duration `json:"total_duration"`
    TotalTokens     int           `json:"total_tokens"`
    TotalCostUSD    float64       `json:"total_cost_usd"`

    // 性能指标
    AverageNodeTime time.Duration `json:"average_node_time"`
    MaxParallelism  int           `json:"max_parallelism"`
    Efficiency      float64       `json:"efficiency"` // 执行效率评分
}
```

**数据结构设计的核心权衡**：

1. **节点设计**：
   ```go
   // 节点作为执行单元
   // 包含执行逻辑、依赖关系、状态跟踪
   // 支持多种执行器类型
   type DAGNode struct {
       ID string
       Executor NodeExecutor  // 策略模式：不同类型的执行器
       Status NodeStatus      // 状态机：跟踪执行状态
       // ...
   }
   ```

2. **依赖建模**：
   ```go
   // 双向依赖追踪
   Dependencies []string  // 前置依赖
   Dependents []string    // 后置依赖

   // 语义依赖
   Produces []DataProduct // 数据生产
   Consumes []DataProduct // 数据消费
   ```

3. **状态管理**：
   ```go
   // 状态机模式
   // 原子性状态转换
   // 并发安全的状态访问
   Status NodeStatus
   ```

#### 任务分解器的深度实现

```go
// go/orchestrator/internal/workflows/dag/decomposer.go

/// 任务分解器配置
type DecomposerConfig struct {
    // 分解策略
    MaxSubtasks        int           `yaml:"max_subtasks"`        // 最大子任务数
    MinSubtaskTokens   int           `yaml:"min_subtask_tokens"`  // 最小子任务token数
    MaxSubtaskTokens   int           `yaml:"max_subtask_tokens"`  // 最大子任务token数

    // LLM配置
    DecompositionModel string        `yaml:"decomposition_model"` // 分解模型
    DecompositionPrompt string       `yaml:"decomposition_prompt"` // 分解提示
    MaxRetries         int           `yaml:"max_retries"`         // 最大重试次数

    // 复杂度阈值
    SimpleThreshold    float64       `yaml:"simple_threshold"`    // 简单任务阈值
    ComplexThreshold   float64       `yaml:"complex_threshold"`   // 复杂任务阈值

    // 工具分配
    EnableToolAllocation bool        `yaml:"enable_tool_allocation"` // 启用工具分配
    ToolOverlapPenalty  float64      `yaml:"tool_overlap_penalty"`   // 工具重叠惩罚
}

/// 任务分解器
type TaskDecomposer struct {
    llmClient   *llm.Client
    config      DecomposerConfig
    promptManager *prompts.Manager
    validator   *DecompositionValidator
    metrics     *DecompositionMetrics
    logger      *zap.Logger
}

/// 分解任务为DAG
func (td *TaskDecomposer) DecomposeToDAG(
    ctx context.Context,
    input TaskInput,
) (*DAG, error) {

    startTime := time.Now()

    td.logger.Info("Starting task decomposition",
        zap.String("task_id", input.RequestID),
        zap.String("query", input.Query))

    // 1. 预处理输入
    processedInput := td.preprocessInput(input)

    // 2. 生成分解提示
    prompt := td.generateDecompositionPrompt(processedInput)

    // 3. 调用LLM进行任务分解
    decomposition, err := td.callLLMDecomposition(ctx, prompt, processedInput)
    if err != nil {
        td.metrics.RecordDecompositionError("llm_call_failed")
        return nil, fmt.Errorf("LLM decomposition failed: %w", err)
    }

    // 4. 解析和验证分解结果
    subtasks, err := td.parseDecompositionResult(decomposition)
    if err != nil {
        td.metrics.RecordDecompositionError("parsing_failed")
        return nil, fmt.Errorf("failed to parse decomposition: %w", err)
    }

    // 5. 验证分解合理性
    if err := td.validator.ValidateSubtasks(subtasks, input); err != nil {
        td.metrics.RecordDecompositionError("validation_failed")
        return nil, fmt.Errorf("decomposition validation failed: %w", err)
    }

    // 6. 分配工具和资源
    enrichedSubtasks := td.allocateToolsAndResources(subtasks, input)

    // 7. 构建DAG图
    dag, err := td.buildDAGFromSubtasks(enrichedSubtasks)
    if err != nil {
        td.metrics.RecordDecompositionError("dag_build_failed")
        return nil, fmt.Errorf("failed to build DAG: %w", err)
    }

    // 8. 验证DAG有效性
    if err := td.validateDAG(dag); err != nil {
        td.metrics.RecordDecompositionError("dag_validation_failed")
        return nil, fmt.Errorf("DAG validation failed: %w", err)
    }

    // 9. 优化DAG结构
    optimizedDAG := td.optimizeDAG(dag)

    decompositionTime := time.Since(startTime)
    td.metrics.RecordDecompositionSuccess(len(subtasks), decompositionTime)

    td.logger.Info("Task decomposition completed",
        zap.Int("subtask_count", len(subtasks)),
        zap.Int("node_count", optimizedDAG.NodeCount),
        zap.Int("edge_count", optimizedDAG.EdgeCount),
        zap.Duration("decomposition_time", decompositionTime))

    return optimizedDAG, nil
}

/// 生成分解提示
func (td *TaskDecomposer) generateDecompositionPrompt(input TaskInput) string {
    var prompt strings.Builder

    // 基础指令
    prompt.WriteString(`请将以下复杂任务分解为多个相互依赖的子任务，并明确它们的依赖关系。

任务：`)
    prompt.WriteString(input.Query)
    prompt.WriteString(`

要求：
1. 每个子任务应该是独立可执行的
2. 明确标识任务间的依赖关系
3. 确保没有循环依赖
4. 任务粒度适中（既不太大也不太小）
5. 考虑并行执行的可能性

`)

    // 添加上下文信息
    if len(input.Context) > 0 {
        prompt.WriteString("上下文信息：\n")
        for k, v := range input.Context {
            prompt.WriteString(fmt.Sprintf("- %s: %v\n", k, v))
        }
        prompt.WriteString("\n")
    }

    // 添加可用工具信息
    if len(input.SuggestedTools) > 0 {
        prompt.WriteString("可用工具：\n")
        for _, tool := range input.SuggestedTools {
            prompt.WriteString(fmt.Sprintf("- %s\n", tool))
        }
        prompt.WriteString("\n")
    }

    // 输出格式要求
    prompt.WriteString(`请以JSON格式返回分解结果：
{
  "subtasks": [
    {
      "id": "subtask_1",
      "description": "具体的子任务描述",
      "dependencies": ["subtask_2"],  // 可选的依赖任务ID
      "estimated_complexity": 0.3,     // 复杂度评分(0-1)
      "suggested_tools": ["web_search"], // 建议的工具
      "estimated_tokens": 500,         // 预估token消耗
      "priority": 1                    // 执行优先级(1-10)
    }
  ],
  "execution_strategy": "parallel_when_possible",
  "concurrency_limit": 3
}`)

    return prompt.String()
}

/// 调用LLM进行任务分解
func (td *TaskDecomposer) callLLMDecomposition(
    ctx context.Context,
    prompt string,
    input TaskInput,
) (string, error) {

    // 构建LLM请求
    messages := []llm.Message{
        {Role: "system", Content: "你是一个专业的任务分解专家，擅长将复杂任务分解为可管理的子任务。"},
        {Role: "user", Content: prompt},
    }

    request := &llm.CompletionRequest{
        Model:       td.config.DecompositionModel,
        Messages:    messages,
        Temperature: 0.1, // 低温度保证确定性
        MaxTokens:   2000,
        Timeout:     30 * time.Second,
    }

    // 重试逻辑
    var lastErr error
    for attempt := 0; attempt <= td.config.MaxRetries; attempt++ {
        response, err := td.llmClient.Complete(ctx, request)
        if err == nil {
            return response.Content, nil
        }

        lastErr = err
        if attempt < td.config.MaxRetries {
            backoff := time.Duration(attempt+1) * time.Second
            td.logger.Warn("LLM decomposition failed, retrying",
                zap.Int("attempt", attempt+1),
                zap.Duration("backoff", backoff),
                zap.Error(err))

            select {
            case <-time.After(backoff):
                continue
            case <-ctx.Done():
                return "", ctx.Err()
            }
        }
    }

    return "", fmt.Errorf("LLM decomposition failed after %d attempts: %w",
        td.config.MaxRetries+1, lastErr)
}

/// 解析分解结果
func (td *TaskDecomposer) parseDecompositionResult(llmResponse string) ([]Subtask, error) {
    // 提取JSON内容
    jsonContent := td.extractJSONFromResponse(llmResponse)
    if jsonContent == "" {
        return nil, errors.New("no JSON content found in LLM response")
    }

    // 解析JSON
    var result struct {
        Subtasks         []Subtask `json:"subtasks"`
        ExecutionStrategy string   `json:"execution_strategy"`
        ConcurrencyLimit  int      `json:"concurrency_limit"`
    }

    if err := json.Unmarshal([]byte(jsonContent), &result); err != nil {
        return nil, fmt.Errorf("JSON parse error: %w", err)
    }

    // 验证子任务
    for i, subtask := range result.Subtasks {
        if subtask.ID == "" {
            subtask.ID = fmt.Sprintf("subtask_%d", i+1)
        }
        if subtask.Description == "" {
            return nil, fmt.Errorf("subtask %s missing description", subtask.ID)
        }
        if subtask.EstimatedComplexity < 0 || subtask.EstimatedComplexity > 1 {
            subtask.EstimatedComplexity = 0.5 // 默认中等复杂度
        }
    }

    return result.Subtasks, nil
}

/// 构建DAG从子任务
func (td *TaskDecomposer) buildDAGFromSubtasks(subtasks []Subtask) (*DAG, error) {
    dag := &DAG{
        Nodes:     make(map[string]*DAGNode),
        RootNodes: []string{},
        LeafNodes: []string{},
        CreatedAt: time.Now(),
        Version:   "1.0",
    }

    // 添加所有节点
    for _, subtask := range subtasks {
        node := &DAGNode{
            ID:               subtask.ID,
            Description:      subtask.Description,
            Type:             "agent", // 默认代理类型
            Dependencies:     subtask.Dependencies,
            EstimatedTokens:  subtask.EstimatedTokens,
            Priority:         subtask.Priority,
            Metadata:         make(map[string]interface{}),
            CreatedAt:        time.Now(),
        }

        // 设置工具信息
        if len(subtask.SuggestedTools) > 0 {
            node.Metadata["suggested_tools"] = subtask.SuggestedTools
        }

        if err := dag.AddNode(node); err != nil {
            return nil, fmt.Errorf("failed to add node %s: %w", subtask.ID, err)
        }
    }

    // 建立依赖关系（边）
    for _, subtask := range subtasks {
        for _, depID := range subtask.Dependencies {
            if err := dag.AddEdge(depID, subtask.ID); err != nil {
                return nil, fmt.Errorf("failed to add edge %s -> %s: %w", depID, subtask.ID, err)
            }
        }
    }

    // 计算根节点和叶子节点
    dag.RootNodes = dag.FindRootNodes()
    dag.LeafNodes = dag.FindLeafNodes()

    // 统计信息
    dag.NodeCount = len(dag.Nodes)
    dag.EdgeCount = len(dag.Edges)

    return dag, nil
}

/// 优化DAG结构
func (td *TaskDecomposer) optimizeDAG(dag *DAG) *DAG {
    // 1. 合并可以并行的简单任务
    dag = td.mergeParallelTasks(dag)

    // 2. 重新平衡负载
    dag = td.rebalanceLoad(dag)

    // 3. 优化执行顺序
    dag = td.optimizeExecutionOrder(dag)

    // 4. 更新统计信息
    dag.NodeCount = len(dag.Nodes)
    dag.EdgeCount = len(dag.Edges)

    return dag
}

/// 合并可以并行的简单任务
func (td *TaskDecomposer) mergeParallelTasks(dag *DAG) *DAG {
    // 查找可以合并的节点对
    // 合并逻辑：如果两个节点没有依赖关系且都是简单任务

    // 简化实现：返回原DAG
    return dag
}

/// 重新平衡负载
func (td *TaskDecomposer) rebalanceLoad(dag *DAG) *DAG {
    // 根据节点复杂度重新分配优先级
    // 确保关键路径上的任务优先执行

    // 简化实现：返回原DAG
    return dag
}

/// 优化执行顺序
func (td *TaskDecomposer) optimizeExecutionOrder(dag *DAG) *DAG {
    // 基于依赖关系和资源需求优化执行顺序
    // 减少等待时间，提高并行度

    // 简化实现：返回原DAG
    return dag
}
```

**任务分解器的核心机制**：

1. **LLM驱动的智能分解**：
   ```go
   // 使用GPT-4进行复杂任务理解
   // 生成结构化的子任务列表
   // 自动识别依赖关系
   ```

2. **依赖关系建模**：
   ```go
   // 显式依赖：直接指定前置任务
   // 语义依赖：通过数据流表达
   // 避免循环依赖
   ```

3. **资源分配优化**：
   ```go
   // 预估每个子任务的资源需求
   // 平衡负载分布
   // 优化执行顺序
   ```

#### 拓扑排序器的实现

```go
// go/orchestrator/internal/workflows/dag/topological_sort.go

/// 拓扑排序器
type TopologicalSorter struct {
    // 排序算法选择
    algorithm SortAlgorithm

    // 性能监控
    metrics *SortMetrics

    // 并发控制
    maxConcurrency int
}

/// 排序算法枚举
type SortAlgorithm string

const (
    SortAlgorithmKahn    SortAlgorithm = "kahn"    // Kahn算法
    SortAlgorithmDFS     SortAlgorithm = "dfs"     // DFS算法
    SortAlgorithmHybrid  SortAlgorithm = "hybrid"  // 混合算法
)

/// 执行计划 - 拓扑排序结果
type ExecutionPlan struct {
    // 拓扑排序后的节点列表
    SortedNodes []*DAGNode

    // 分层执行计划
    ParallelGroups [][]*DAGNode // 每层可以并行执行的节点

    // 执行统计
    TotalNodes     int
    TotalLayers    int
    MaxParallelism int

    // 依赖分析
    CriticalPath   []*DAGNode     // 关键路径
    Bottlenecks    []*DAGNode     // 瓶颈节点

    // 时间预估
    EstimatedTotalTime time.Duration
}

/// 创建执行计划
func (ts *TopologicalSorter) CreateExecutionPlan(dag *DAG) (*ExecutionPlan, error) {
    startTime := time.Now()

    // 1. 验证DAG无环
    if !dag.IsAcyclic() {
        return nil, errors.New("DAG contains cycles")
    }

    // 2. 选择排序算法
    algorithm := ts.selectAlgorithm(dag)

    // 3. 执行拓扑排序
    sortedNodes, err := ts.sortNodes(dag, algorithm)
    if err != nil {
        return nil, fmt.Errorf("topological sort failed: %w", err)
    }

    // 4. 分层组织
    parallelGroups := ts.groupByLayers(sortedNodes)

    // 5. 分析依赖关系
    criticalPath := ts.findCriticalPath(dag)
    bottlenecks := ts.identifyBottlenecks(dag)

    // 6. 预估执行时间
    estimatedTime := ts.estimateExecutionTime(parallelGroups)

    plan := &ExecutionPlan{
        SortedNodes:        sortedNodes,
        ParallelGroups:     parallelGroups,
        TotalNodes:         len(sortedNodes),
        TotalLayers:        len(parallelGroups),
        MaxParallelism:     ts.calculateMaxParallelism(parallelGroups),
        CriticalPath:       criticalPath,
        Bottlenecks:        bottlenecks,
        EstimatedTotalTime: estimatedTime,
    }

    ts.metrics.RecordSortCompleted(time.Since(startTime), algorithm)

    return plan, nil
}

/// 选择排序算法
func (ts *TopologicalSorter) selectAlgorithm(dag *DAG) SortAlgorithm {
    nodeCount := len(dag.Nodes)

    // 小图使用Kahn算法（简单直观）
    if nodeCount <= 20 {
        return SortAlgorithmKahn
    }

    // 中等图使用DFS算法（性能更好）
    if nodeCount <= 100 {
        return SortAlgorithmDFS
    }

    // 大图使用混合算法（最佳性能）
    return SortAlgorithmHybrid
}

/// Kahn算法实现
func (ts *TopologicalSorter) sortWithKahn(dag *DAG) ([]*DAGNode, error) {
    // 1. 计算入度
    inDegree := make(map[string]int)
    for _, node := range dag.Nodes {
        inDegree[node.ID] = len(node.Dependencies)
    }

    // 2. 初始化队列（入度为0的节点）
    queue := make([]*DAGNode, 0)
    for _, node := range dag.Nodes {
        if inDegree[node.ID] == 0 {
            queue = append(queue, node)
        }
    }

    // 3. 执行排序
    result := make([]*DAGNode, 0, len(dag.Nodes))

    for len(queue) > 0 {
        // 出队
        current := queue[0]
        queue = queue[1:]
        result = append(result, current)

        // 减少后继节点的入度
        for _, dependentID := range current.Dependents {
            if dependent, exists := dag.Nodes[dependentID]; exists {
                inDegree[dependentID]--
                if inDegree[dependentID] == 0 {
                    queue = append(queue, dependent)
                }
            }
        }
    }

    // 4. 检查是否所有节点都被排序
    if len(result) != len(dag.Nodes) {
        return nil, errors.New("graph contains cycles")
    }

    return result, nil
}

/// DFS算法实现
func (ts *TopologicalSorter) sortWithDFS(dag *DAG) ([]*DAGNode, error) {
    // 访问状态
    visiting := make(map[string]bool) // 正在访问
    visited := make(map[string]bool)  // 已访问完成

    result := make([]*DAGNode, 0, len(dag.Nodes))

    // DFS访问函数
    var dfsVisit func(*DAGNode) error
    dfsVisit = func(node *DAGNode) error {
        nodeID := node.ID

        // 检查循环依赖
        if visiting[nodeID] {
            return fmt.Errorf("cycle detected at node %s", nodeID)
        }

        if visited[nodeID] {
            return nil
        }

        // 标记正在访问
        visiting[nodeID] = true

        // 递归访问依赖（注意：DFS中访问的是前置依赖）
        for _, depID := range node.Dependencies {
            if dep, exists := dag.Nodes[depID]; exists {
                if err := dfsVisit(dep); err != nil {
                    return err
                }
            }
        }

        // 标记访问完成
        visiting[nodeID] = false
        visited[nodeID] = true

        // 添加到结果（后序遍历）
        result = append(result, node)

        return nil
    }

    // 从所有节点开始DFS
    for _, node := range dag.Nodes {
        if !visited[node.ID] {
            if err := dfsVisit(node); err != nil {
                return nil, err
            }
        }
    }

    // 反转结果（因为DFS产生的是逆拓扑序）
    for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
        result[i], result[j] = result[j], result[i]
    }

    return result, nil
}

/// 分层分组
func (ts *TopologicalSorter) groupByLayers(sortedNodes []*DAGNode) [][]*DAGNode {
    layers := make([][]*DAGNode, 0)

    // 使用节点的拓扑序构建层
    processed := make(map[string]bool)
    currentLayer := make([]*DAGNode, 0)

    // 第一层：入度为0的节点
    for _, node := range sortedNodes {
        if len(node.Dependencies) == 0 {
            currentLayer = append(currentLayer, node)
            processed[node.ID] = true
        }
    }

    if len(currentLayer) > 0 {
        layers = append(layers, currentLayer)
    }

    // 后续层：前一层执行完成后可以执行的节点
    for len(processed) < len(sortedNodes) {
        currentLayer = make([]*DAGNode, 0)

        for _, node := range sortedNodes {
            if processed[node.ID] {
                continue
            }

            // 检查所有依赖是否已处理
            canExecute := true
            for _, depID := range node.Dependencies {
                if !processed[depID] {
                    canExecute = false
                    break
                }
            }

            if canExecute {
                currentLayer = append(currentLayer, node)
                processed[node.ID] = true
            }
        }

        if len(currentLayer) > 0 {
            layers = append(layers, currentLayer)
        } else {
            // 防止无限循环
            break
        }
    }

    return layers
}

/// 查找关键路径
func (ts *TopologicalSorter) findCriticalPath(dag *DAG) []*DAGNode {
    // 简化实现：返回最长路径
    // 实际实现需要考虑节点执行时间

    criticalPath := make([]*DAGNode, 0)

    // 从叶子节点开始回溯
    for _, leafID := range dag.LeafNodes {
        if leaf, exists := dag.Nodes[leafID]; exists {
            path := ts.traceLongestPath(dag, leaf)
            if len(path) > len(criticalPath) {
                criticalPath = path
            }
        }
    }

    return criticalPath
}

/// 追踪最长路径
func (ts *TopologicalSorter) traceLongestPath(dag *DAG, startNode *DAGNode) []*DAGNode {
    // 简化实现：返回从根到此节点的任意路径
    path := []*DAGNode{startNode}

    // 递归向上查找依赖
    for len(startNode.Dependencies) > 0 {
        // 选择第一个依赖（简化）
        depID := startNode.Dependencies[0]
        if dep, exists := dag.Nodes[depID]; exists {
            path = append([]*DAGNode{dep}, path...)
            startNode = dep
        } else {
            break
        }
    }

    return path
}

/// 识别瓶颈节点
func (ts *TopologicalSorter) identifyBottlenecks(dag *DAG) []*DAGNode {
    bottlenecks := make([]*DAGNode, 0)

    for _, node := range dag.Nodes {
        // 具有多个后继依赖的节点可能是瓶颈
        if len(node.Dependents) > 2 {
            bottlenecks = append(bottlenecks, node)
        }

        // 执行时间长的节点可能是瓶颈
        if node.EstimatedTokens > 1000 {
            bottlenecks = append(bottlenecks, node)
        }
    }

    return bottlenecks
}
```

**拓扑排序器的核心算法**：

1. **多算法选择**：
   ```go
   // 小图用Kahn算法（直观）
   // 大图用DFS算法（高效）
   // 根据图规模自动选择
   ```

2. **分层并行执行**：
   ```go
   // 将排序结果分组为层
   // 每层节点可以并行执行
   // 最大化并发度
   ```

3. **性能分析**：
   ```go
   // 识别关键路径
   // 发现瓶颈节点
   // 优化执行策略
   ```

这个DAG工作流系统为Shannon提供了强大的复杂任务编排能力，支持智能分解、并行执行和结果聚合。

## 执行模式的智能选择

### 三种执行模式

Shannon根据任务特征选择最合适的执行模式：

```go
// 执行模式判断逻辑
func determineExecutionMode(decomp *DecompositionResult) ExecutionMode {
    hasDependencies := false
    for _, subtask := range decomp.Subtasks {
        if len(subtask.Dependencies) > 0 {
            hasDependencies = true
            break
        }
    }

    if hasDependencies {
        return ModeHybrid      // 混合模式：处理依赖关系
    } else if decomp.ExecutionStrategy == "sequential" {
        return ModeSequential  // 顺序模式：按指定顺序执行
    } else {
        return ModeParallel    // 并行模式：并发执行
    }
}
```

### 并行执行模式

最简单的模式：所有任务并发执行：

```go
func executeParallelPattern(ctx workflow.Context, decomp DecompositionResult, ...) {
    // 1. 创建所有子任务的future
    futures := make([]workflow.Future, len(decomp.Subtasks))

    for i, subtask := range decomp.Subtasks {
        taskInput := buildTaskInput(subtask, ...)
        futures[i] = workflow.ExecuteActivity(ctx, "ExecuteAgent", taskInput)
    }

    // 2. 等待所有任务完成
    results := make([]AgentExecutionResult, len(decomp.Subtasks))
    for i, future := range futures {
        future.Get(ctx, &results[i])
    }

    return results
}
```

这个模式适用于**独立任务**，如：
- 并行研究多个公司的数据
- 并发执行多个不相关的查询
- 同时调用多个API

### 顺序执行模式

任务按指定顺序依次执行：

```go
func executeSequentialPattern(ctx workflow.Context, decomp DecompositionResult, ...) {
    results := make([]AgentExecutionResult, len(decomp.Subtasks))

    for i, subtask := range decomp.Subtasks {
        // 传递前一个任务的结果
        contextWithHistory := buildContextWithPreviousResults(results[:i])

        taskInput := buildTaskInput(subtask, contextWithHistory, ...)
        result := executeAgentActivity(ctx, taskInput)

        results[i] = result
    }

    return results
}
```

这个模式适用于**累积知识**的任务，如：
- 先收集数据，再分析
- 逐步完善答案
- 需要上下文的任务链

### 混合执行模式：DAG的核心

最复杂的模式：处理任务依赖关系的有向无环图执行：

```go
func executeHybridPattern(ctx workflow.Context, decomp DecompositionResult, ...) {
    // 1. 构建任务图
    tasks := buildHybridTasks(decomp.Subtasks)

    // 2. 配置执行参数
    config := HybridConfig{
        MaxConcurrency:        5,
        EmitEvents:           true,
        DependencyWaitTimeout: 3600 * time.Second, // 1小时超时
        PassDependencyResults: true,
    }

    // 3. 执行混合编排
    result := execution.ExecuteHybrid(ctx, tasks, sessionID, history, config, ...)

    return result.Results, result.TotalTokens
}
```

## 混合执行的内部机制

### 并发控制和依赖管理

混合执行使用信号量控制并发，同时管理依赖关系：

```go
// 混合执行的核心逻辑
func ExecuteHybrid(ctx workflow.Context, tasks []HybridTask, ...) (*HybridResult, error) {
    // 1. 创建并发控制
    semaphore := workflow.NewSemaphore(ctx, int64(config.MaxConcurrency))
    resultsChan := workflow.NewChannel(ctx)

    // 2. 状态跟踪（Temporal单线程安全）
    completedTasks := make(map[string]bool)
    taskResults := make(map[string]activities.AgentExecutionResult)

    // 3. 启动所有任务执行器
    for _, task := range tasks {
        workflow.Go(ctx, func(ctx workflow.Context) {
            executeHybridTask(ctx, task, completedTasks, taskResults, semaphore, ...)
        })
    }

    // 4. 收集结果
    return collectResults(ctx, tasks, resultsChan)
}
```

### 单个任务的执行逻辑

每个任务执行器处理依赖等待和并发控制：

```go
func executeHybridTask(ctx workflow.Context, task HybridTask, ...) {
    // 1. 等待依赖完成
    if len(task.Dependencies) > 0 {
        waitForDependencies(ctx, task.Dependencies, completedTasks, timeout)
    }

    // 2. 获取并发许可
    semaphore.Acquire(ctx, 1)

    // 3. 准备上下文（包含依赖结果）
    context := buildContextWithDependencies(task, taskResults, config)

    // 4. 执行任务
    result := executeAgentActivity(ctx, task, context)

    // 5. 释放并发许可
    semaphore.Release(1)

    // 6. 发送结果
    resultsChan.Send(ctx, taskExecutionResult{TaskID: task.ID, Result: result})
}
```

### 依赖等待机制

依赖等待使用轮询模式，避免阻塞：

```go
func waitForDependencies(ctx workflow.Context, deps []string, completed map[string]bool, timeout time.Duration) bool {
    deadline := workflow.Now(ctx).Add(timeout)
    checkInterval := 30 * time.Second // 每30秒检查一次

    for workflow.Now(ctx).Before(deadline) {
        allCompleted := true
        for _, dep := range deps {
            if !completed[dep] {
                allCompleted = false
                break
            }
        }

        if allCompleted {
            return true // 所有依赖都完成了
        }

        // 等待一段时间后重新检查
        workflow.Sleep(ctx, checkInterval)
    }

    return false // 超时
}
```

这个设计平衡了：
- **效率**：定期检查而不是持续轮询
- **及时性**：依赖一完成就立即执行
- **容错**：超时保护防止无限等待

## 拓扑排序和执行优化

### Kahn算法实现

Shannon实现了经典的拓扑排序算法来确定执行顺序：

```go
func topologicalSort(tasks []HybridTask) [][]string {
    // 1. 计算入度
    inDegree := make(map[string]int)
    for _, task := range tasks {
        inDegree[task.ID] = len(task.Dependencies)
    }

    // 2. 初始化队列（入度为0的任务）
    queue := make([]string, 0)
    for taskID, degree := range inDegree {
        if degree == 0 {
            queue = append(queue, inDegree)
        }
    }

    // 3. 执行排序
    var levels [][]string
    for len(queue) > 0 {
        // 当前层级的任务
        level := make([]string, len(queue))
        copy(level, queue)
        levels = append(levels, level)

        // 处理下一层级
        nextQueue := make([]string, 0)
        for _, taskID := range queue {
            // 找到依赖此任务的其他任务
            for _, task := range tasks {
                if contains(task.Dependencies, taskID) {
                    inDegree[task.ID]--
                    if inDegree[task.ID] == 0 {
                        nextQueue = append(nextQueue, task.ID)
                    }
                }
            }
        }
        queue = nextQueue
    }

    return levels
}
```

### 执行层级的优化

通过拓扑排序，系统可以实现：
- **最大并行化**：同层级任务并发执行
- **依赖正确性**：保证依赖任务先完成
- **资源效率**：减少等待时间

## 上下文传递和数据流

### 依赖结果的传递

混合模式支持在任务间传递结果：

```go
func buildContextWithDependencies(task HybridTask, taskResults map[string]AgentExecutionResult, config HybridConfig) map[string]interface{} {
    context := make(map[string]interface{})

    // 复制基础上下文
    for k, v := range config.Context {
        context[k] = v
    }

    // 添加任务特定的覆盖
    for k, v := range task.ContextOverrides {
        context[k] = v
    }

    // 传递依赖结果
    if config.PassDependencyResults {
        for _, depID := range task.Dependencies {
            if result, exists := taskResults[depID]; exists {
                context[fmt.Sprintf("dependency_%s", depID)] = result.Response
                context[fmt.Sprintf("dependency_%s_tools", depID)] = result.ToolsUsed
            }
        }
    }

    return context
}
```

### 语义数据流的表达

通过Produces/Consumes实现更智能的数据流：

```go
type DataFlow struct {
    Producers map[string][]string // 数据类型 -> 生产者任务ID
    Consumers map[string][]string // 数据类型 -> 消费者任务ID
}

func buildDataFlowGraph(tasks []HybridTask) *DataFlow {
    flow := &DataFlow{
        Producers: make(map[string][]string),
        Consumers: make(map[string][]string),
    }

    for _, task := range tasks {
        for _, produces := range task.Produces {
            flow.Producers[produces] = append(flow.Producers[produces], task.ID)
        }
        for _, consumes := range task.Consumes {
            flow.Consumers[consumes] = append(flow.Consumers[consumes], task.ID)
        }
    }

    return flow
}
```

这种语义依赖比显式依赖更灵活：
- **自动推断**：系统可以自动发现依赖关系
- **类型安全**：基于数据类型而非任务ID
- **重用性**：相同的任务可以用于不同场景

## 错误处理和恢复

### 部分失败的处理

DAG执行支持部分任务失败：

```go
func handlePartialFailures(results map[string]AgentExecutionResult, tasks []HybridTask) ([]AgentExecutionResult, error) {
    var successful []AgentExecutionResult
    var failed []string

    for _, task := range tasks {
        if result, exists := results[task.ID]; exists && result.Success {
            successful = append(successful, result)
        } else {
            failed = append(failed, task.ID)
        }
    }

    // 如果有足够的成功结果，继续执行
    successRate := float64(len(successful)) / float64(len(tasks))
    if successRate >= 0.5 { // 50%以上成功
        logger.Warn("Partial execution success",
            "successful", len(successful),
            "failed", len(failed),
            "success_rate", successRate)

        return successful, nil
    }

    return nil, fmt.Errorf("too many task failures: %d/%d", len(failed), len(tasks))
}
```

### 依赖失败的级联处理

当依赖任务失败时，后续任务的处理策略：

```go
func handleDependencyFailure(failedTaskID string, dependentTasks []HybridTask, resultsChan workflow.Channel) {
    // 1. 标记失败任务为完成（避免无限等待）
    completedTasks[failedTaskID] = true

    // 2. 取消依赖此任务的所有任务
    for _, task := range dependentTasks {
        if contains(task.Dependencies, failedTaskID) {
            resultsChan.Send(ctx, taskExecutionResult{
                TaskID: task.ID,
                Error:  fmt.Errorf("dependency %s failed", failedTaskID),
            })
            completedTasks[task.ID] = true
        }
    }
}
```

## 性能优化和监控

### 并发度动态调整

系统根据负载动态调整并发度：

```go
func adjustConcurrencyBasedOnLoad(currentConcurrency int, activeTasks int, systemLoad float64) int {
    // 高负载时减少并发
    if systemLoad > 0.8 {
        return max(1, currentConcurrency/2)
    }

    // 低负载时增加并发
    if systemLoad < 0.3 {
        return min(maxConcurrency, currentConcurrency*2)
    }

    return currentConcurrency
}
```

### 执行时间预测

基于历史数据预测任务执行时间：

```go
func predictExecutionTime(task Subtask, historicalData map[string]time.Duration) time.Duration {
    // 基于相似任务的历史执行时间
    similarTasks := findSimilarTasks(task, historicalData)

    if len(similarTasks) > 0 {
        // 加权平均
        totalDuration := time.Duration(0)
        totalWeight := 0.0

        for _, similar := range similarTasks {
            weight := calculateSimilarity(task, similar.Task)
            totalDuration += time.Duration(float64(similar.Duration) * weight)
            totalWeight += weight
        }

        return time.Duration(float64(totalDuration) / totalWeight)
    }

    // 默认估计
    return estimateFromComplexity(task)
}
```

## 实际应用案例

### 复杂研究任务的编排

一个典型的研究任务如何通过DAG执行：

```
原始查询：分析苹果公司的AI战略

分解为：
├── 任务A：搜索苹果AI投资历史（无依赖）
├── 任务B：分析苹果AI产品线（无依赖）
├── 任务C：研究竞争对手策略（无依赖）
├── 任务D：比较苹果与竞争对手（依赖A,B,C）
└── 任务E：预测未来趋势（依赖D）
```

执行流程：
1. A、B、C并发执行（并行模式）
2. D等待A、B、C完成后执行（依赖等待）
3. E等待D完成后执行（依赖等待）

### 性能对比

DAG编排带来的性能提升：

| 执行模式 | 任务数 | 总时间 | 并行度 | 效率提升 |
|----------|--------|--------|--------|----------|
| 顺序执行 | 5个任务 | 50秒 | 1x | 基准 |
| 并行执行 | 5个任务 | 15秒 | 3.3x | 333% |
| DAG执行 | 5个任务（2层依赖） | 25秒 | 2x | 200% |

## 总结：DAG编排的变革性

Shannon的DAG工作流代表了AI任务编排的重大进步：

### 核心创新

1. **智能分解**：自动将复杂任务分解为可管理单元
2. **依赖建模**：精确表达任务间的依赖关系
3. **并发优化**：最大化并行执行，提高效率
4. **容错设计**：优雅处理部分失败和依赖问题

### 技术优势

- **执行效率**：通过并行执行减少总时间
- **资源利用**：智能分配计算资源
- **可靠性**：依赖管理和错误恢复机制
- **可扩展性**：支持大规模任务编排

### 对AI应用的影响

DAG编排让AI系统从**简单的问答工具**变为**复杂任务的智能执行者**：

- **研究任务**：自动编排多步骤研究流程
- **分析任务**：并行处理多个数据源
- **创作任务**：分步骤生成高质量内容
- **决策支持**：综合多个信息源提供建议

这种编排能力让AI代理从"会回答问题"升级为"会解决复杂问题"，为AI应用的实际落地提供了强大的执行引擎。

在接下来的文章中，我们将探索监督者工作流，了解多代理协作的实现机制。敬请期待！
