# Domain Feature Extraction & Evolution Prompt

## 任务目标
你将接收一批属于同一个域（Domain）的图片，以及当前图数据库中该域的已有结构。  
你需要：

1. 自动总结这些图片中的关键信息（Appearance、Identity、Quality、Content、Processing、Generator、RealData）。
2. 按严格 JSON Schema 输出：
   - domain
   - subtype
   - features（数组）
   - cypher（数组，仅新增，不覆盖旧节点）

## 输入结构（由后端组织）
{
  "domain_name": "...",
  "existing_schema": {...},
  "images": [
    {"path": "...", "base64": "..."},
    ...
  ]
}

## 输出 JSON 结构（必须严格符合）
```json
{
  "domain": "Appearance",
  "subtype": "Pose",
  "features": [
    {"name": "大幅度侧脸", "confidence": 0.94},
    {"name": "头部下倾", "confidence": 0.88}
  ],
  "cypher": [
    "MERGE (d:Domain {name:'外观条件域'})",
    "MERGE (s:Appearance {name:'Pose'})-[:APPEARANCE_CONDITION]->(d)",
    "MERGE (f:Feature {name:'大幅度侧脸'})-[:CHARACTER_OF]->(s)"
  ]
}

规则
若无新增内容，返回空 cypher。
仅允许新增节点，不得修改旧节点。
子域关系固定，如：
APPEARANCE_CONDITION
IDENTITY_ATTRIBUTE
QUALITY_RESOLUTION
CONTENT_OF
POST_PROCESSING_EDITING
GENERATOR_OF
第四层特征统一使用 CHARACTER_OF。
结果必须易于被 Neo4j 执行。
