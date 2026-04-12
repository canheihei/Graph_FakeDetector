# Cleaned Baseline Graph Cypher

以下 Cypher 用于创建清洗后的基础域泛化知识图谱基线。

清洗原则：
- 保留三层结构 `MainDomain <- SpecificDomain <- SubDomain`
- 历史近义脏节点收敛到统一 ontology
- 补充 `display_name` / `canonical_name` / `semantic_source` / `semantic_version`
- 将 `面部比例失调` 迁移到 `内容异常域`
- 将历史粗粒度节点统一升级为更专业的属性表达

清洗后结构规模：
- `1` 个 `MainDomain`
- `7` 个 `SpecificDomain`
- `22` 个 `SubDomain`
- 共 `30` 个节点、`29` 条关系

```cypher
// 1. 批量创建所有节点（三层结构，统一保留 main_id / specific_id / sub_id）

CREATE
// 第一层：中心节点
(main:MainDomain {
  main_id: 'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d',
  name: '域泛化',
  describe: '面向虚假人脸检测的域泛化知识图谱中心节点，涵盖生成机制、内容异常、后处理痕迹、身份偏移、外观扰动及质量退化等维度'
}),

// 第二层：各大领域（SpecificDomain）
(gen_domain:SpecificDomain {
  specific_id: 'b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e',
  name: '生成机制域',
  describe: '描述虚假人脸图像的生成技术来源，影响底层纹理与统计特性。'
}),
(real_domain:SpecificDomain {
  specific_id: 'c3d4e5f6-a7b8-4c9d-0e1f-2a3b4c5d6e7f',
  name: '真实数据扰动域',
  describe: '源自真实人脸但在采集、传输或成像链路中引入退化，可能被伪造模型误用或与伪造痕迹发生混淆。'
}),
(content_domain:SpecificDomain {
  specific_id: 'd4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f8a',
  name: '内容异常域',
  describe: '图像内容层面的结构性不合理现象，通常表现为几何比例、对称性或器官拓扑与真实人脸先验不一致。'
}),
(post_domain:SpecificDomain {
  specific_id: 'e5f6a7b8-c9d0-4e1f-2a3b-4c5d6e7f8a9b',
  name: '后处理痕迹域',
  describe: '生成后或融合阶段的人为编辑操作留下的压缩、裁剪、平滑和边界残留痕迹。'
}),
(identity_domain:SpecificDomain {
  specific_id: 'f6a7b8c9-d0e1-4f2a-3b4c-5d6e7f8a9b0c',
  name: '身份属性偏移域',
  describe: '伪造过程中身份相关属性在年龄、性别、族裔和生理层面的不一致性或分布偏移。'
}),
(appear_domain:SpecificDomain {
  specific_id: 'a7b8c9d0-e1f2-4a3b-4c5d-6e7f8a9b0c1d',
  name: '外观扰动域',
  describe: '姿态、表情、遮挡和光照等外观条件在生成中的不自然组合或物理一致性破坏。'
}),
(quality_domain:SpecificDomain {
  specific_id: 'b8c9d0e1-f2a3-4b4c-5d6e-7f8a9b0c1d2e',
  name: '质量与分辨率域',
  describe: '图像质量退化或分辨率异常，反映生成或传输链路中的上采样、压缩和纹理失真问题。'
}),

// 第三层：各子域（SubDomain）
// -- 生成机制域
(face_spec:SubDomain {
  sub_id: 'c9d0e1f2-a3b4-4c5d-6e7f-8a9b0c1d2e3f',
  name: '人脸专用生成器',
  display_name: '人脸专用生成器',
  canonical_name: 'face_specific_generator',
  describe: '专用于人脸合成的生成模型，通常具备较高保真度，但仍可能在局部纹理与高频分布上残留特定生成伪影。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(gan:SubDomain {
  sub_id: 'd0e1f2a3-b4c5-4d6e-7f8a-9b0c1d2e3f4a',
  name: 'GAN类生成器',
  display_name: 'GAN类生成器',
  canonical_name: 'gan_generator_family',
  describe: '基于对抗训练的生成器，常伴随棋盘纹、模式崩溃和局部纹理重复等问题。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(diffusion:SubDomain {
  sub_id: 'e1f2a3b4-c5d6-4e7f-8a9b-0c1d2e3f4a5b',
  name: '扩散类生成器',
  display_name: '扩散类生成器',
  canonical_name: 'diffusion_generator_family',
  describe: '基于扩散过程的生成器，整体细节更自然，但可能在边界过渡和频域统计上残留扩散式噪声模式。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),

// -- 真实数据扰动域
(candid:SubDomain {
  sub_id: 'f2a3b4c5-d6e7-4f8a-9b0c-1d2e3f4a5b6c',
  name: '非合作抓拍退化',
  display_name: '非合作抓拍退化',
  canonical_name: 'unconstrained_capture_degradation',
  describe: '真实监控或街拍图像中的模糊、低分辨率和姿态不稳定退化，常作为负样本干扰检测器判别。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(blur:SubDomain {
  sub_id: 'a3b4c5d6-e7f8-4a9b-0c1d-2e3f4a5b6c7d',
  name: '运动/失焦模糊',
  display_name: '运动/失焦模糊',
  canonical_name: 'motion_or_defocus_blur',
  describe: '由相机抖动或对焦失败引起的模糊，会掩盖高频异常并削弱局部伪造痕迹可见性。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(low_light:SubDomain {
  sub_id: 'b4c5d6e7-f8a9-4b0c-1d2e-3f4a5b6c7d8e',
  name: '低光照噪声',
  display_name: '低光照噪声',
  canonical_name: 'low_light_sensor_noise',
  describe: '高 ISO 与低照度成像条件引入的噪声和细节丢失，容易与伪造高频残差发生混淆。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),

// -- 内容异常域
(asymmetry:SubDomain {
  sub_id: 'c5d6e7f8-a9b0-4c1d-2e3f-4a5b6c7d8e9f',
  name: '五官不对称异常',
  display_name: '五官不对称异常',
  canonical_name: 'facial_asymmetry_abnormality',
  describe: '面部左右结构、比例或纹理分布出现不自然偏差，违反真实人脸的整体对称先验。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(proportion:SubDomain {
  sub_id: 'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c6b',
  name: '面部比例失调',
  display_name: '面部比例失调',
  canonical_name: 'facial_proportion_distortion',
  describe: '五官相对位置、三庭五眼比例或局部几何关系出现系统性失衡，反映身份重构阶段的结构先验破坏。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),

// -- 后处理痕迹域
(jpeg:SubDomain {
  sub_id: 'd6e7f8a9-b0c1-4d2e-3f4a-5b6c7d8e9f0a',
  name: 'JPEG压缩伪影',
  display_name: 'JPEG压缩伪影',
  canonical_name: 'jpeg_compression_artifact',
  describe: '块效应、振铃效应和量化失真等压缩痕迹，常在保存或传播过程中进一步强化。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(beauty:SubDomain {
  sub_id: 'e7f8a9b0-c1d2-4e3f-4a5b-6c7d8e9f0a1b',
  name: '美颜滤镜平滑',
  display_name: '美颜滤镜平滑',
  canonical_name: 'beauty_filter_smoothing',
  describe: '过度平滑、锐化或磨皮操作破坏自然皮肤纹理和频谱分布，掩盖真实细节层次。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(boundary:SubDomain {
  sub_id: 'f8a9b0c1-d2e3-4f4a-5b6c-7d8e9f0a1b2c',
  name: '边界融合不连续',
  display_name: '边界融合不连续',
  canonical_name: 'boundary_blending_discontinuity',
  describe: '人脸与头发、皮肤或背景交界处出现颜色场、纹理场或透明度过渡的不连续，常见于局部替换与融合失败。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(crop:SubDomain {
  sub_id: 'a9b0c1d2-e3f4-4a5b-6c7d-8e9f0a1b2c3d',
  name: '非自然裁剪',
  display_name: '非自然裁剪',
  canonical_name: 'unnatural_cropping_pattern',
  describe: '异常宽高比、边界截断或构图不连续会暴露局部合成区域或后处理裁切行为。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),

// -- 身份属性偏移域
(ethnicity:SubDomain {
  sub_id: 'b0c1d2e3-f4a5-4b6c-7d8e-9f0a1b2c3d4e',
  name: '族裔特征混叠',
  display_name: '族裔特征混叠',
  canonical_name: 'identity_ethnicity_feature_conflict',
  describe: '面部的族裔线索在跨身份合成中被异常混合，导致肤色、骨相和局部纹理组合出现不自然冲突。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(gender:SubDomain {
  sub_id: 'c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f',
  name: '性别属性冲突',
  display_name: '性别属性冲突',
  canonical_name: 'identity_gender_attribute_conflict',
  describe: '面部的性别表达与目标身份属性之间出现冲突，通常体现在性别二态特征被混合、弱化或错误迁移。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(age:SubDomain {
  sub_id: 'd2e3f4a5-b6c7-4d8e-9f0a-1b2c3d4e5f6a',
  name: '年龄属性偏移',
  display_name: '年龄属性偏移',
  canonical_name: 'identity_age_attribute_shift',
  describe: '年龄相关线索与整体身份语义不一致，常表现为皮肤状态、面部成熟度和骨相年龄感的异常重写。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(physiology:SubDomain {
  sub_id: 'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5f',
  name: '生理属性错配',
  display_name: '生理属性错配',
  canonical_name: 'identity_physiology_attribute_conflict',
  describe: '面部生理属性与身份表征之间不一致，常见于成熟度、器官比例或人体属性被异常组合。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(identity_boundary:SubDomain {
  sub_id: 'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c6a',
  name: '身份边界模糊',
  display_name: '身份边界模糊',
  canonical_name: 'identity_boundary_ambiguity',
  describe: '身份主体边界缺乏清晰稳定的语义约束，导致多个身份特征在同一张脸上呈现模糊归属。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),

// -- 外观扰动域
(pose:SubDomain {
  sub_id: 'e3f4a5b6-c7d8-4e9f-0a1b-2c3d4e5f6a7b',
  name: '极端姿态结构失真',
  display_name: '极端姿态结构失真',
  canonical_name: 'extreme_pose_structural_distortion',
  describe: '大角度偏转、俯仰或侧脸条件下，五官拓扑、轮廓连接和局部投影关系出现结构性失真。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(expression:SubDomain {
  sub_id: 'f4a5b6c7-d8e9-4f0a-1b2c-3d4e5f6a7b8c',
  name: '表情肌肉不协调',
  display_name: '表情肌肉不协调',
  canonical_name: 'expression_muscle_incoordination',
  describe: '表情驱动时局部肌肉收缩关系不自然，如微笑时眼周无联动或口周张力异常。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(occlusion:SubDomain {
  sub_id: 'a5b6c7d8-e9f0-4a1b-2c3d-4e5f6a7b8c9d',
  name: '遮挡合成伪影',
  display_name: '遮挡合成伪影',
  canonical_name: 'occlusion_compositing_artifact',
  describe: '眼镜、口罩和头发等遮挡物与人脸边界之间出现融合错误、轮廓断裂或颜色混合异常。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),
(lighting:SubDomain {
  sub_id: 'b6c7d8e9-f0a1-4b2c-3d4e-5f6a7b8c9d0e',
  name: '光照一致性冲突',
  display_name: '光照一致性冲突',
  canonical_name: 'illumination_consistency_conflict',
  describe: '面部不同区域的受光方向、阴影衰减或高光分布不一致，违反单一物理光照条件下的整体一致性。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
}),

// -- 质量与分辨率域
(high_res:SubDomain {
  sub_id: 'c7d8e9f0-a1b2-4c3d-4e5f-6a7b8c9d0e1f',
  name: '虚假高分辨率',
  display_name: '虚假高分辨率',
  canonical_name: 'synthetic_high_resolution_artifact',
  describe: '通过上采样或伪细节生成制造的高分辨率假象，高频细节缺乏真实自然性或呈现重复纹理。',
  semantic_source: 'baseline_ontology_migration',
  semantic_version: 'graph_semantics_v3_baseline'
})

// 2. 批量建立关系

CREATE
// 第二层 -> 第一层
(gen_domain)-[:KINDS_OF]->(main),
(real_domain)-[:KINDS_OF]->(main),
(content_domain)-[:KINDS_OF]->(main),
(post_domain)-[:KINDS_OF]->(main),
(identity_domain)-[:KINDS_OF]->(main),
(appear_domain)-[:KINDS_OF]->(main),
(quality_domain)-[:KINDS_OF]->(main),

// 第三层 -> 第二层
(face_spec)-[:SPECIFIC_OF]->(gen_domain),
(gan)-[:SPECIFIC_OF]->(gen_domain),
(diffusion)-[:SPECIFIC_OF]->(gen_domain),

(candid)-[:SPECIFIC_OF]->(real_domain),
(blur)-[:SPECIFIC_OF]->(real_domain),
(low_light)-[:SPECIFIC_OF]->(real_domain),

(asymmetry)-[:SPECIFIC_OF]->(content_domain),
(proportion)-[:SPECIFIC_OF]->(content_domain),

(jpeg)-[:SPECIFIC_OF]->(post_domain),
(beauty)-[:SPECIFIC_OF]->(post_domain),
(boundary)-[:SPECIFIC_OF]->(post_domain),
(crop)-[:SPECIFIC_OF]->(post_domain),

(ethnicity)-[:SPECIFIC_OF]->(identity_domain),
(gender)-[:SPECIFIC_OF]->(identity_domain),
(age)-[:SPECIFIC_OF]->(identity_domain),
(physiology)-[:SPECIFIC_OF]->(identity_domain),
(identity_boundary)-[:SPECIFIC_OF]->(identity_domain),

(pose)-[:SPECIFIC_OF]->(appear_domain),
(expression)-[:SPECIFIC_OF]->(appear_domain),
(occlusion)-[:SPECIFIC_OF]->(appear_domain),
(lighting)-[:SPECIFIC_OF]->(appear_domain),

(high_res)-[:SPECIFIC_OF]->(quality_domain)

// 3. 返回结果
RETURN "清洗后的域泛化基础知识图谱创建完成：1个MainDomain、7个SpecificDomain、22个SubDomain，共30个节点和29个关系" AS result;
```
