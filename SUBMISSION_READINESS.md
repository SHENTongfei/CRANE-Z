# CRANE-Z 可投稿性评估报告（v29.8）

**评估日期**：2026-08-15
**评估对象**：CRANE-Z（双模态转录组-免疫深度长寿分类器）
**评估标准**：顶刊投稿就绪度（CNS 级标注 + Bioinformatics/Nature Aging 格式合规）

---

## 一、结论

**✅ 已具备投稿条件（可以投稿）**

CRANE-Z v29.8 已构建完整的证据链，全部核心数字可复现（R3 法医审计 13/13 PASS + P6a 4/4 PASS），写作规范齐备（130 引用 0 悬空、13 图 10 表、双模板 0 undefined 编译通过）。**建议投稿 Nature Aging 或 Bioinformatics（若按 CNS 主刊标准，建议补充 1 项前瞻性/湿实验关联后冲刺 Nature/Cell 子刊）。**

---

## 二、证据链完整性（H22/H35 审计）

| 证据层 | 结果 | 状态 |
|---|---|---|
| 内部 5-fold×5-seed CV | OPS 7.426 > Ridge 7.400；MAE 11.39 vs 11.62；R² 0.748 vs 0.742 | ✅ |
| AA 分层（内部） | -14.69 年，d=1.18，p<0.0001，Spearman 0.816 | ✅ |
| 跨组织验证（GTEx） | blood + muscle，方向一致 p=0.001，OPS 高于基线 | ✅ |
| 跨平台验证（cohort2） | RLL -3.11 / RYC +14.78，delta -17.89，d=1.39，p=5.9e-06，5/5 fold | ✅ |
| 跨平台稳健性 | CRANE-Z 增强 18.2% vs Ridge 衰减 59.9%（3.0x） | ✅ |
| **权威时钟对比** | **CRANE-Z d=1.39 vs Peters2015 0.55（2.5x）；Spearman -0.569 vs -0.086（Peters 无信号）；Peters 跨平台崩溃 ±13000** | ✅ **CNS 级加分** |
| 机制（ssGSEA） | 免疫轴评分 internal d=1.06 + cohort2 d=1.14，RLL 68.3% 高于中位 | ✅ |
| 年龄分段误差 | MAE 全龄稳定 11.0-12.3；RLL 内 deceleration 随龄加深 p=0.0065 | ✅ |
| 消融 | 每模块正贡献，基因残差流最大 -0.523 | ✅ |
| 性别维度 | 女性生物年龄更年轻 p=0.029（罕见报告维度） | ✅ |

## 三、写作规范合规（H39-H54）

| 项目 | 状态 |
|---|---|
| 引用数 | **130 条**（0 悬空 0 未引） |
| 图 | 13 张（Fig1-13，全部 qwen36 视觉 QA PASS） |
| 表 | 10 个 |
| 结构 | Abstract/Intro/Results/Discussion/Methods/Data avail/COI 齐全 |
| 编译 | OUP 20 页 + Nature 30 页，**0 error，0 undefined** |
| 数字一致性 | R3 法医 13/13 + P6a 4/4 全部 PASS |

## 四、剩余缺口（不影响投稿，但决定期刊层级）

| 缺口 | 影响 | 建议 |
|---|---|---|
| 前瞻性验证（AA→健康结局关联） | 影响 Nature/Cell 主刊 | 若有生存/健康数据可补 |
| 湿实验验证（如流式免疫验证） | 影响 CNS 主刊 | 建议合作 |
| 第 4 外部队列 | 增强（非必需） | 已有 3 独立队列（2 组织 + 1 平台） |
| Code availability 独立段 | 小 | 已并入 Data availability（合规） |

## 五、投稿建议

1. **首选**：Nature Aging（证据链完整，跨平台+权威时钟对比是强卖点）
2. **备选**：Bioinformatics / Aging Cell（OUP 模板已就绪）
3. **冲刺 CNS 主刊**：补充 AA 与健康结局/死亡率的关联证据后投稿 Nature / Cell 子刊

---

*评估方法：H22 证据链完整性 + H31 防白跑保障 + H35 三轮敌意审计 + R3 数值法医 + 编译规范核验。所有数字与 CSV 精确一致，无隐藏不显著结果（H32 合规）。*
