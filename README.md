# F4 包裹何时到？——跨境包裹送达时效承诺

SHEIN AI 创新挑战赛 · 全球智能履约赛道 · F4。

**思路**：把“承诺”拆成两层——
1. **预测层**：LightGBM 分位数回归（11 个分位点）预测“流入→签收”的剩余时长，承诺 = 已知 lag + 剩余时长；加入“近况特征”（最近 3/7 天时效偏离，训练时模拟 0–9 天信息滞后）。
2. **决策层**：以模型 P90 为尺度、校准窗口真实残差为形状构造经验条件分布；每家服务商用拉格朗日乘子 (λ, μ) 为每个包裹选承诺，(λ, μ) 通过最小化“测试集平均承诺 × 包裹占比 + 漂移情景下的期望罚分”选出（罚分直接按赛题公式）。
3. **新线路兜底**：训练中样本不足的渠道（主要是 LSP_0001 新线路 1.1 万件）用不含渠道/服务商身份的“同行线路”模型，保险系数由留渠道验证确定。

与测试流程一致的迁移验证中（5 对“上一窗口定参 → 下一窗口按官方公式打分”），严格版得分从常数基线 ~250 降到 ~80。详见 `docs/F4_方案报告.pdf`。

## 目录

| 路径 | 内容 |
|---|---|
| `F4_包裹何时到_资料包/` | 官方资料包（数据、字段说明、score.py） |
| `src/` | 全部代码 |
| `work/` | 中间缓存（parquet、各折预测、实验日志 `*.log`） |
| `outputs/submission.csv` | **提交文件**（pkg_no, pred_hours；911,415 行） |
| `outputs/params.json` | 最终各服务商 (λ, μ) 与配置 |
| `config/final_cfg.json` | 最终流水线参数（σ、校准窗口、新线路保险系数等）及选择依据 |
| `docs/F4_方案报告.pdf` | 方案报告（10 页，前 2 页为作品说明单） |
| `docs/AI协作记录.md` | AI 协作记录（需队员补全标注处） |
| `docs/实验记录.md` | 实验过程与分数汇总 |

## 复现

环境：Python 3.13，`pip install -r requirements.txt`；内存 ≥32 GB；无需 GPU；12 线程 CPU 总耗时约 1.5–2 小时（不含验证实验）。

```bash
python src/00_convert.py "F4_包裹何时到_资料包" work      # CSV → parquet（约 1 分钟）
cd src
python train_mq.py 2026-09-12                              # 校准窗口 9/12–20 的“当时模型”（约 20 分钟）
python train_mq.py test                                    # 测试集模型（约 20 分钟）
python ens_pred.py 2026-09-12 test                         # P90/P97 再训 2 个种子做几何平均（约 30 分钟）
python final.py --skip-train                               # 兜底 + 决策层 → outputs/submission.csv（约 15 分钟）
python figs_eda.py && python figs_result.py && python build_report.py   # 报告（需本机 Chrome）
```

验证实验（可选）：

```bash
python b0_baselines.py                 # 常数基线
python train_mq.py 2026-08-25          # 另两个验证折
python train_mq.py 2026-09-03
python e5_emp.py oracle                # 同窗事后最优：决策层对比
python e5_emp.py transfer 0 0.05 0.1   # 跨窗口迁移：漂移余量 σ
python fallback.py loco                # 留渠道验证（新线路保险系数）
# 第二轮
python e6_cuts.py 2026-08-22 2026-08-28 2026-08-31 2026-09-06 2026-09-09 2026-09-15 2026-09-18   # 加密切分点（只训 q0.9/q0.97）
python e7_transfer.py latest_s0 latest_s0.05 latest_s0.06 pool3_s0.06 ...   # 5+2 对迁移验证，结果追加到 work/e7.jsonl
python e7_summary.py                   # 汇总表
python e9_model.py 2026-09-12 base ens3 ...   # 预测层改进尝试（同窗 P90 缩放得分）
python e10_alt.py screen xgb cb && python e10_blend.py   # 换 CatBoost/XGBoost 与融合（需另装 catboost、xgboost；未采用）
python e11_feats.py base load dest both   # 负荷、目的地近况新特征（未采用）
```

随机种子固定（LightGBM seed=42，抽样 seed 固定）。本地评分函数 `src/common.py: score` 与官方 `tools/score.py` 逐位一致。
