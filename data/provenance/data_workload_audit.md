# 数据来源与工作量核实（2026-10-25）

正式 cohort 的 **53 个唯一 Dust2 DEMO 合计 25,818,929,645 字节，即 25.818930 GB / 24.045752 GiB**。当前两个 raw 目录恰好只有这 53 个 DEMO；文件存在、体积与 manifest、已归档 raw fingerprints 一致。53 个归档 SHA 各不相同。本次没有为强调工作量重算 25.8 GB 原始哈希；原准备流程已核验它们。

来源是 `blanchon/opencs2_dataset_demo` 的固定 revision `912a363098feec3f220e41d68634d2f71676c28b`，通过 Hugging Face 下载；上游收集自 HLTV。每场的原 HLTV match URL、demo URL 与本次镜像下载 URL 均在 `source_manifest.csv`。不能写成研究者直接爬取了 HLTV，也不能把项目目录约 29 GB 写成下载了 30 GB 原始录像。

覆盖 31 个赛事标签，日期 2026-03-23 19:29:37+00:00 至 2026-04-20 18:19:28+00:00。下表按 manifest 的原始 event 标签统计，31 个标签不等于31个独立且同级别的联赛。

| 来源赛事标签 | DEMO 数 | 原始体积 GB |
|---|---:|---:|
| BC Game Masters Championship Season 1 | 1 | 0.554 |
| BetBoom Storm Season 2 | 3 | 1.080 |
| CCT Season 3 Europe Series 19 | 2 | 0.903 |
| CCT Season 3 Europe Series 20 | 1 | 0.408 |
| CCT Season 3 Europe Series 20 Closed Qualifier | 1 | 0.522 |
| CCT Season 3 North America Series 4 | 1 | 0.420 |
| CCT Season 3 South America Series 10 | 2 | 0.684 |
| Digital Crusade DraculaN Season 6 | 2 | 0.831 |
| Dust2.us Eagle Masters Series 7 | 2 | 1.212 |
| ESEA Season 56 Europe Finals | 2 | 1.025 |
| ESL Challenger League Season 51 Asia-Pacific Cup 3 | 2 | 1.440 |
| ESL Challenger League Season 51 Europe Cup 2 | 1 | 0.306 |
| ESL Challenger League Season 51 Europe Cup 3 | 2 | 0.766 |
| ESL Challenger League Season 51 North America Cup 3 | 2 | 1.455 |
| ESL Challenger League Season 51 Oceania Cup 3 | 2 | 0.917 |
| Gamers Assembly Festival 2026 | 1 | 0.267 |
| HLC Belgrade PRO 2026 | 2 | 0.863 |
| IEM Rio 2026 | 6 | 2.693 |
| LORGAR RANKINGS Season 1 Closed Qualifier | 2 | 0.906 |
| NODWIN Clutch Series 6 | 2 | 1.141 |
| NODWIN Clutch Series 7 | 1 | 0.553 |
| NODWIN Clutch Series 7 Closed Qualifier | 2 | 1.300 |
| PGL Astana 2026 Europe Closed Qualifier | 1 | 0.573 |
| PGL Astana 2026 South America Closed Qualifier | 1 | 0.592 |
| PGL Bucharest 2026 | 2 | 1.055 |
| Parken Challenger Championship Season 3 | 1 | 0.457 |
| Parken Challenger Championship Season 5 | 2 | 0.830 |
| Tipsport Conquest of Prague 2026 | 1 | 0.426 |
| United21 League Season 46 | 1 | 0.572 |
| United21 League Season 47 | 1 | 0.633 |
| Yuqilin Pinnacle of Battle Season 3 | 1 | 0.438 |

经过解析、有效完整回合筛选、公共事件时钟修正、雷达代理融合、队友状态/烟雾/声音历史编码、训练集空间分区和匿名人数标签构造，得到 **275,577 帧、1,127 小回合、98,203 完整历史前缀，每帧 311 维**。训练/验证/测试为 34/5/14 场、707/127/293 回合、60,338/10,831/27,034 前缀。比赛内前缀相关，不是98,203个独立比赛。

上游数据卡为 CC BY 4.0：https://huggingface.co/datasets/blanchon/opencs2_dataset_demo 。代码采用独立 MIT 许可；衍生数值数据保留来源、CC BY 4.0 标注和游戏/赛事第三方权利说明。公开包提供processed数据而不重复分发原始DEMO。当前尚未公开GitHub；已按用户要求准备本地可上传目录。

适合论文表述：*We curate a task-specific occupancy dataset from 53 professional-event Dust2 replays, totaling 25.82 GB of raw DEMO files, retrieved from a version-pinned OpenCS2 mirror with original HLTV provenance. Our processing pipeline yields 1,127 valid rounds, 275,577 observation frames and 98,203 full-history prefixes with 311 features per frame.*
