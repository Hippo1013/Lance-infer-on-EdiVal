# exp0状态与交接

## 当前基线与研究阶段

2026-10-07双协议六组推理与四组评分完成，2644会话7928轮、184.09 GB token／8×8区域attention；12:11:59工程终态验收passed。MICE16个首次回答组件仍待人工复核；自动遍历完成与人工校准分别报告。详细分数、分母和身份见[六组结果](sixrun-results-20261007.md)，实现、恢复规则见[实验说明](sixrun-20261007.md)。六组v1 VAE空间标签的已知缺陷和数值验收遗漏见[attention观测](attention-observation.md#vae空间标签的已知缺陷)。

EdiVal v2观测与七项描述分析已于2026-10-10完成。572会话1716轮固定首次历史的输出逐像素复现；前三项17张表、1716份派生数组和12组图文，以及后四项6864组配对、61776项既有标量复现和14组新增图文均通过验收。任务身份、源码及最终回执位置见[Attention分析](attention-analysis.md)。其他bench未补采；编辑成败关联、干预与训练尚未开展。

完整归档位于 `a800_0:/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/sixrun_20261007/experiment`，attention留在receipt指定生产机。迁移逐文件比对、冻结身份与基础环境导入检查均通过；旧绝对路径按外部映射解析，见 [存储布局](storage-layout-20261007.md)。两机/home/chs互不共享，公共盘仅可保留共用模型／原始数据资源，本项目环境、工具和产物不放公共盘。Burn与代理等原有资产保持原位置。

六组worker／supervisor已退出，四卡burn于终态时恢复，heartbeat `lance` 已暂停；实时GPU和进程状态须现场核查，不能凭本页推断。历史 `mice-v2`、`edival-gpu1` 也已暂停。

## 归档查询与验收边界

六组队列只在0号机的新归档根查询，具体status命令见实验说明。已完成目录不能使用启动命令重新生成；完整图像、attention、裁判首答、日志、数据库及清单留服务器。本地维护源码、方案、结论与必要位置说明。

最终验收覆盖全部会话／轮次、真实历史、7928份attention、90项冻结源码、评分来源／首答／依赖hash与汇总。工程通过不替代能力得分，MICE缺项不补零，人工待审不重新采样。未来人工补评须创建独立快照、保留自动原件；新提示词或干预须建立独立实验身份。

## 历史实验索引

以下路径相对于0号机项目，均为独立历史运行；不能将旧人工分数或旧页面套用到六组实验。

| 内容 | 结果与入口 | 服务器路径 |
| --- | --- | --- |
| MICE bare，2026-10-04 | 720会话2160轮，validation passed | `outputs/mice/full_bare_20261004_attention/` |
| ImgEdit bare，2026-10-04 | 30会话88轮，validation passed；旧人工表88/88明确判定 | `outputs/imgedit/full_bare_20261004_attention/` |
| EdiVal bare，2026-10-06 | 572会话1716轮，512／prefix；推理与HTTP查看复核passed | `outputs/edival/full_512_bare_20261005_attention/` |
| MICE单票v3及人工第一版 | 31项人工补全，IF／CC／累计GA=41.7557%／67.7011%／27.2685%，分母2153／1883／2160；[结果](mice-first-edition-results.md) | `outputs/scoring/mice_first_edition_human_v1/` |
| EdiVal官方历史评分 | 572会话1716轮、2288张HPS图；[原分量](edival-scoring-results.md)、[论文口径](edival-paper-tables.md) | `outputs/scoring/edival_official_full_20261006/` |

旧查看页8772对应已补评的31项，8768对应旧MICE bare抽查，8769对应旧ImgEdit人工表，8770对应旧EdiVal全量。恢复及身份见 [文档索引](README.md)，不宣称这些页已展示六组新结果。

## 源码与部署身份

开发工作树、GitHub提交、服务器部署与冻结runtime分开核对；HEAD一致不表示工作树或远端部署一致。2026-10-06的逐项部署审计属于历史快照，同步检查方法见[源码同步](source-sync.md)。冻结runtime不得覆盖，首次产物不得因当前源码变化重建。

四份历史 [源码索引](../versions/README.md) 的64项引用已按原SHA256补齐独立源码；六组冻结版本位于两机 `runtime/sixrun_20261007_tokenregion_v3`，90项原身份由归档plan和最终audit绑定。历史GPFS队列仅作归档；未来多机调度需在服务器/home/chs内设计通信与状态，不复用公共实验根。GPFS flock单机锁的事故与原子目录lease修复说明见六组实验文档。

## 后续研究

后续沿用已确定的输入构造，在已完成的七项描述分析上，确定干预问题与预算，并研究任务需求及编辑表现与分布的关系。区分当前轮IF、历史全局约束GA、保持性CC与裁判未决；相关性不能直接证明attention因果机制。ViT空间偏向的机制仍需位置／内容控制实验；干预层／步骤／token或区域范围、样本和预算另定。研究课题与用户工作日志继续由用户维护。
