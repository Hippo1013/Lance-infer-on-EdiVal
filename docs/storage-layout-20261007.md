# 实验资产存储布局

## 存储职责

用户于2026-10-07确认：所有实验产物保存在服务器，本地只维护思路、方案、结论和必要位置说明。两机实验源码、环境、工具、运行状态及产物的实体文件均应位于各自 `/home/chs`；两机共用的模型权重和原始数据集可以长期存放于公共资源目录，并允许由/home/chs/model、dataset引用；实验源码、环境、工具和产物不得通过软链接转存公共盘。

## 实验归档位置

| 内容 | 主机与实体位置 |
| --- | --- |
| 六组全量RGB图片与JSON、四组评分、首答缓存、日志、队列、验收 | a800_0：`/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/sixrun_20261007/experiment` |
| Attention原件及各机生成断点 | receipt记录的生产机：`/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/sixrun_20261007/<run>/run/` |
| 冻结runtime | 两机：`/home/chs/exp0_attention/Lance-infer-on-EdiVal/runtime/sixrun_20261007_tokenregion_v3` |
| 迁移证据及路径映射 | a800_0项目：`outputs/sixrun_20261007/storage_migration/`，`locations.json`映射历史路径 |
| 模型、数据、环境和EdiVal工具 | 各机：`/home/chs/model`、`dataset`、`conda/envs`、`tools/EdiVal` |

完整归档复制与递归逐文件比对已通过，迁移后状态查询仍覆盖2644会话7928轮，2698队列项全部completed。1号机六个环境、EdiVal工具、三个benchmark数据集及模型权重的复制与逐文件比对均已通过，六个本机环境的基础导入检查通过，软链接已替换为实体目录。公共实验目录、setup、环境和工具副本均已清理；公共 `chs_umm_shared` 仅保留 `model` 与 `dataset`。两机实验软链接检查通过，90项冻结源码及归档审计证据hash一致，迁移临时目录已清理。完成记录位于0号机 `outputs/sixrun_20261007/storage_migration/completion.json`。

## 历史证据身份

迁移不改写原始JSON、日志、冻结源码中的历史绝对路径，不改变SHA256、fingerprint或首次裁判回答。读取指向原公共实验根目录的路径时，通过外部 `locations.json` 将前缀映射到0号机的新归档。Attention receipt中的主机与 `/home/chs` 路径未变。旧公共实验根目录不保留软链接；历史共享队列方案只作归档，不作为后续新实验的存储设计。

## 服务器原有资产

用户明确将Burn、网络代理、monitor、watch_dog及其既有环境、缓存和运行目录排除在本次实验迁移之外：保留原位置与配置。公共盘长期共用的模型权重和原始数据集同样允许保留；实验生成的数据与产物不属于基础资源例外。

本次已经复制到1号机的权重和数据集按用户确认保留；公共资源目录 `/media/damoxing/tangzecong/chs_umm_shared/model` 与 `dataset` 也保留供今后两机共用，不需要再次复制资源。本机实体副本是本次迁移结果，不作为以后必须复制权重的规则。
