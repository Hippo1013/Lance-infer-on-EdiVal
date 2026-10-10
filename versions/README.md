# 实验源码版本索引

这些清单用于把已完成实验的方法对应到确切实现，不存实验数据。每份 `manifest.json` 的 `files` 以实验原路径为键，给出SHA256和本地 `local_source`。初次补齐五份历史源码；2026-10-07进一步修复13条受开发更新影响的引用，原字节固定在各版本的source目录，其余仅引用仍匹配原SHA256的现有文件。四清单共64项引用已重新通过哈希核查。

| 版本目录 | 实验范围 | 文件数 |
| --- | --- | ---: |
| `mice_imgedit_bare_20261004` | MICE / ImgEdit完整历史bare推理 | 19 |
| `edival_512_20261005` | EdiVal 512推理 | 24 |
| `mice_qwen_once_v3` | MICE历史单票v3评分 | 17 |
| `edival_scoring_official_20261006_v2` | EdiVal官方评分 | 4 |

重建时按manifest将各 `local_source` 复制为对应原路径，再检查全部SHA256。外部模型、数据、包依赖按 [资源清单](../docs/experiment-resources.md) 准备；这些清单只覆盖实验冻结记录中的项目源码，不包含完整第三方运行环境。

修改被manifest引用的现有文件前，先把原字节移入对应版本目录，将 `local_source` 改为新位置而保持历史SHA256不变，再修改开发文件。版本来源和运行身份不得随当前开发版漂移。服务器运行中的冻结目录不覆盖。

六组实验的90项冻结源码位于两机 `/home/chs/exp0_attention/Lance-infer-on-EdiVal/runtime/sixrun_20261007_tokenregion_v3`，原身份由0号机归档plan及最终audit绑定，见 [六组说明](../docs/sixrun-20261007.md)。本索引的四份清单不冒充六组源码清单。
