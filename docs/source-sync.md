# 源码同步与部署边界

GitHub维护代码、配置、测试、页面、分析程序及必要说明。模型、原始数据、生成图、attention、数值产物、评分数据库、运行日志和验收清单保存在服务器；本地UMM的课题、研究报告、最终图表、管理规则与助手日志不整体发布。

## 版本核对

| 状态 | 核对方法 | 含义 |
| --- | --- | --- |
| 本地开发工作树 | `git status --short`、`git diff`及未跟踪文件 | 包括HEAD之外的开发内容；HEAD相同不表示开发内容已同步 |
| GitHub发布版本 | `git fetch origin`，比较`HEAD`与`origin/main`及`git ls-remote origin refs/heads/main` | 确认提交已发布，不证明服务器已部署 |
| 服务器开发目录 | 对明确部署文件核对路径与SHA256，另记录Git ref（若有） | 服务器可能单文件部署，Git ref不能代替实际内容 |
| 冻结runtime | 原manifest及首次实验身份、逐文件SHA256 | 保持原字节；开发或发布更新不能覆盖它 |

提交前审阅全部已跟踪修改和新增文件，按目录明确加入代码及必要说明，检查排除目录和敏感内容，再运行CPU检查。提交后从该提交的独立导出检查依赖、文档链接与测试，避免只在带未跟踪文件的开发目录中通过。推送前再次fetch；若main出现新提交，先核对差异，不强推、不覆盖他人修改。

## 克隆与CPU验证

安装所需Python版本与依赖以`pyproject.toml`为准。在独立环境中执行`python -m pip install -e .`，然后在仓库根运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m unittest discover -s tests -v
```

attention分析的额外数学检查见[分析入口](attention-analysis.md#本地cpu检查)。本地CPU通过不替代CUDA／模型验收或真实数据覆盖。运行实验还需[资源清单](experiment-resources.md)、已验收环境与独立的新任务身份。

## 服务器更新

GitHub推送与服务器部署分别授权、分别验收。部署前明确目标机、目录、文件集合和原状态，核对运行进程；只更新授权的开发目录，核对实际文件哈希。不执行整目录覆盖，不改写`runtime/`、原始输出和首次裁判回答。历史复现使用[版本索引](../versions/README.md)或服务器冻结清单，而不是最新main。

两机同名`/home/chs`不是共享存储。六组历史归档与产物位置见[存储布局](storage-layout-20261007.md)，当前EdiVal v2及分析任务见[分析入口](attention-analysis.md)。历史共享GPFS队列不作为新实验调度目录。
