# F 题论文 LaTeX 初稿

入口：main.tex。正文分为七章与复算附录，图表直接引用 ../outputs/ 的现有 PDF，不另造数据。

## 编译

在本目录运行两遍：xelatex -interaction=nonstopmode -halt-on-error main.tex

要求 XeLaTeX、ctex 及 Windows 的宋体/黑体字体。本机未安装 XeLaTeX；已通过仓库的 GitHub Actions（运行 36106800567）在 TeX Live 2026 / Fandol 字体环境下编译出 main.pdf，共 20 页。已抽查封面、摘要、目录、正文图表缩略图和末页 AI 披露；全部章节与 15 张结果图路径存在。Windows 本地使用宋体/黑体再次编译时分页可能略有变化。

## 证据口径

- 四问正文按题目附件、现有代码输出及对应结果/验收报告写成。
- 第四问新增条件 90% 分位、固定任务标尺和 C4 小样本联合特征的审计；详见 reports/问题四/验证验收/IMAGE_SUGGESTION_AUDIT.md。
- 第一问推荐配比是后续训练实验候选；第二问的 B6 质量响应来自半合成数据；第三问固定该配比，只优化 N、D、Q_B；第四问不填无法识别的未来能力分数。
- 正式提交前须逐式逐表复核、填写真实团队信息，并用最终字体再次检查分页及当届赛制的封面要求。文末已披露 AI 工具参与范围。

## 主要结果来源

- ../reports/问题一/结果分析/RESULTS_REPORT_PROBLEM1.md
- ../reports/问题二/结果分析/RESULTS_REPORT_PROBLEM2.md
- ../reports/问题三/结果分析/RESULTS_REPORT_PROBLEM3.md
- ../reports/问题四/结果分析/RESULTS_REPORT_PROBLEM4.md

原始项目其他未跟踪文件不属于本论文稿。
