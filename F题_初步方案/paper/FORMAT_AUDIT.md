# 2026 年华为杯论文排版核查

依据：[本届论文格式规范](https://www.cmathc.org.cn/cpmcm/news/644.html)与[开赛公告](https://www.cmathc.org.cn/cpmcm/news/642.html)。此文只记录可由当前源文件和已编译 PDF 核对的项目；官方 Word 模板本体尚未纳入仓库，不能宣称封皮与其逐像素一致。

| 项目 | 当前实现 | 核查方式 |
| --- | --- | --- |
| 首页封皮；摘要页起页码 1 | `main.tex` 的 `coverpage` 在封皮后将页码设为 1 | 渲染前两页 |
| 除封皮外无身份信息 | 学校、队号、姓名仅作为封皮空栏；正文无填写值 | 搜索 `main.tex` 和逐页渲染 |
| 无页眉；页脚居中阿拉伯数字 | `\pagestyle{plain}`，封皮 `\thispagestyle{empty}` | 渲染目录、正文和末页 |
| 题目三号黑体、一级标题四号黑体 | 16 pt 与 14 pt 的 `\heiti` | 检查 LaTeX 定义和渲染 |
| 正文小四号、单倍行距 | `ctexart` 12 pt、`\linespread{1}` | 检查 LaTeX 定义 |
| 摘要、关键词在摘要页，正文从下一页开始 | `abstract` 后 `\newpage` | 渲染摘要和正文首页 |
| PDF 文字可检索 | XeTeX ActualText 开启 | 用 PDF 文本提取器核验中文字符数 |

正式提交前还须用组委会附件 3 原模板核对封皮四个 logo、表单位置和摘要页固定版式；填入真实队伍信息后重新编译并复核页码。最终 PDF 的 MD5 一经提交不得再修改。
