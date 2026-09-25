# 今日头条自动发文工作流 v2.0

> 拟人化操作的今日头条自动发文工作流：Playwright 驱动真实浏览器 + 多账号 Cookie 池，模拟真人完成登录、写标题、生成正文、智能配图、择时发布的全流程；内置中文向量去重与标题 A/B 效果学习。

## 项目简介

与调用后台接口的机器人不同，本项目用 Playwright 驱动本机真实的 Edge 浏览器，以真人登录获得的 Cookie 操作今日头条创作者平台（mp.toutiao.com），整个发布过程的行为特征与人工操作一致。

工作流模拟一位真实编辑的完整工作节奏：

| 步骤 | 说明 |
|------|------|
| 1. 选题定标题 | DeepSeek 生成候选标题，经中文语义向量库（bge-large-zh）查重去重，避免与历史文章撞题 |
| 2. 撰写正文 | 按预设文风生成 800–1200 字、适合手机阅读的短句流文章 |
| 3. 智能配图 | 分析文章段落场景，从 Unsplash / Pixabay 按文意插入配图 |
| 4. 择时发布 | 只在 20:00–21:59 的晚间窗口发布，模拟真人作息 |
| 5. 多账号轮换 | Cookie 池管理最多 5 个账号，按使用频率自动调度 |
| 6. 效果回流 | 标题 A/B 性能追踪记录历史表现，反哺后续标题生成 |

## 核心模块

- **向量数据库模块**（`EmbeddingProvider` / `VectorStore`）—— 标题语义去重，支持 SentenceTransformer 与 TF-IDF 降级
- **标题生成模块**（`TitleGenerator` / `TitlePerformanceTracker`）—— 多策略生成 + 历史表现学习
- **Cookie 池模块**（`CookiePool` / `MultiAccountPublisher`）—— 多账号管理与轮换发布
- **智能配图模块**（`SmartImageInserter`）—— 场景识别 + 图库检索插入
- **主工作流**（`ToutiaoWorkflow`）—— 整合上述模块的完整发布流水线

## 环境要求

- Windows 10/11（依赖本机 Edge 浏览器）
- Python 3.10+
- 网络可访问 DeepSeek API 与今日头条

## 安装

```bash
git clone <仓库地址>
cd aTouyiao

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install openai chromadb playwright requests schedule numpy scikit-learn
# 可选：更好的中文语义嵌入（体积较大）
pip install sentence-transformers
```

> ⚠️ 本仓库提交时已排除 `.venv/`，克隆后需自行创建虚拟环境。

## 配置

通过环境变量注入密钥（**不要硬编码到代码里**）：

```bat
set DEEPSEEK_API_KEY=sk-xxxx        # 必需：内容与标题生成
set PIXABAY_KEY=xxxx                # 可选：配图来源
set UNSPLASH_KEY=xxxx               # 可选：配图来源
```

## 使用

### 1. 获取 Cookie（首次必须）

双击 `cookie_setup.bat`，在弹出的 Edge 中登录 `mp.toutiao.com`，登录完成后回车，Cookie 自动保存到 `cookies/account_01.json`。

### 2. 运行工作流

双击 `run.bat`，或在激活的虚拟环境中执行 `python toutiao.py`，按菜单选择：

```
1. 立即发布（单账号）
2. 启动定时调度（每日 20:00-21:59 窗口）
3. 获取/更新 Cookie
4. 查看标题表现统计
5. 测试标题生成
6. 测试图片插入分析
```

## 项目结构

```
aTouyiao/
├── toutiao.py           # 工作流主程序（全部模块）
├── run.bat              # 一键运行
├── cookie_setup.bat     # Cookie 获取工具
├── cookies/             # 账号 Cookie（不入库）
├── vector_db/           # 向量数据库（运行时生成，不入库）
└── .venv/               # 虚拟环境（不入库）
```

## 隐私与安全

- `cookies/` 存放真实账号登录凭据，已在 `.gitignore` 中排除，**请勿提交**
- 所有 API Key 均通过环境变量读取
- 本项目仅供学习与个人内容管理使用，请遵守今日头条平台规范

## License

仅供个人学习使用，未附开源协议。
