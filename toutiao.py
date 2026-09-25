"""
今日头条自动发文工作流 v2.0
核心升级: 向量去重 + 标题模型训练 + Cookie 池 + 智能图片插入
"""

import asyncio
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import random
import json
import os
import hashlib
import pickle
import time
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from pathlib import Path

# 向量数据库
import chromadb
from chromadb.config import Settings

# SentenceTransformer (需要 torch，有些 Windows 环境可能不兼容)
_USE_SENTENCE_TRANSFORMER = False
_SENTENCE_TRANSFORMER = None
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMER = SentenceTransformer
    _USE_SENTENCE_TRANSFORMER = True
except Exception as e:
    print(f"[Warning] SentenceTransformer 加载失败，使用 TF-IDF 替代: {e}")
    from sklearn.feature_extraction.text import TfidfVectorizer

# 浏览器自动化
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# API
import openai
import requests

# 调度
import schedule

# ==================== 配置 ====================

@dataclass
class Config:
    # API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    
    # 向量数据库
    VECTOR_DB_PATH: str = "./vector_db"
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"  # 中文语义理解强
    
    # Cookie 池
    COOKIE_POOL_DIR: str = "./cookies"
    MAX_ACCOUNTS: int = 5
    
    # 标题模型
    TITLE_MODEL_PATH: str = "./models/title_generator.pkl"
    PERFORMANCE_DB: str = "./data/performance.json"
    
    # 图片
    UNSPLASH_ACCESS_KEY: str = os.getenv("UNSPLASH_KEY", "")
    PIXABAY_API_KEY: str = os.getenv("PIXABAY_KEY", "")
    
    # 发布
    PUBLISH_URL: str = "https://mp.toutiao.com/profile_v4/graphic/publish"
    TIME_WINDOW: Tuple[int, int] = (20, 21)  # 20:00-21:59

CONFIG = Config()


# ==================== 1. 向量数据库模块 (标题去重) ====================

class EmbeddingProvider:
    """嵌入提供者：优先用 SentenceTransformer，回退到 TF-IDF"""

    def __init__(self):
        self.use_sentence = _USE_SENTENCE_TRANSFORMER
        if self.use_sentence:
            print("加载嵌入模型...")
            self.model = _SENTENCE_TRANSFORMER(CONFIG.EMBEDDING_MODEL)
            print("嵌入模型加载完成")
        else:
            print("使用 TF-IDF 嵌入（无需 torch）")
            self.vectorizer = TfidfVectorizer(
                max_features=512,
                analyzer='char',
                ngram_range=(2, 4),
                token_pattern=None
            )
            self._fitted = False

    def encode(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        if self.use_sentence:
            result = self.model.encode(texts)
        else:
            # TF-IDF 嵌入
            if not self._fitted:
                embeddings = self.vectorizer.fit_transform(texts).toarray()
                self._fitted = True
            else:
                embeddings = self.vectorizer.transform(texts).toarray()
            # 归一化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            result = embeddings / norms
        return result[0] if single else result


class VectorStore:
    """基于 ChromaDB 的向量存储，用于标题去重和相似度检索"""

    def __init__(self, db_path: str = CONFIG.VECTOR_DB_PATH):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="toutiao_titles",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = EmbeddingProvider()
    
    def add_title(self, title: str, article_type: str = "情感", 
                  publish_time: str = None, read_count: int = 0):
        """添加已发布标题到向量库"""
        embedding = self.embedder.encode(title).tolist()
        doc_id = hashlib.md5(title.encode()).hexdigest()
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[title],
            metadatas=[{
                "type": article_type,
                "publish_time": publish_time or datetime.now().isoformat(),
                "read_count": read_count
            }]
        )
        return doc_id
    
    def is_duplicate(self, title: str, threshold: float = 0.85) -> Tuple[bool, float]:
        """
        检查标题是否重复
        threshold: 相似度阈值，超过则认为重复
        """
        results = self.collection.query(
            query_embeddings=[self.embedder.encode(title).tolist()],
            n_results=1
        )
        
        if not results['ids'][0]:
            return False, 0.0
        
        similarity = 1 - results['distances'][0][0]  # cosine distance -> similarity
        return similarity > threshold, similarity
    
    def get_similar_titles(self, title: str, n: int = 5) -> List[Dict]:
        """获取相似标题（用于标题生成参考）"""
        results = self.collection.query(
            query_embeddings=[self.embedder.encode(title).tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        
        similar = []
        for i in range(len(results['ids'][0])):
            similar.append({
                "title": results['documents'][0][i],
                "similarity": 1 - results['distances'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        return similar
    
    def get_title_performance_stats(self) -> Dict[str, Dict]:
        """获取各类标题的表现统计"""
        all_data = self.collection.get(include=["metadatas"])
        
        stats = defaultdict(lambda: {"count": 0, "total_reads": 0, "titles": []})
        for meta in all_data['metadatas']:
            t = meta.get("type", "unknown")
            stats[t]["count"] += 1
            stats[t]["total_reads"] += meta.get("read_count", 0)
            stats[t]["titles"].append({
                "title": meta.get("title", ""),
                "reads": meta.get("read_count", 0)
            })
        
        # 计算平均阅读量
        for t in stats:
            stats[t]["avg_reads"] = stats[t]["total_reads"] / max(stats[t]["count"], 1)
        
        return dict(stats)


# ==================== 2. 标题生成模型 (带 A/B 测试学习) ====================

class TitlePerformanceTracker:
    """追踪标题表现，训练生成策略"""
    
    def __init__(self, db_path: str = CONFIG.PERFORMANCE_DB):
        self.db_path = db_path
        self.data = self._load()
        
        # 标题模板库（按类型分类）
        self.templates = {
            "疑问型": [
                "为什么{topic}的人，最后都{result}？",
                "{topic}，到底是一种什么样的体验？",
                "你有没有发现，{topic}的人都有一个共同点？"
            ],
            "共鸣型": [
                "真正的{topic}，是{action}",
                "后来我才明白，{topic}不需要{thing}",
                "{topic}的人，都懂这句话",
                "成年人的{topic}，从{signal}开始"
            ],
            "数字型": [
                "{topic}的{num}个信号，中{num2}条就该{action}了",
                "关于{topic}，我想告诉你{num}件事",
                "{num}年{topic}，我终于学会了{lesson}"
            ],
            "对比型": [
                "以前觉得{thing_a}，后来才懂{thing_b}",
                "不是{thing_a}，而是{thing_b}",
                "{topic}和{topic2}，你选哪个？"
            ],
            "情绪型": [
                "{emotion}，是{topic}最好的方式",
                "那些{topic}的夜晚，你是怎么过来的？",
                "写给{target}：关于{topic}，{message}"
            ]
        }
        
        # 热门关键词库
        self.hot_keywords = {
            "情感": ["释怀", "遗憾", "错过", "孤独", "成长", "自愈", "边界感", 
                    "情绪价值", "内耗", "讨好型人格", "安全感", "仪式感"],
            "成长": ["清醒", "通透", "格局", "认知", "觉醒", "蜕变", "沉淀"],
            "生活": ["烟火气", "仪式感", "松弛感", "钝感力", "屏蔽力"]
        }
        
        # 学习到的权重（哪些模板表现更好）
        self.template_weights = self.data.get("template_weights", {
            t: 1.0 for category in self.templates for t in self.templates[category]
        })
    
    def _load(self) -> Dict:
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"history": [], "template_weights": {}}
    
    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def record_performance(self, title: str, template_used: str, 
                         read_count: int, like_count: int = 0):
        """记录标题表现"""
        self.data["history"].append({
            "title": title,
            "template": template_used,
            "read_count": read_count,
            "like_count": like_count,
            "time": datetime.now().isoformat()
        })
        
        # 更新模板权重（简单强化学习）
        if template_used in self.template_weights:
            # 阅读量越高，权重增加越多
            reward = min(read_count / 10000, 2.0)  # 封顶 2x
            self.template_weights[template_used] = (
                self.template_weights[template_used] * 0.9 + reward * 0.1
            )
        
        self.save()
    
    def select_template(self, article_type: str = "情感") -> Tuple[str, str]:
        """
        根据学习到的权重选择模板
        返回: (模板类别, 具体模板)
        """
        # 过滤相关类别的模板
        available = {}
        for category, templates in self.templates.items():
            for t in templates:
                available[f"{category}|{t}"] = self.template_weights.get(t, 1.0)
        
        # 按权重采样
        total = sum(available.values())
        r = random.uniform(0, total)
        cumulative = 0
        selected = None
        
        for key, weight in available.items():
            cumulative += weight
            if r <= cumulative:
                selected = key
                break
        
        category, template = selected.split("|", 1)
        return category, template
    
    def fill_template(self, template: str, article_type: str = "情感") -> str:
        """填充模板生成标题"""
        keywords = self.hot_keywords.get(article_type, self.hot_keywords["情感"])
        topic = random.choice(keywords)

        # 简单的关键词替换
        fill_map = {
            "{topic}": topic,
            "{topic2}": random.choice([k for k in keywords if k != topic]) if len(keywords) > 1 else topic,
            "{result}": random.choice(["赢了", "输了", "释然了", "放下了", "成长了"]),
            "{action}": random.choice(["放下", "释怀", "沉默", "转身", "接受"]),
            "{thing}": random.choice(["解释", "道歉", "告别", "挽留", "证明"]),
            "{signal}": random.choice(["沉默", "微笑", "独处", "早睡", "断联"]),
            "{num}": str(random.choice([3, 5, 7, 10])),
            "{num2}": str(random.choice([1, 2, 3])),
            "{lesson}": random.choice(["放手", "爱自己", "不讨好", "有边界"]),
            "{emotion}": random.choice(["沉默", "独处", "清醒", "坦然", "温柔"]),
            "{target}": random.choice(["敏感的人", "失眠的人", "懂事的人", "心软的人"]),
            "{message}": random.choice(["我懂你的不容易", "你值得被爱", "别为难自己"]),
            "{thing_a}": random.choice(["要轰轰烈烈", "要仪式感", "要追问到底", "要讨个说法"]),
            "{thing_b}": random.choice([" quietly 就够了", "放下才是答案", "沉默是体面", "时间会给答案"]),
        }
        
        title = template
        for key, value in fill_map.items():
            title = title.replace(key, value, 1)
        
        return title


class TitleGenerator:
    """智能标题生成器（结合模板 + AI + 历史表现）"""
    
    def __init__(self, api_key: str, vector_store: VectorStore):
        self.client = openai.OpenAI(api_key=api_key, base_url=CONFIG.DEEPSEEK_BASE_URL)
        self.vector_store = vector_store
        self.tracker = TitlePerformanceTracker()
    
    def generate(self, article_type: str = "情感", use_ai: bool = True) -> Tuple[str, str]:
        """
        生成标题
        返回: (标题, 使用的模板/策略)
        """
        # 策略1: 使用学习到的模板（70%概率）
        if random.random() < 0.7:
            category, template = self.tracker.select_template(article_type)
            title = self.tracker.fill_template(template, article_type)
            
            # 检查重复
            is_dup, sim = self.vector_store.is_duplicate(title)
            if not is_dup:
                return title, f"template:{category}"
        
        # 策略2: AI 生成（30%概率 或 模板重复时）
        if use_ai:
            return self._generate_by_ai(article_type), "ai:deepseek"
        
        # 策略3: 兜底 - 随机组合
        return self._generate_fallback(article_type), "fallback"
    
    def _generate_by_ai(self, article_type: str) -> str:
        """使用 DeepSeek 生成标题"""
        # 获取历史高表现标题作为参考
        similar = self.vector_store.get_similar_titles("情感 治愈 成长", n=3)
        examples = "\n".join([
            f"- {s['title']} (阅读量: {s['metadata'].get('read_count', 0)})"
            for s in similar
        ])
        
        prompt = f"""你是一个爆款标题专家。请生成一个{article_type}类的今日头条文章标题。

要求：
1. 长度在 15-30 字之间
2. 要有情绪共鸣，引发好奇或共鸣
3. 避免夸张和标题党，但要足够吸引人
4. 参考以下历史高表现标题风格：
{examples}

请直接输出标题，不要加引号或解释。"""
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=50
            )
            title = response.choices[0].message.content.strip()
            # 清理
            title = re.sub(r'^["\']|["\']$', '', title)
            return title
        except Exception as e:
            print(f"AI 标题生成失败: {e}")
            return self._generate_fallback(article_type)
    
    def _generate_fallback(self, article_type: str) -> str:
        """兜底标题生成"""
        prefixes = ["后来我才明白", "真正的", "成年人的", "那些", "写给"]
        middles = ["释怀", "孤独", "遗憾", "成长", "自愈", "边界感"]
        suffixes = ["，不需要解释", "，是沉默", "，从接纳开始", "，你懂吗", "，最好的方式"]
        
        return f"{random.choice(prefixes)}{random.choice(middles)}{random.choice(suffixes)}"


# ==================== 3. Cookie 池管理模块 ====================

class CookiePool:
    """多账号 Cookie 池管理"""
    
    def __init__(self, pool_dir: str = CONFIG.COOKIE_POOL_DIR):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(exist_ok=True)
        self.accounts = self._load_all()
    
    def _load_all(self) -> Dict[str, Dict]:
        """加载所有账号"""
        accounts = {}
        for cookie_file in self.pool_dir.glob("*.json"):
            account_id = cookie_file.stem
            with open(cookie_file, 'r', encoding='utf-8') as f:
                accounts[account_id] = json.load(f)
        return accounts
    
    def save_account(self, account_id: str, cookies: List[Dict], 
                     user_agent: str, fingerprint: Dict = None):
        """保存账号 Cookie"""
        data = {
            "account_id": account_id,
            "cookies": cookies,
            "user_agent": user_agent,
            "fingerprint": fingerprint or {},
            "last_used": datetime.now().isoformat(),
            "use_count": 0,
            "status": "active"
        }
        
        file_path = self.pool_dir / f"{account_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.accounts[account_id] = data
        print(f"账号 {account_id} 已保存")
    
    def get_account(self, strategy: str = "random") -> Optional[Dict]:
        """
        获取账号
        strategy: random / round_robin / least_used / best_performance
        """
        active = {k: v for k, v in self.accounts.items() 
                 if v.get("status") == "active"}
        
        if not active:
            return None
        
        if strategy == "random":
            return random.choice(list(active.values()))
        
        elif strategy == "round_robin":
            # 简单轮询：找最久未使用的
            return min(active.values(), 
                      key=lambda x: x.get("last_used", ""))
        
        elif strategy == "least_used":
            return min(active.values(),
                      key=lambda x: x.get("use_count", 0))
        
        return random.choice(list(active.values()))
    
    def mark_used(self, account_id: str):
        """标记账号已使用"""
        if account_id in self.accounts:
            self.accounts[account_id]["use_count"] += 1
            self.accounts[account_id]["last_used"] = datetime.now().isoformat()
            self._save_account_file(account_id)
    
    def mark_failed(self, account_id: str, reason: str = ""):
        """标记账号异常"""
        if account_id in self.accounts:
            self.accounts[account_id]["status"] = "failed"
            self.accounts[account_id]["fail_reason"] = reason
            self._save_account_file(account_id)
    
    def _save_account_file(self, account_id: str):
        """保存单个账号文件"""
        file_path = self.pool_dir / f"{account_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.accounts[account_id], f, ensure_ascii=False, indent=2)


class MultiAccountPublisher:
    """支持多账号轮换的发布器"""
    
    def __init__(self, cookie_pool: CookiePool):
        self.cookie_pool = cookie_pool
        self.current_account = None
        self.browser = None
        self.context = None
        self.page = None
    
    async def launch_with_account(self, account_id: str = None, 
                                   headless: bool = False):
        """使用指定账号启动浏览器"""
        playwright = await async_playwright().start()
        
        # 获取账号
        if account_id:
            account = self.cookie_pool.accounts.get(account_id)
        else:
            account = self.cookie_pool.get_account("least_used")
        
        if not account:
            raise Exception("没有可用的账号，请先登录获取 Cookie")
        
        self.current_account = account
        
        # 浏览器指纹伪装
        fingerprint = account.get("fingerprint", {})
        
        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                f'--window-size={fingerprint.get("viewport", "1920,1080")}',
            ]
        )
        
        # 创建带 Cookie 的上下文
        self.context = await self.browser.new_context(
            viewport={
                "width": int(fingerprint.get("viewport", "1920,1080").split(',')[0]),
                "height": int(fingerprint.get("viewport", "1920,1080").split(',')[1])
            },
            user_agent=account.get("user_agent", 
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
            locale=fingerprint.get("locale", "zh-CN"),
            timezone_id=fingerprint.get("timezone", "Asia/Shanghai"),
        )
        
        # 注入 Cookie
        cookies = account.get("cookies", [])
        if cookies:
            await self.context.add_cookies(cookies)
        
        # 反检测脚本
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
            window.chrome = { runtime: {} };
        """)
        
        self.page = await self.context.new_page()
        
        # 标记使用
        self.cookie_pool.mark_used(account["account_id"])
        print(f"已使用账号: {account['account_id']}")
        
        return self
    
    async def refresh_cookies(self):
        """刷新当前账号的 Cookie"""
        if self.current_account and self.context:
            cookies = await self.context.cookies()
            self.cookie_pool.save_account(
                self.current_account["account_id"],
                cookies,
                self.current_account.get("user_agent", ""),
                self.current_account.get("fingerprint", {})
            )
            print("Cookie 已刷新")
    
    async def close(self):
        """关闭并保存状态"""
        if self.current_account:
            await self.refresh_cookies()
        if self.browser:
            await self.browser.close()


# ==================== 4. 智能图片插入模块 ====================

class SmartImageInserter:
    """
    智能图片插入器
    1. 分析文章语义，提取关键场景
    2. 搜索匹配图片
    3. 计算最佳插入位置（避免打断阅读流）
    """
    
    # 场景-图片映射
    SCENE_KEYWORDS = {
        "早晨/日出": ["晨光", "日出", "清晨", "早安", "新的一天", "黎明"],
        "夜晚/城市": ["夜晚", "深夜", "凌晨", "失眠", "城市夜景", "灯火"],
        "自然/户外": ["阳光", "微风", "走路", "散步", "风", "天气好"],
        "室内/独处": ["房间", "家里", "沙发", "窗前", "镜子", "一个人"],
        "情感/回忆": ["想起", "回忆", "曾经", "过去", "日记", "照片"],
        "治愈/温暖": ["舒服", "温暖", "治愈", "安心", "平静", "释然"],
        "雨天/忧郁": ["下雨", "雨天", "淋雨", "潮湿", "阴天"],
        "咖啡/生活": ["咖啡", "咖啡馆", "杯子", "热气", "生活"],
        "道路/前行": ["路", "走路", "前行", "方向", "远方", " journey"],
        "音乐/耳机": ["歌", "音乐", "耳机", "听歌", "旋律"],
        "书信/文字": ["信", "写", "文字", "日记", "留言", "纸条"],
        "海边/开阔": ["海", "海边", "波浪", "沙滩", "辽阔", " horizon"],
    }
    
    def __init__(self):
        self.embedder = EmbeddingProvider()
    
    def analyze_article(self, article: str) -> List[Dict]:
        """
        分析文章，提取适合配图的场景段落
        返回: [{position, paragraph, scene_type, confidence}]
        """
        paragraphs = [p.strip() for p in article.split('\n\n') if p.strip()]
        paragraphs = [p for p in paragraphs if len(p) > 20]  # 过滤太短段落
        
        results = []
        
        for i, para in enumerate(paragraphs):
            # 计算段落与每个场景的语义匹配度
            para_embedding = self.embedder.encode(para)
            
            best_scene = None
            best_score = -1
            
            for scene, keywords in self.SCENE_KEYWORDS.items():
                # 关键词匹配 + 语义匹配
                keyword_score = sum(1 for kw in keywords if kw in para) / len(keywords)
                
                # 语义相似度
                scene_text = "，".join(keywords[:3])
                scene_embedding = self.embedder.encode(scene_text)
                semantic_score = np.dot(para_embedding, scene_embedding) / (
                    np.linalg.norm(para_embedding) * np.linalg.norm(scene_embedding)
                )
                
                # 综合得分
                score = keyword_score * 0.6 + semantic_score * 0.4
                
                if score > best_score:
                    best_score = score
                    best_scene = scene
            
            # 只保留高置信度的场景
            if best_score > 0.3:
                results.append({
                    "position": i,
                    "paragraph": para[:50] + "...",
                    "scene_type": best_scene,
                    "confidence": float(best_score),
                    "full_text": para
                })
        
        # 按置信度排序，选择前 N 个，并确保间距合理
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        selected = []
        used_positions = set()
        
        for r in results:
            # 确保图片之间至少间隔 2 个段落
            if all(abs(r["position"] - p) > 2 for p in used_positions):
                selected.append(r)
                used_positions.add(r["position"])
            
            if len(selected) >= 3:  # 最多 3 张图
                break
        
        # 按原文顺序排序
        selected.sort(key=lambda x: x["position"])
        return selected
    
    async def search_images(self, scene_type: str, count: int = 1) -> List[str]:
        """
        根据场景类型搜索图片
        支持 Unsplash 和 Pixabay
        """
        # 将场景类型转换为英文搜索词
        search_map = {
            "早晨/日出": "morning sunrise warm light",
            "夜晚/城市": "night city lights mood",
            "自然/户外": "nature sunlight peaceful",
            "室内/独处": "cozy room alone window",
            "情感/回忆": "memories vintage photo",
            "治愈/温暖": "warm healing comfort",
            "雨天/忧郁": "rainy day mood window",
            "咖啡/生活": "coffee cup cozy life",
            "道路/前行": "road journey ahead",
            "音乐/耳机": "headphones music relax",
            "书信/文字": "letter handwriting vintage",
            "海边/开阔": "sea sunset horizon calm",
        }
        
        query = search_map.get(scene_type, "warm mood aesthetic")
        image_urls = []
        
        # Unsplash API
        if CONFIG.UNSPLASH_ACCESS_KEY:
            try:
                url = f"https://api.unsplash.com/search/photos"
                params = {
                    "query": query,
                    "per_page": count,
                    "orientation": "landscape",
                    "client_id": CONFIG.UNSPLASH_ACCESS_KEY
                }
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for img in data.get("results", []):
                        image_urls.append(img["urls"]["regular"])
            except Exception as e:
                print(f"Unsplash 搜索失败: {e}")
        
        # Pixabay 备用
        if not image_urls and CONFIG.PIXABAY_API_KEY:
            try:
                url = "https://pixabay.com/api/"
                params = {
                    "key": CONFIG.PIXABAY_API_KEY,
                    "q": query,
                    "per_page": count,
                    "orientation": "horizontal"
                }
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for img in data.get("hits", []):
                        image_urls.append(img["webformatURL"])
            except Exception as e:
                print(f"Pixabay 搜索失败: {e}")
        
        return image_urls
    
    async def insert_images_to_page(self, page: Page, article: str, 
                                    image_positions: List[Dict]):
        """
        在浏览器页面中智能插入图片
        """
        # 分析最佳插入位置
        scenes = self.analyze_article(article)
        
        for i, scene in enumerate(scenes):
            print(f"准备插入图片 {i+1}: 场景={scene['scene_type']}, "
                  f"位置=段落{scene['position']}, 置信度={scene['confidence']:.2f}")
            
            # 搜索图片
            image_urls = await self.search_images(scene["scene_type"], count=1)
            if not image_urls:
                continue
            
            # 定位到对应段落
            # 策略：在段落末尾插入图片
            paragraphs = await page.query_selector_all(
                '.ProseMirror p, .editor-content p, [contenteditable="true"] p'
            )
            
            target_idx = min(scene["position"], len(paragraphs) - 1)
            if target_idx < 0:
                continue
            
            target_para = paragraphs[target_idx]
            
            # 滚动到可视区域
            await target_para.scroll_into_view_if_needed()
            await asyncio.sleep(0.5)
            
            # 在段落后点击，准备插入图片
            await target_para.click()
            await page.keyboard.press('End')  # 移到段落末尾
            await page.keyboard.press('Enter')
            
            # 打开图片选择器
            add_img_btn = await page.query_selector(
                'button:has-text("图片"), .image-btn, [title="插入图片"]'
            )
            if add_img_btn:
                await add_img_btn.click()
                await asyncio.sleep(1)
                
                # 选择免费正版图片
                free_tab = await page.query_selector(
                    'text=免费正版, [data-tab="free"], .free-image-tab'
                )
                if free_tab:
                    await free_tab.click()
                    await asyncio.sleep(1)
                
                # 搜索对应场景
                search_box = await page.query_selector(
                    'input[placeholder*="搜索"], .image-search input'
                )
                if search_box:
                    # 使用中文关键词搜索
                    cn_keywords = {
                        "早晨/日出": "阳光 温暖",
                        "夜晚/城市": "夜晚 城市",
                        "自然/户外": "自然 风景",
                        "室内/独处": "室内 安静",
                        "情感/回忆": "回忆 怀旧",
                        "治愈/温暖": "治愈 温暖",
                        "雨天/忧郁": "雨天 忧郁",
                        "咖啡/生活": "咖啡 生活",
                        "道路/前行": "道路 远方",
                        "音乐/耳机": "音乐 耳机",
                        "书信/文字": "书信 文字",
                        "海边/开阔": "海边 日落",
                    }
                    search_term = cn_keywords.get(scene["scene_type"], "温暖")
                    await search_box.fill(search_term)
                    await page.keyboard.press('Enter')
                    await asyncio.sleep(2)
                
                # 选择第一张
                first_img = await page.query_selector(
                    '.image-item:first-child, .image-grid img:first-child, .free-image-item'
                )
                if first_img:
                    await first_img.click()
                    await asyncio.sleep(0.5)
                    
                    insert_btn = await page.query_selector(
                        'button:has-text("插入"), .insert-btn, button:has-text("确认")'
                    )
                    if insert_btn:
                        await insert_btn.click()
                        print(f"  ✓ 图片 {i+1} 插入成功")
                        await asyncio.sleep(1)
            
            # 在图片后添加空行
            await page.keyboard.press('Enter')
        
        return len(scenes)


# ==================== 5. 主工作流整合 ====================

class ToutiaoWorkflow:
    """完整工作流"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.title_generator = TitleGenerator(CONFIG.DEEPSEEK_API_KEY, self.vector_store)
        self.cookie_pool = CookiePool()
        self.image_inserter = SmartImageInserter()
        self.performance_tracker = TitlePerformanceTracker()
        
        # 内容生成器
        self.content_client = openai.OpenAI(
            api_key=CONFIG.DEEPSEEK_API_KEY,
            base_url=CONFIG.DEEPSEEK_BASE_URL
        )
    
    async def generate_content(self, title: str) -> str:
        """生成正文"""
        prompt = f"""围绕标题《{title}》写一篇情感类文章。

要求：
1. 800-1200字，适合手机阅读
2. 多用短句和分段，有画面感
3. 开头引入场景，中间有具体细节，结尾留白
4. 避免说教，像朋友聊天
5. 包含2-3个适合配图的场景描述（如阳光、咖啡、夜晚等）

直接输出正文，不要标题。"""
        
        try:
            resp = self.content_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2000
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"内容生成失败: {e}")
            return self._fallback_content(title)
    
    def _fallback_content(self, title: str) -> str:
        """备用内容"""
        return f"""曾经以为，{title}是一件很隆重的事。

应该有仪式感。比如删掉所有聊天记录，比如烧掉那本日记，比如对着镜子大哭一场然后说"我好了"。

后来才发现，真正的{title}，根本不是这样。

它发生得很安静。

就是你走在路上，阳光很好，风吹过来有点舒服。你突然想起某件事、某个人，然后发现心里已经不那么难受了。

它变成了一件很小的事，小到你想跟别人说起的时候，嘴巴张了张，觉得没什么好说的。

于是闭口不提。

---

其实人都是这样长大的。

那些以为永远过不去的坎，后来都成了茶余饭后的谈资。那些哭得撕心裂肺的夜晚，天亮之后也就那样。

生活还在继续。太阳照常升起，地铁依然拥挤，咖啡还是要点。

而你，也在不知不觉中，变成了更好的自己。

这大概就是{title}最好的样子。"""
    
    async def publish(self, account_id: str = None, headless: bool = False):
        """执行完整发布流程"""
        print(f"\n{'='*60}")
        print(f"🚀 开始发布流程 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # 1. 生成标题
        print("📌 Step 1: 生成标题...")
        title, strategy = self.title_generator.generate()
        
        # 去重检查
        is_dup, sim = self.vector_store.is_duplicate(title)
        if is_dup:
            print(f"  ⚠️ 标题相似度 {sim:.2f}，重新生成...")
            title, strategy = self.title_generator.generate(use_ai=True)
        
        print(f"  ✓ 标题: {title}")
        print(f"  ✓ 策略: {strategy}")
        
        # 2. 生成正文
        print("\n📌 Step 2: 生成正文...")
        article = await self.generate_content(title)
        print(f"  ✓ 字数: {len(article)}")
        
        # 3. 分析图片插入位置
        print("\n📌 Step 3: 分析图片插入位置...")
        image_scenes = self.image_inserter.analyze_article(article)
        print(f"  ✓ 识别到 {len(image_scenes)} 个配图场景:")
        for s in image_scenes:
            print(f"    - [{s['scene_type']}] {s['paragraph'][:30]}...")
        
        # 4. 启动浏览器
        print("\n📌 Step 4: 启动浏览器...")
        publisher = MultiAccountPublisher(self.cookie_pool)
        await publisher.launch_with_account(account_id, headless=headless)
        
        try:
            # 5. 导航到发布页
            print("\n📌 Step 5: 导航到发布页面...")
            await publisher.page.goto(CONFIG.PUBLISH_URL, wait_until="networkidle")
            await asyncio.sleep(3)
            
            # 检查登录状态
            if "login" in publisher.page.url:
                print("  ⚠️ 需要登录，等待手动登录...")
                await asyncio.sleep(60)
            
            # 6. 填写标题
            print("\n📌 Step 6: 填写标题...")
            await self._fill_title(publisher.page, title)
            
            # 7. 填写正文
            print("\n📌 Step 7: 填写正文...")
            await self._fill_content(publisher.page, article)
            
            # 8. 智能插入图片
            print("\n📌 Step 8: 智能插入图片...")
            inserted = await self.image_inserter.insert_images_to_page(
                publisher.page, article, image_scenes
            )
            print(f"  ✓ 成功插入 {inserted} 张图片")
            
            # 9. 配置发布选项
            print("\n📌 Step 9: 配置发布选项...")
            await self._configure_settings(publisher.page)
            
            # 10. 预览并发布
            print("\n📌 Step 10: 预览并发布...")
            success = await self._preview_and_publish(publisher.page)
            
            if success:
                # 记录到向量库
                self.vector_store.add_title(
                    title=title,
                    article_type="情感",
                    publish_time=datetime.now().isoformat()
                )
                print(f"\n✅ 发布成功！账号: {publisher.current_account['account_id']}")
            else:
                print("\n❌ 发布失败")
            
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            await publisher.close()
    
    async def _fill_title(self, page: Page, title: str):
        """填写标题"""
        selectors = [
            'input[placeholder*="标题"]',
            '.title-input input',
            '[data-testid="title-input"]',
            'input[maxlength="30"]'
        ]
        
        for sel in selectors:
            try:
                await page.wait_for_selector(sel, timeout=3000)
                await page.fill(sel, title)
                return
            except:
                continue
        
        # 兜底：尝试点击第一个输入框
        await page.click('div[contenteditable="true"]:first-child')
        await page.keyboard.type(title)
    
    async def _fill_content(self, page: Page, content: str):
        """填写正文"""
        editor_selectors = [
            '.ProseMirror',
            '.editor-content',
            '[contenteditable="true"].notranslate',
            'div[contenteditable="true"]'
        ]
        
        editor = None
        for sel in editor_selectors:
            editor = await page.query_selector(sel)
            if editor:
                break
        
        if not editor:
            raise Exception("未找到编辑器")
        
        await editor.click()
        await page.keyboard.press('Control+a')
        await page.keyboard.press('Delete')
        
        # 分段输入
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for para in paragraphs:
            clean = para.replace('---', '').strip()
            if clean:
                await page.keyboard.type(clean)
                await page.keyboard.press('Enter')
                await page.keyboard.press('Enter')
                await asyncio.sleep(0.3)
    
    async def _configure_settings(self, page: Page):
        """配置发布选项"""
        # 封面：随机单图/三图
        cover_options = await page.query_selector_all('.cover-option, [name="cover"]')
        if len(cover_options) >= 2:
            await random.choice(cover_options[:2]).click()
        
        # 投放广告
        for text in ["投放广告赚收益", "广告收益", "开通广告"]:
            checkbox = await page.query_selector(f'text={text}, [name*="ad"], .ad-toggle')
            if checkbox:
                try:
                    await checkbox.click()
                except:
                    pass
                break
        
        # 同时发布微头条
        for text in ["同时发布微头条", "微头条", "同步微头条"]:
            checkbox = await page.query_selector(f'text={text}, [name*="micro"]')
            if checkbox:
                try:
                    await checkbox.click()
                except:
                    pass
                break
        
        # 作品声明
        declare_btn = await page.query_selector('text=作品声明, .declare-btn')
        if declare_btn:
            await declare_btn.click()
            await asyncio.sleep(1)
            
            personal_view = await page.query_selector('text=个人观点，仅供参考')
            if personal_view:
                await personal_view.click()
            
            confirm = await page.query_selector('button:has-text("确认")')
            if confirm:
                await confirm.click()
    
    async def _preview_and_publish(self, page: Page) -> bool:
        """预览并发布"""
        # 预览
        preview_btn = await page.query_selector('button:has-text("预览")')
        if preview_btn:
            await preview_btn.click()
            await asyncio.sleep(2)
            close = await page.query_selector('.preview-close, .close-btn')
            if close:
                await close.click()
        
        # 发布
        publish_btn = await page.query_selector(
            'button:has-text("发布"), button:has-text("立即发布"), .publish-btn'
        )
        if publish_btn:
            await publish_btn.click()
            await asyncio.sleep(2)
            
            # 确认弹窗
            confirm = await page.query_selector(
                'button:has-text("确认"), .confirm-btn, button:has-text("确定")'
            )
            if confirm:
                await confirm.click()
            
            await asyncio.sleep(3)
            return True
        
        return False
    
    def schedule_daily(self):
        """设置每日定时"""
        def job():
            hour = random.randint(CONFIG.TIME_WINDOW[0], CONFIG.TIME_WINDOW[1])
            minute = random.randint(0, 59)
            print(f"⏰ 今日发布时间: {hour:02d}:{minute:02d}")
            time.sleep((hour - datetime.now().hour) * 3600 + (minute - datetime.now().minute) * 60)
            asyncio.run(self.publish())
        
        schedule.every().day.at("19:55").do(job)
        print(f"📅 已设置每日定时任务 (窗口: {CONFIG.TIME_WINDOW[0]}:00-{CONFIG.TIME_WINDOW[1]}:59)")


# ==================== 6. Cookie 获取工具 ====================

async def capture_cookies():
    """交互式 Cookie 获取工具"""
    print("""
    ╔══════════════════════════════════════════╗
    ║     Cookie 获取工具                        ║
    ║     请按提示操作                           ║
    ╚══════════════════════════════════════════╝
    """)
    
    account_id = input("请输入账号标识 (如: account_01): ").strip()
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        channel="msedge",
        headless=False
    )
    
    context = await browser.new_context(viewport={"width": 1920, "height": 1080})
    page = await context.new_page()
    
    print("请在浏览器中登录今日头条创作者平台...")
    await page.goto("https://mp.toutiao.com")
    
    input("登录完成后按回车继续...")
    
    # 获取 Cookie
    cookies = await context.cookies()
    user_agent = await page.evaluate("() => navigator.userAgent")
    
    # 获取指纹信息
    fingerprint = {
        "viewport": "1920,1080",
        "locale": await page.evaluate("() => navigator.language"),
        "timezone": await page.evaluate("() => Intl.DateTimeFormat().resolvedOptions().timeZone"),
        "platform": await page.evaluate("() => navigator.platform")
    }
    
    # 保存
    cookie_pool = CookiePool()
    cookie_pool.save_account(account_id, cookies, user_agent, fingerprint)
    
    print(f"\n✅ 账号 {account_id} 的 Cookie 已保存！")
    await browser.close()


# ==================== 主入口 ====================

async def main():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║     今日头条自动发文工作流 v2.0                   ║
    ║     向量去重 | 标题学习 | Cookie池 | 智能配图      ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    print("请选择操作:")
    print("1. 🚀 立即发布 (单账号)")
    print("2. ⏰ 启动定时调度")
    print("3. 🔑 获取/更新 Cookie")
    print("4. 📊 查看标题表现统计")
    print("5. 🧪 测试标题生成")
    print("6. 🖼️  测试图片插入分析")
    
    choice = input("\n输入选项: ").strip()
    workflow = ToutiaoWorkflow()
    
    if choice == "1":
        account = input("账号ID (留空自动选择): ").strip() or None
        await workflow.publish(account_id=account, headless=False)
    
    elif choice == "2":
        workflow.schedule_daily()
        while True:
            schedule.run_pending()
            time.sleep(30)
    
    elif choice == "3":
        await capture_cookies()
    
    elif choice == "4":
        stats = workflow.vector_store.get_title_performance_stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    elif choice == "5":
        for i in range(5):
            title, strategy = workflow.title_generator.generate()
            is_dup, sim = workflow.vector_store.is_duplicate(title)
            print(f"{i+1}. [{strategy}] {title} (重复: {is_dup}, 相似度: {sim:.2f})")
    
    elif choice == "6":
        test_article = """曾经以为，释怀是一件很隆重的事。

应该有仪式感。比如删掉所有聊天记录，比如烧掉那本日记。

后来才发现，真正的释怀，根本不是这样。

就是你走在路上，阳光很好，风吹过来有点舒服。

那些哭得撕心裂肺的夜晚，天亮之后也就那样。生活还在继续，咖啡还是要点。

这大概就是释怀最好的样子。"""
        
        scenes = workflow.image_inserter.analyze_article(test_article)
        print(f"\n识别到 {len(scenes)} 个场景:")
        for s in scenes:
            print(f"  [{s['scene_type']}] 段落{s['position']}: {s['paragraph'][:40]}...")
    
    else:
        print("无效选项")

if __name__ == "__main__":
    asyncio.run(main())