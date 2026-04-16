import os
from dotenv import load_dotenv

# 加载 .env 文件中的变量到环境变量
load_dotenv()

# 读取配置 (如果没读到，第二个参数是默认值)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# PostgreSQL 数据库配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "192.168.104.129"),
    "port": os.getenv("DB_PORT", "5455"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres")
}

# Neo4j 数据库配置
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "password")
}

GRAPH_NAME = os.getenv("GRAPH_NAME", "neo4j")
ORIGIN_NAME = os.getenv("ORIGIN_NAME", "kg2_stg")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-max")

# 简单检查
if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")