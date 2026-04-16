import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from sentence_transformers import SentenceTransformer
from config import DB_CONFIG, GRAPH_NAME, ORIGIN_NAME

# 复用你脚本里的配置
MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
SEARCH_COLUMN = "核查描述"

# === 1. 定义多表配置 (核心修改) ===
# 这里定义你想让 Agent 检索的所有业务表
# 注意：列名必须与数据库中的实际列名一致
VECTOR_TABLES_CONFIG = [
    {
        "name": "防御区",               # 业务名称
        "source_table": "防御区",       # 原始表名
        "target_table": "防御区_embeddings", # 向量表名
        "search_column": "补充描述",    # 用于向量化的文本列（实际列名）
        "id_column": "防御区编码",       # 业务主键（实际列名）
    },
    {
        "name": "承灾体",               # 业务名称
        "source_table": "承灾体",       # 原始表名
        "target_table": "承灾体_embeddings", # 向量表名
        "search_column": "补充描述",    # 用于向量化的文本列
        "id_column": "承灾体编码",       # 业务主键
    }
]

def sync_data_to_pgvector():
    print("1. 加载本地 Embedding 模型...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("2. 连接数据库...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 确保向量插件开启
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 创建存储向量的表 (如果不存在)
    cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{ORIGIN_NAME}";') # 确保 Schema 存在



    # === 2. 循环处理每个配置 ===
    for config in VECTOR_TABLES_CONFIG:
        src_table = config["source_table"]
        tgt_table = config["target_table"]
        col_name = config["search_column"]
        id_col = config["id_column"]
        
        print(f"\n🚀 正在处理业务: {config['name']} ...")

        # A. 动态建表
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS "{ORIGIN_NAME}"."{tgt_table}" (
                id SERIAL PRIMARY KEY,
                node_id VARCHAR(50),      
                content TEXT,             
                full_metadata JSONB,      
                embedding vector(512)     
            );
            CREATE INDEX IF NOT EXISTS "idx_{tgt_table}" 
            ON "{ORIGIN_NAME}"."{tgt_table}" USING hnsw (embedding vector_cosine_ops);
        """)
        
        # B. 动态读取
        print(f"   读取源表: {src_table}...")
        try:
            cursor.execute(f'SELECT * FROM "{ORIGIN_NAME}"."{src_table}" WHERE "{col_name}" IS NOT NULL')
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        except Exception as e:
            print(f"   ⚠️ 跳过: 表 {src_table} 读取失败或不存在 ({e})")
            continue

        if not rows:
            print("   ⚠️ 跳过: 无数据")
            continue

        # C. 批量向量化
        data_to_insert = []
        print(f"   正在向量化 {len(rows)} 条数据...")
        
        for row in rows:
            row_dict = dict(zip(columns, row))
            text_content = row_dict.get(col_name, "")
            
            # 动态获取 ID
            node_id = str(row_dict.get(id_col, 'unknown'))
            
            vector = model.encode(text_content).tolist()
            
            import json
            data_to_insert.append((
                node_id, 
                text_content, 
                json.dumps(row_dict, default=str), 
                vector
            ))

        # D. 写入 (先清空旧数据，防止重复叠加，根据需求可选)
        cursor.execute(f'TRUNCATE TABLE "{ORIGIN_NAME}"."{tgt_table}"')
        
        insert_query = f"""
            INSERT INTO "{ORIGIN_NAME}"."{tgt_table}" 
            (node_id, content, full_metadata, embedding) 
            VALUES (%s, %s, %s, %s)
        """
        cursor.executemany(insert_query, data_to_insert)
        conn.commit()
        print(f"   ✅ {config['name']} 处理完成！")

    conn.close()

if __name__ == "__main__":
    sync_data_to_pgvector()