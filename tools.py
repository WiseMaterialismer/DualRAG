import json, re, os
import pg8000
from neo4j import GraphDatabase
from langchain_core.tools import tool
from streamlit_agraph import agraph, Node, Edge, Config
from sentence_transformers import SentenceTransformer, CrossEncoder

# 设置环境变量来解决编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'

from config import DB_CONFIG, NEO4J_CONFIG, GRAPH_NAME, ORIGIN_NAME
from prompts import get_zero_results_hint


# ==========================================
# PostgreSQL 向量检索表映射配置
# ==========================================
# 说明：此配置仅用于 search_knowledge_base 工具的向量语义检索
#       将 category 参数映射到 PostgreSQL 中的向量表名
#       这些表名必须与 etl_vector_local.py 中 VECTOR_TABLES_CONFIG 定义的 target_table 一致
#
# 与 Neo4j 的关系：
# - Neo4j 用于结构化关系查询（Cypher）
# - PostgreSQL 用于语义向量检索（向量相似度）
# - 两者查询的是相同业务实体，只是查询方式不同
#
# 添加新表步骤：
# 1. 在 etl_vector_local.py 的 VECTOR_TABLES_CONFIG 中添加配置
# 2. 运行 ETL 脚本：python scripts/etl_vector_local.py
# 3. 在此添加 category -> 表名 的映射
# ==========================================
PG_VECTOR_TABLE_MAP = {
    # category: 向量表名 (Schema: ORIGIN_NAME)
    # 注意：运行 ETL 脚本后会创建这些向量表
    "defense_area": "防御区_embeddings",  # 防御区向量表 - 用于语义检索防御区描述
    "disaster_body": "承灾体_embeddings", # 承灾体向量表 - 用于语义检索承灾体描述
}

# 保留旧名称兼容性（可选）
TABLE_MAP = PG_VECTOR_TABLE_MAP


# 全局加载模型 (避免每次调用工具都重新加载，耗时)
# 注意：Streamlit 启动时会执行这里，可能会稍微慢几秒
# print("⏳ 正在加载检索模型...")
# RETRIEVER = SentenceTransformer('BAAI/bge-small-zh-v1.5')
# RERANKER = CrossEncoder('BAAI/bge-reranker-base')
# print("✅ 模型加载完毕")

# 延迟加载模型
RETRIEVER = None
RERANKER = None

# 设置国内镜像，加速模型下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def get_retriever():
    """获取嵌入模型（延迟加载）"""
    global RETRIEVER
    if RETRIEVER is None:
        try:
            print("⏳ 正在加载嵌入模型...")
            from sentence_transformers import SentenceTransformer
            # 尝试使用本地模型
            local_model_path = "./models/models--BAAI--bge-small-zh-v1.5"
            if os.path.exists(local_model_path):
                print(f"使用本地模型: {local_model_path}")
                RETRIEVER = SentenceTransformer(local_model_path)
            else:
                print("使用在线模型: BAAI/bge-small-zh-v1.5")
                RETRIEVER = SentenceTransformer('BAAI/bge-small-zh-v1.5')
            print("✅ 嵌入模型已加载")
        except Exception as e:
            print(f"❌ 嵌入模型加载失败: {e}")
            RETRIEVER = None
    return RETRIEVER

def get_reranker():
    """获取重排序模型（延迟加载）"""
    global RERANKER
    if RERANKER is None:
        try:
            print("⏳ 正在加载重排序模型...")
            from sentence_transformers import CrossEncoder
            # 尝试使用本地模型
            local_model_path = "./models/models--BAAI--bge-reranker-base"
            if os.path.exists(local_model_path):
                print(f"使用本地模型: {local_model_path}")
                RERANKER = CrossEncoder(local_model_path)
            else:
                print("使用在线模型: BAAI/bge-reranker-base")
                RERANKER = CrossEncoder('BAAI/bge-reranker-base')
            print("✅ 重排序模型已加载")
        except Exception as e:
            print(f"❌ 重排序模型加载失败: {e}")
            RERANKER = None
    return RERANKER






def _clean_age_data(raw_data):
    """
    (内部函数) 使用正则清洗 AGE 返回的数据，去除 ::vertex, ::edge, ::numeric 等后缀
    """
    # 1. 如果不是字符串（比如已经是数字或None），直接返回
    if not isinstance(raw_data, str):
        return raw_data

    # print(f"[Debug] 清洗前: {raw_data}")

    # 2. 核心修改：使用正则替换，将 "::xxxx" 替换为空字符串
    # r'::\w+' 匹配双冒号后跟任意字母/数字/下划线
    clean_str = re.sub(r'::\w+', '', raw_data)

    # 3. 尝试解析 JSON
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        # 如果不是 JSON（比如只是普通字符串 "Hello"），就返回清洗后的字符串
        return clean_str

@tool
def execute_cypher_query(cypher_query: str) -> str:
    """
    执行 Cypher 查询。
    输入必须是纯 Cypher 语句，例如: MATCH (n:负责人) RETURN {info: n}
    不要包含 SQL 包装。
    """
    print(f"\n[图谱精准检索] 大模型生成的Cypher: {cypher_query}")
    
    driver = None
    try:
        # 连接 Neo4j 数据库
        driver = GraphDatabase.driver(
            NEO4J_CONFIG["uri"],
            auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
        )
        
        with driver.session() as session:
            # 执行 Cypher 查询
            result = session.run(cypher_query)
            
            # 处理结果
            results = []
            for record in result:
                # 将记录转换为字典格式
                record_dict = {}
                for key, value in record.items():
                    if hasattr(value, 'labels'):
                        # 节点对象
                        item = {
                            'label': list(value.labels)[0] if value.labels else None,
                            'properties': dict(value),
                            'id': value.id
                        }
                        record_dict[key] = item
                    elif hasattr(value, 'type'):
                        # 关系对象
                        item = {
                            'label': value.type,
                            'properties': dict(value),
                            'id': value.id,
                            'start_id': value.start_node.id,
                            'end_id': value.end_node.id
                        }
                        record_dict[key] = item
                    else:
                        # 普通值
                        record_dict[key] = value
                results.append(record_dict)
        
        # === 核心修改：零结果处理策略 ===
        if len(results) == 0:
            print("[图谱精准检索] ⚠️ 查询结果为空，返回引导提示")
            return get_zero_results_hint(query_info=cypher_query)
        # ===============================

        print(f"[图谱精准检索] 返回 {len(results)} 条数据")
        print(f"[图谱精准检索] 内容：{results}")
        return json.dumps(results, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"查询失败: {str(e)}"
        print(f"[Tool] ❌ 报错: {error_msg}")
        return error_msg
    finally:
        if driver:
            driver.close()

@tool
def search_knowledge_base(query: str, category: str = "defense_area") -> str:
    """
    通用语义检索工具。
    返回：匹配到的原始 JSON 数据列表。
    """
    print(f"\n[语义检索] 开始执行查询...")
    print(f"[语义检索] 查询内容: {query}")
    print(f"[语义检索] 查询类别: {category}")
    
    # 1. 生成查询向量
    use_random = False
    RETRIEVER = get_retriever()
    if RETRIEVER is not None:
        try:
            query_vector = RETRIEVER.encode(query).tolist()
            print(f"[语义检索] 查询向量生成成功，维度: {len(query_vector)}")
        except Exception as e:
            print(f"[语义检索] 模型编码失败，使用随机向量: {e}")
            use_random = True
    else:
        print("[语义检索] 模型未加载，使用随机向量")
        use_random = True
    
    if use_random:
        # 使用随机向量进行测试
        import random
        query_vector = [random.uniform(-0.1, 0.1) for _ in range(512)]
        print(f"[语义检索] 随机查询向量生成成功，维度: {len(query_vector)}")
    
    # 1. 确定要查哪张表（PostgreSQL 向量表）
    target_table = PG_VECTOR_TABLE_MAP.get(category)
    if not target_table:
        available_categories = list(PG_VECTOR_TABLE_MAP.keys())
        error_msg = f"系统错误: 未知的分类 '{category}'。可用的分类: {available_categories}"
        print(f"[语义检索] ❌ 错误: {error_msg}")
        return error_msg

    print(f"[语义检索] 目标表名: {ORIGIN_NAME}.{target_table}")

    conn = None
    try:
        # 尝试连接数据库，使用pg8000
        try:
            print(f"[语义检索] 尝试连接PostgreSQL数据库: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
            # 使用pg8000连接，添加超时设置
            conn = pg8000.connect(
                host=DB_CONFIG['host'],
                port=int(DB_CONFIG['port']),
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                timeout=10  # 10秒超时
            )
            cursor = conn.cursor()
            print("[语义检索] ✅ PostgreSQL数据库连接成功！")
        except Exception as e:
            print(f"[语义检索] ❌ PostgreSQL数据库连接失败: {str(e)}")
            # 直接返回错误信息，不使用测试模式
            error_msg = f"数据库连接失败：无法连接到语义检索数据库。请检查：1) PostgreSQL服务是否运行；2) 网络是否畅通；3) 配置的数据库地址是否正确。错误详情: {str(e)}"
            return error_msg
        
        # 先检查表是否存在
        try:
            cursor.execute("""
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            """, (ORIGIN_NAME, target_table))
            
            table_exists = cursor.fetchone()
            if not table_exists:
                # 查询所有可用的表
                cursor.execute("""
                    SELECT table_schema, table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = %s
                """, (ORIGIN_NAME,))
                available_tables = cursor.fetchall()
                
                error_msg = f"表不存在: {ORIGIN_NAME}.{target_table}\n"
                if available_tables:
                    error_msg += f"可用的表: {[t[1] for t in available_tables]}"
                else:
                    error_msg += f"Schema '{ORIGIN_NAME}' 中没有找到任何表"
                print(f"[语义检索] ❌ {error_msg}")
                return error_msg
            else:
                print(f"[语义检索] ✅ 表存在: {table_exists[0]}.{table_exists[1]}")
                
                # 检查表中的数据量
                cursor.execute(f'SELECT COUNT(*) FROM "{ORIGIN_NAME}"."{target_table}"')
                count = cursor.fetchone()[0]
                print(f"[语义检索] 表数据量: {count} 条")
                
                if count == 0:
                    return f"表 {ORIGIN_NAME}.{target_table} 存在但没有数据，请先执行数据同步。"
        except Exception as e:
            print(f"[语义检索] ⚠️ 检查表存在时出错: {str(e)}")
        
        # 2. 数据库查询 - 获取所有数据（因为没有pgvector扩展，用Python计算相似度）
        print(f"[语义检索] 执行SQL查询...")
        
        # 检查表结构，确定embedding列的类型
        cursor.execute("""
            SELECT data_type 
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s AND column_name = 'embedding'
        """, (ORIGIN_NAME, target_table))
        
        embedding_col_type = cursor.fetchone()
        if embedding_col_type:
            print(f"[语义检索] embedding列类型: {embedding_col_type[0]}")
        
        # 查询所有数据
        sql = f"""
            SELECT content, full_metadata, embedding
            FROM "{ORIGIN_NAME}"."{target_table}" 
            WHERE content IS NOT NULL
        """
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        print(f"[语义检索] 从数据库获取 {len(rows)} 条记录")
        
        if not rows:
            return f"未在表 {ORIGIN_NAME}.{target_table} 中找到与查询 '{query}' 相关的信息。"
        
        # 2.1 计算向量相似度（在Python中计算，因为没有pgvector扩展）
        import numpy as np
        
        def cosine_similarity(v1, v2):
            """计算余弦相似度"""
            v1 = np.array(v1)
            v2 = np.array(v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))
        
        # 计算查询向量与所有存储向量的相似度
        similarities = []
        for row in rows:
            content, full_metadata, embedding_json = row
            try:
                # 解析存储的向量
                if isinstance(embedding_json, str):
                    stored_vector = json.loads(embedding_json)
                else:
                    stored_vector = embedding_json
                
                # 计算余弦相似度
                similarity = cosine_similarity(query_vector, stored_vector)
                similarities.append({
                    'content': content,
                    'data': full_metadata,
                    'similarity': similarity
                })
            except Exception as e:
                print(f"[语义检索] 计算相似度时出错: {e}")
                continue
        
        # 按相似度排序，取Top 100
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_100 = similarities[:100]
        
        print(f"[语义检索] 向量相似度计算完成，Top 100 相似度范围: {top_100[0]['similarity']:.4f} - {top_100[-1]['similarity']:.4f}")
        
        # 3. 重排序 (Reranking) - 提升精度的关键
        # 检查是否有重排序模型
        RERANKER = get_reranker()
        if RERANKER is not None:
            # 准备数据对: [[query, doc1], [query, doc2]...]
            pairs = [[query, item['content']] for item in top_100]
            
            # 计算相关性分数
            scores = RERANKER.predict(pairs)
            
            # 将分数和原始数据绑定
            ranked_results = []
            for i in range(len(top_100)):
                ranked_results.append({
                    "score": float(scores[i]),
                    "data": top_100[i]['data'] # full_metadata (JSON格式)
                })
                
            # 按分数降序排列，取 Top 20
            ranked_results.sort(key=lambda x: x["score"], reverse=True)
            final_top_5 = ranked_results[:20]
        else:
            # 没有重排序模型，直接使用相似度排序结果
            final_top_5 = [
                {
                    "score": item['similarity'],
                    "data": item['data']
                }
                for item in top_100[:20]
            ]

        print(f"[语义检索] 内容： {final_top_5}")
        
        # 4. 格式化返回 (通用化改造)
        final_response = {
            # 1. 元数据 (Meta Info)：告诉 LLM 这是怎么来的
            "meta_context": {
                "source_tool": "vector_semantic_search", # 明确告知是向量检索
                "retrieval_query": query,                # 明确告知用的什么关键词查的
                "target_category": category,             # 明确告知查的什么分类
                "record_count": len(final_top_5),     # 查到了几条
                "description": "The following data was retrieved based on vector semantic similarity. Please use this context to answer the user's question."
            },
            
            # 2. 数据载荷 (Payload)：纯净的原始数据列表
            "search_results": final_top_5
        }

        # 返回整个大的 JSON 对象
        return json.dumps(final_response, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"检索出错: {str(e)}"
    finally:
        if conn: conn.close()


def generate_graph_from_data(data_list):
    """
    将 AGE 返回的 [{source:..., rel:..., target:...}, ...] 转换为 agraph 的节点和边
    """
    nodes = []
    edges = []
    node_ids = set() # 用于去重，防止重复添加同一个节点炸裂

    for item in data_list:
        # 1. 解析 Source 节点
        if "source" in item:
            src = item["source"]
            src_id = str(src.get("id")) # ID 转字符串
            # 尝试获取显示名称：优先找 '姓名'，其次 'name'，最后用 'label'
            src_label = src.get("properties", {}).get("姓名") or \
                        src.get("properties", {}).get("name") or \
                        src.get("label")
            
            if src_id not in node_ids:
                # size=25 是节点大小，color 是颜色
                nodes.append(Node(id=src_id, label=str(src_label), size=25, shape="dot"))
                node_ids.add(src_id)

        # 2. 解析 Target 节点
        if "target" in item:
            tgt = item["target"]
            tgt_id = str(tgt.get("id"))
            tgt_label = tgt.get("properties", {}).get("姓名") or \
                        tgt.get("properties", {}).get("name") or \
                        tgt.get("label")
            
            if tgt_id not in node_ids:
                nodes.append(Node(id=tgt_id, label=str(tgt_label), size=25, shape="dot"))
                node_ids.add(tgt_id)

        # 3. 解析 Relationship 边
        if "rel" in item and "source" in item and "target" in item:
            rel = item["rel"]
            # start_id 和 end_id 必须和上面 Node 的 id 对应
            # AGE 返回的边包含 start_id 和 end_id
            source_id_ref = str(rel.get("start_id"))
            target_id_ref = str(rel.get("end_id"))
            label = rel.get("label") # 关系名称，如 "核查"
            
            edges.append(Edge(source=source_id_ref, 
                              target=target_id_ref, 
                              label=label,
                              type="CURVE_SMOOTH")) # 线条样式

    # 配置图的物理引擎效果
    config = Config(width="100%", 
                    height=400, 
                    directed=True, 
                    nodeHighlightBehavior=True, 
                    highlightColor="#F7A7A6", # 鼠标悬停颜色
                    collapsible=False)
    
    return nodes, edges, config