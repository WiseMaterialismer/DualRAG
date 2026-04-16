import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("向量数据生成脚本")
print("=" * 60)

# 检查模型路径
model_path = "./models/models--BAAI--bge-small-zh-v1.5"
if not os.path.exists(model_path):
    # 尝试另一种路径
    model_path = "./models/bge-small-zh-v1.5"
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("使用随机向量进行测试...")
        use_random = True
    else:
        use_random = False
else:
    use_random = False

if not use_random:
    try:
        from sentence_transformers import SentenceTransformer
        print("\n1. 加载 Embedding 模型...")
        model = SentenceTransformer(model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"⚠️ 模型加载失败: {e}")
        print("使用随机向量进行测试...")
        use_random = True

try:
    from config import DB_CONFIG, ORIGIN_NAME
    import pg8000
    
    print("\n2. 连接数据库...")
    conn = pg8000.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✅ 数据库连接成功")
    
    # 处理防御区表
    print("\n" + "-" * 60)
    print("3. 处理防御区数据")
    print("-" * 60)
    
    # 检查并创建防御区_embeddings表
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS "{ORIGIN_NAME}"."防御区_embeddings" (
            id SERIAL PRIMARY KEY,
            node_id VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            full_metadata JSONB NOT NULL,
            embedding JSONB NOT NULL
        )
    ''')
    print(f"  ✅ 检查/创建防御区_embeddings表成功")
    
    # 读取防御区数据
    cursor.execute(f'SELECT "防御区编码", "风险等级", "防御区等级", "地理位置", "补充描述" FROM "{ORIGIN_NAME}"."防御区" WHERE "补充描述" IS NOT NULL')
    defense_data = cursor.fetchall()
    print(f"  读取到 {len(defense_data)} 条有描述的防御区数据")
    
    if defense_data:
        # 清空旧数据
        cursor.execute(f'TRUNCATE TABLE "{ORIGIN_NAME}"."防御区_embeddings"')
        print(f"  已清空旧数据")
        
        # 准备插入数据
        insert_data = []
        for i, (node_id, risk_level, defense_level, location, content) in enumerate(defense_data):
            if content and len(content.strip()) > 0:
                # 组合所有字段内容用于生成向量
                combined_content = f"防御区编码: {node_id}, 风险等级: {risk_level or '无'}, 防御区等级: {defense_level or '无'}, 地理位置: {location or '无'}, 补充描述: {content}"
                
                # 生成向量
                if use_random:
                    import random
                    vector = [random.uniform(-0.1, 0.1) for _ in range(512)]
                else:
                    vector = model.encode(combined_content).tolist()
                
                # 准备元数据
                metadata = {
                    "防御区编码": node_id,
                    "风险等级": risk_level or "",
                    "防御区等级": defense_level or "",
                    "地理位置": location or "",
                    "补充描述": content[:100]  # 限制长度
                }
                
                insert_data.append((
                    node_id,
                    combined_content,  # 存储组合内容
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(vector)
                ))
                
                if (i + 1) % 20 == 0:
                    print(f"    已处理 {i + 1}/{len(defense_data)} 条...")
        
        # 批量插入
        insert_sql = f'''
            INSERT INTO "{ORIGIN_NAME}"."防御区_embeddings" 
            (node_id, content, full_metadata, embedding)
            VALUES (%s, %s, %s, %s)
        '''
        
        cursor.executemany(insert_sql, insert_data)
        conn.commit()
        print(f"  ✅ 成功插入 {len(insert_data)} 条向量数据")
    
    # 处理承灾体表
    print("\n" + "-" * 60)
    print("4. 处理承灾体数据")
    print("-" * 60)
    
    # 检查并创建承灾体_embeddings表
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS "{ORIGIN_NAME}"."承灾体_embeddings" (
            id SERIAL PRIMARY KEY,
            node_id VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            full_metadata JSONB NOT NULL,
            embedding JSONB NOT NULL
        )
    ''')
    print(f"  ✅ 检查/创建承灾体_embeddings表成功")
    
    cursor.execute(f'SELECT "承灾体编码", "地理位置", "补充描述", "备注", "区域面积" FROM "{ORIGIN_NAME}"."承灾体" WHERE "补充描述" IS NOT NULL')
    disaster_data = cursor.fetchall()
    print(f"  读取到 {len(disaster_data)} 条有描述的承灾体数据")
    
    if disaster_data:
        # 清空旧数据
        cursor.execute(f'TRUNCATE TABLE "{ORIGIN_NAME}"."承灾体_embeddings"')
        print(f"  已清空旧数据")
        
        # 准备插入数据
        insert_data = []
        for i, (node_id, location, content, remark, area) in enumerate(disaster_data):
            if content and len(content.strip()) > 0:
                # 组合所有字段内容用于生成向量
                combined_content = f"承灾体编码: {node_id}, 地理位置: {location or '无'}, 补充描述: {content}, 备注: {remark or '无'}, 区域面积: {area or '无'}"
                
                # 生成向量
                if use_random:
                    import random
                    vector = [random.uniform(-0.1, 0.1) for _ in range(512)]
                else:
                    vector = model.encode(combined_content).tolist()
                
                # 准备元数据
                metadata = {
                    "承灾体编码": node_id,
                    "地理位置": location or "",
                    "补充描述": content[:100],
                    "备注": remark or "",
                    "区域面积": area or ""
                }
                
                insert_data.append((
                    node_id,
                    combined_content,  # 存储组合内容
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(vector)
                ))
                
                if (i + 1) % 20 == 0:
                    print(f"    已处理 {i + 1}/{len(disaster_data)} 条...")
        
        # 批量插入
        insert_sql = f'''
            INSERT INTO "{ORIGIN_NAME}"."承灾体_embeddings" 
            (node_id, content, full_metadata, embedding)
            VALUES (%s, %s, %s, %s)
        '''
        
        cursor.executemany(insert_sql, insert_data)
        conn.commit()
        print(f"  ✅ 成功插入 {len(insert_data)} 条向量数据")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ 向量数据生成完成")
    print("=" * 60)
    if use_random:
        print("⚠️ 注意: 使用的是随机向量，仅用于测试")
        print("   如需真实语义检索，请确保模型文件存在并重新运行")
    
except Exception as e:
    print(f"\n❌ 执行失败: {str(e)}")
    import traceback
    traceback.print_exc()
