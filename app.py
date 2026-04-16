# app.py
import streamlit as st
import pandas as pd
import json
import pg8000
from neo4j import GraphDatabase
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from streamlit_agraph import agraph, Node, Edge, Config

# --- 导入解耦的模块 ---
from config import DB_CONFIG, GRAPH_NAME, LLM_MODEL_NAME, NEO4J_CONFIG
from tools import execute_cypher_query, generate_graph_from_data, search_knowledge_base
from prompts import get_system_prompt       
from memory import build_chat_context       

# ================== 0. 连接状态检查 ==================
@st.cache_resource
def test_database_connections():
    """检查 Neo4j 和 PostgreSQL 的真实连接状态"""
    neo4j_status = "❌ 未检测"
    postgres_status = "❌ 未检测"

    try:
        driver = GraphDatabase.driver(
            NEO4J_CONFIG["uri"],
            auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
        )
        with driver.session() as session:
            session.run("RETURN 1 AS result").single()
        neo4j_status = "✅ 已连接"
    except Exception as e:
        neo4j_status = f"❌ 连接失败: {str(e)}"
    finally:
        try:
            if driver:
                driver.close()
        except Exception:
            pass

    try:
        conn = pg8000.connect(
            host=DB_CONFIG["host"],
            port=int(DB_CONFIG["port"]),
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            timeout=5
        )
        conn.close()
        postgres_status = "✅ 已连接"
    except Exception as e:
        postgres_status = f"❌ 连接失败: {str(e)}"

    return {
        "neo4j": neo4j_status,
        "postgres": postgres_status
    }

# ================== 1. 页面配置 ==================
st.set_page_config(page_title="地灾数据助手", page_icon="🌍", layout="centered")

# ================== 2. 侧边栏配置 ==================
with st.sidebar:
    st.header("⚙️ 设置面板")
    
    # --- 记忆策略控制 ---
    st.subheader("🧠 记忆设置")
    memory_type = st.radio(
        "记忆模式",
        ("滑动窗口 (推荐)", "全量记忆 (Token消耗大)", "不记忆 (单轮对话)"),
        index=0
    )
    
    # 映射 UI 选项到代码策略 key
    strategy_map = {
        "滑动窗口 (推荐)": "window",
        "全量记忆 (Token消耗大)": "full",
        "不记忆 (单轮对话)": "none"
    }
    selected_strategy = strategy_map[memory_type]
    
    # 只有选滑动窗口时才显示滑块
    window_k = 6
    if selected_strategy == "window":
        window_k = st.slider("记忆轮数 (消息条数)", min_value=2, max_value=20, value=6, step=2)

    st.divider()
    
    # --- 常用功能 ---
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 💡 快捷提问")
    example_questions = ["朱炳湖负责的防御区中面积最大的是哪个？",
                         "哪些防御区风险等级是中级？",
                         "承灾体里威胁财产最多的前5个？",
                         "哪些防御区是坡度较缓",
                         "人工切坡高2米的防御区具体信息，以及对应的负责人是谁"]
    for q in example_questions:
        if st.button(q):
            st.session_state.current_prompt = q

# ================== 3. 主界面 & Agent 初始化 ==================
conn_status = test_database_connections()

st.title(f"🌍 地灾数据智能助手")
st.caption(
    f"当前连接图谱: `{GRAPH_NAME}` | Neo4j: {conn_status['neo4j']} | PostgreSQL: {conn_status['postgres']} | 记忆模式: `{memory_type}`"
)

@st.cache_resource
def get_agent_instance():
    # 初始化模型
    llm = ChatTongyi(model_name=LLM_MODEL_NAME, temperature=0) # type: ignore
    tools = [execute_cypher_query, search_knowledge_base]
    
    # 从 prompts.py 获取提示词
    system_prompt = get_system_prompt()
    print(f"[app] 提示词： {system_prompt}")
    
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)

agent = get_agent_instance()

# ================== 4. 渲染历史与处理输入 ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if isinstance(msg, HumanMessage) else "🤖"
    with st.chat_message(msg.type, avatar=avatar):
        st.markdown(msg.content)

# 2. 获取输入（关键修改：让 chat_input 始终渲染）
chat_input_text = st.chat_input("请输入查询内容...")
button_input_text = st.session_state.get("current_prompt", None)

# 3. 决定最终使用哪个输入 (优先响应按钮，其次响应输入框)
user_input = None

if button_input_text:
    user_input = button_input_text
    # 消费掉这个状态，防止刷新后死循环
    del st.session_state["current_prompt"]
elif chat_input_text:
    user_input = chat_input_text

if user_input:
    if "current_prompt" in st.session_state: del st.session_state["current_prompt"]

    # 1. UI 显示用户问题
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # 2. 调用 Agent
    with st.chat_message("assistant", avatar="🤖"):
        msg_placeholder = st.empty()
        status = st.status("🧠 思考中...", expanded=True)
        
        try:
            status.write(f"正在构建上下文 (策略: {selected_strategy})...")
            
            # [新] 调用 memory.py 构建上下文
            input_payload = build_chat_context(
                current_prompt=user_input,
                history=st.session_state.messages,
                strategy=selected_strategy,
                k=window_k
            )
            
            # Agent 执行
            result = agent.invoke(input_payload) # type: ignore
            # result = agent.invoke(input_data)

            final_response = result['messages'][-1].content

            # --- 数据提取逻辑 ---
            raw_data_json = None # 保存原始 JSON 列表
            raw_data_df = None   # 保存表格 DF

            print(f"[App] Agent回复的信息：{result['messages']}")

            # 提取数据
            raw_data_list = []
            for msg in result['messages']:
                if isinstance(msg, ToolMessage):
                    try:
                        # 1. 解析 JSON
                        data = json.loads(msg.content) # type: ignore
                        
                        # 2. 情况 A: 图谱查询 (直接返回 List)
                        if isinstance(data, list):
                            raw_data_list.extend(data) # 建议用 extend 而不是 =，防止被覆盖
                        
                        # 3. 情况 B: 语义检索 (返回 Dict，数据在 'search_results' 里)
                        elif isinstance(data, dict):
                            # 优先找 search_results
                            if 'search_results' in data and isinstance(data['search_results'], list):
                                raw_data_list.extend(data['search_results'])
                            # 兼容性兜底: 如果以后有其他返回 dict 的工具，也可以在这里处理
                            
                    except Exception as e:
                        print(f"数据解析失败: {e}")
                        pass
            
            # === 1. 展示图谱 (新增功能) ===
            # if raw_data_jPson is not None:
            #     # 只有当数据里包含 source/target 结构时才画图，防止纯节点查询报错
            #     # 简单的判断逻辑：看第一条数据有没有 'source' 键
            #     if "source" in raw_data_json[0] and "target" in raw_data_json[0]:
            #         with st.expander("🕸️ 知识图谱可视化", expanded=True):
            #             nodes, edges, config = generate_graph_from_data(raw_data_json)
            #             # 渲染图谱
            #             agraph(nodes=nodes, edges=edges, config=config)
            
            # === 2. 展示表格 (原有功能) ===
            print(f'展示表格数据: {raw_data_list}')
            # 显示表格
            if raw_data_list:
                # 处理数据，展平嵌套结构并使用更通用的列名
                processed_data = []
                for item in raw_data_list:
                    processed_item = {}
                    # 处理嵌套的节点和关系
                    for key, value in item.items():
                        if isinstance(value, dict):
                            # 处理节点对象
                            if 'properties' in value:
                                # 处理节点的属性
                                for prop_key, prop_value in value['properties'].items():
                                    # 使用通用的列名格式：属性名
                                    if 'label' in value:
                                        node_label = value['label']
                                        processed_item[f"{node_label}_{prop_key}"] = prop_value
                                    else:
                                        processed_item[f"{key}_{prop_key}"] = prop_value
                            elif 'label' in value:
                                # 处理关系对象
                                processed_item['关系类型'] = value['label']
                            else:
                                # 处理其他类型的字典
                                for sub_key, sub_value in value.items():
                                    processed_item[sub_key] = sub_value
                        else:
                            # 普通值
                            processed_item[key] = value
                    processed_data.append(processed_item)
                
                # 检查处理后的数据
                print(f'处理后的数据: {processed_data}')
                
                df = pd.DataFrame(processed_data)
                print(f'DataFrame形状: {df.shape}')
                print(f'DataFrame列: {list(df.columns)}')
                
                # 清理列名，移除特殊字符和前缀
                def clean_column_name(col):
                    # 移除特殊字符
                    col = col.replace('{', '').replace('}', '').replace(':', '').replace('"', '').replace(' ', '_')
                    col = col.replace('(', '').replace(')', '').replace(',', '').replace(';', '')
                    # 移除属性名中的前缀，如'p.'、'd.'等
                    if '.' in col:
                        # 分割并保留最后一部分
                        parts = col.split('.')
                        if len(parts) > 1:
                            # 重建列名，移除所有前缀
                            col = '_'.join([parts[0].split('_')[0]] + parts[1:])
                    # 移除重复的下划线
                    while '__' in col:
                        col = col.replace('__', '_')
                    # 移除开头和结尾的下划线
                    col = col.strip('_')
                    return col
                
                df.columns = [clean_column_name(col) for col in df.columns]
                
                # 过滤掉全为None的列
                df = df.dropna(axis=1, how='all')
                
                st.dataframe(df)

            status.update(label="✅ 完成", state="complete", expanded=False)
            
            
            # 显示文本回复
            msg_placeholder.markdown(final_response)
            
            # 3. 更新历史 (成功后才存入)
            # st.session_state.messages.append(HumanMessage(content=user_input))
            # st.session_state.messages.append(AIMessage(content=final_response))
            
        except Exception as e:
            status.update(label="❌ 出错", state="error")
            msg_placeholder.error(f"Error: {str(e)}")