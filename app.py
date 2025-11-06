import streamlit as st
from PIL import Image
import os
import requests

# --- 核心 Agent 依赖 ---
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

# --- 导入我们创建的工具 ---
try:
    from tools import all_tools
except ImportError:
    st.error("严重错误: 无法导入 'tools.py'。请确保该文件存在于您的仓库中。")
    st.stop()

# --- 1. 配置 ---

# DeepSeek API 的 URL 和模型名称
MODEL_NAME = "deepseek-chat" # 确保这个模型支持 Tool Calling

# (关键) 从 Streamlit Secrets 或本地环境变量中安全地读取 API 密钥
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# 临时文件目录 (我们稍后需要将其添加到 .gitignore)
TEMP_DIR = "temp" 
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


# --- 2. 构建 Agent "大脑" ---

if DEEPSEEK_API_KEY:
    # 1. 初始化 LLM (大脑)
    llm = ChatDeepSeek(model=MODEL_NAME, api_key=DEEPSEEK_API_KEY)
    
    # 2. 获取工具列表 (手脚)
    tools = all_tools

    # 3. 创建 Prompt (灵魂/指令)
    # 这是最关键的部分，它告诉 Agent 它是谁，它该做什么。
    system_prompt = """
    你是一个名为 MedRAX 的高级医疗影像分析智能体。
    你的任务是帮助用户分析胸部 X 光片。

    你将收到以下信息：
    1.  一个用户问题 (`input`)。
    2.  一个本地图像的文件路径 (`image_path`)。
    3.  聊天记录 (`chat_history`)。

    你的工作流程是：
    1.  仔细理解用户的 `input`。
    2.  查看你可用的工具 (`tools`)。
    3.  **你必须使用你的工具来分析图像并回答问题。** 不要凭空捏造答案。
    4.  在调用任何工具时，你**必须**将 `image_path` 作为第一个参数传递。
    5.  `classify_lesion_tool` 工具用于分类或回答“是否有什么”的问题。
    6.  `segment_image_tool` 工具用于定位或“圈出”病灶。
    7.  在调用 `segment_image_tool` 之前，你最好先调用 `classify_lesion_tool` 来获取病灶的描述。
    8.  `segment_image_tool` 会返回一个*新*的、已标记的图像路径 (例如: 'segmented_result.png')。
        在你的最终回复中，你必须告诉用户这个新文件的路径。
    9.  用中文回复用户。
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        # 'image_path' 将作为上下文传递给 Agent
        ("human", "问题: {input}\n(请分析这个图像: {image_path})"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Agent 思考/工具输出的地方
    ])

    # 4. 创建 Agent (大脑+手脚)
    agent = create_tool_calling_agent(llm, tools, prompt_template)

    # 5. 创建 Agent 执行器 (运行循环)
    # verbose=True 会在日志中打印 Agent 的思考过程，方便调试
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

else:
    # 如果没有 API 密钥，则不创建 Agent
    agent_executor = None

# --- 3. Streamlit 界面 ---

st.set_page_config(page_title="MedRAX 智能影像分析", layout="wide")
st.title("🩺 MedRAX 智能影像分析 (Agent 驱动版)")

with st.sidebar:
    st.header("1. 上传影像")
    uploaded_file = st.file_uploader("请在此处上传胸部 X 光片...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上传的 X 光片", use_column_width=True)
        
    st.header("2. API 状态")
    if DEEPSEEK_API_KEY:
        st.success("DeepSeek API 密钥已配置！")
    else:
        st.error("API 密钥未配置！")
        st.info("请在 Streamlit Cloud 的 'Secrets' 中添加 `DEEPSEEK_API_KEY = 'sk-...'`")


# 初始化聊天记录
# 'messages' 用于在 UI 上显示
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
# 'agent_history' 用于给 LangChain Agent 提供记忆
if 'agent_history' not in st.session_state:
    st.session_state.agent_history = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果消息是 AI 生成的，并且包含了图片，就显示图片
        if "image_path" in message:
            st.image(message["image_path"], caption="AI 标记的图像")


# --- 4. 核心交互逻辑 ---

if prompt := st.chat_input("您想问什么？(例如：这张片子正常吗？)"):
    
    # 检查1：是否上传了图片
    if uploaded_file is None:
        st.warning("请先在左侧上传一张 X 光片。")
    # 检查2：Agent 是否已成功初始化
    elif agent_executor is None:
        st.error("错误：Agent 执行器未初始化。请检查 API 密钥。")
    else:
        # --- 准备阶段 ---
        
        # 1. 保存上传的文件到临时路径，因为工具需要一个文件路径
        temp_image_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. 在 UI 上显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- 执行阶段 ---
        with st.chat_message("assistant"):
            with st.spinner("智能体正在思考并调用工具..."):
                
                # 准备 Agent 的输入
                agent_input = {
                    "input": prompt,
                    "image_path": temp_image_path,
                    "chat_history": st.session_state.agent_history
                }
                
                # 3. (关键) 调用 Agent 执行器
                try:
                    response = agent_executor.invoke(agent_input)
                    response_text = response["output"]
                    
                    # 4. 更新 Agent 自己的记忆
                    st.session_state.agent_history.append(HumanMessage(content=prompt))
                    st.session_state.agent_history.append(AIMessage(content=response_text))
                    
                    # --- 响应阶段 ---
                    
                    # 5. 检查 Agent 的回复是否提到了“已标记”的图片
                    # (这是基于我们 'tools.py' 中返回的硬编码 'segmented_result.png')
                    new_image_path = None
                    if "segmented_result.png" in response_text:
                        if os.path.exists("segmented_result.png"):
                            new_image_path = "segmented_result.png"
                    
                    # 6. 在 UI 上显示最终回复
                    st.markdown(response_text)
                    if new_image_path:
                        st.image(new_image_path, caption="AI 标记的图像")

                    # 7. 保存带图片路径的 UI 消息
                    ui_message = {"role": "assistant", "content": response_text}
                    if new_image_path:
                        ui_message["image_path"] = new_image_path
                    st.session_state.messages.append(ui_message)

                except Exception as e:
                    error_msg = f"Agent 执行出错: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
