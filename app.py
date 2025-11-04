import streamlit as st
from PIL import Image
import os
import requests

# --- 1. 配置和函数定义 ---

# DeepSeek API 的 URL 和模型名称
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-chat"

# (关键) 从 Streamlit Secrets 或本地环境变量中安全地读取 API 密钥
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

def get_deepseek_response(user_prompt):
    """
    调用 DeepSeek API 并返回模型的回复。
    
    参数:
        user_prompt (str): 用户的输入问题。
        
    返回:
        str: DeepSeek 模型的回复内容或错误信息。
    """
    if not DEEPSEEK_API_KEY:
        st.error("错误：DEEPSEEK_API_KEY 未配置。请在 Streamlit Community Cloud 的 Secrets 中设置它。")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    # 构造发送给 API 的消息体
    # 注意：我们目前还没有把图像信息传给模型
    payload = {
        "model": MODEL_NAME,
        "messages": [
            # 系统提示，引导模型的角色和行为
            {"role": "system", "content": "你是一名专业的医疗影像分析助手。请根据用户的问题提供简洁、准确的分析和建议。"},
            # 用户的具体问题
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        # 发送 POST 请求
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        
        # 检查 HTTP 响应状态码
        if response.status_code == 200:
            result = response.json()
            # 提取并返回模型生成的内容
            return result['choices'][0]['message']['content']
        else:
            # 如果 API 返回错误，显示错误信息
            error_message = f"API 请求失败，状态码: {response.status_code}\n响应内容: {response.text}"
            st.error(error_message)
            return None
            
    except requests.exceptions.RequestException as e:
        # 捕获网络连接等异常
        st.error(f"请求时发生网络异常: {e}")
        return None

# --- 2. Streamlit 界面布局 ---

st.set_page_config(page_title="MedRAX 智能影像分析", layout="wide")
st.title("🩺 MedRAX 智能影像分析 (Demo)")

with st.sidebar:
    st.header("上传您的影像")
    uploaded_file = st.file_uploader("请在此处上传胸部 X 光片...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上传的 X 光片", use_column_width=True)

# 初始化或显示聊天记录
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. 核心交互逻辑 ---

if prompt := st.chat_input("您想问什么？(例如：这张片子正常吗？)"):
    
    if uploaded_file is None:
        st.warning("请先在左侧上传一张 X 光片。")
    elif not DEEPSEEK_API_KEY:
        # 这个检查在函数内部也有，但在这里可以提供更即时的反馈
        st.error("管理员未配置 API 密钥，应用无法工作。")
    else:
        # 将用户消息添加到聊天记录并显示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用后端获取回复
        with st.chat_message("assistant"):
            with st.spinner("智能体正在思考中..."):
                # *** 这是本次更新的核心 ***
                # 调用我们新创建的函数来获取真实回复
                response_text = get_deepseek_response(prompt)
                
                if response_text:
                    st.markdown(response_text)
                    # 将助手的有效回复也加入聊天记录
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
