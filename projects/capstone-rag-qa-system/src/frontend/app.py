"""
DocuMind AI - Streamlit 前端主应用
"""

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown(
    """
<style>
    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 深色主题 */
    .stApp {
        background-color: #0f172a;
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* 自定义按钮 */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
    }
    
    /* 自定义输入框 */
    .stTextInput > div > div > input {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #f8fafc;
    }
    
    /* 自定义选择框 */
    .stSelectbox > div > div {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        color: #f8fafc;
    }
    
    /* 文本样式 */
    p, span, label {
        color: #94a3b8;
    }
    
    /* 聊天消息样式 */
    .user-message {
        background-color: #3b82f6;
        color: white;
        padding: 1rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.5rem 0;
        max-width: 70%;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: #1e293b;
        color: #f8fafc;
        padding: 1rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.5rem 0;
        max-width: 85%;
        border: 1px solid #334155;
    }
    
    /* 来源引用卡片 */
    .source-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem 0;
    }
    
    /* 文档卡片 */
    .doc-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .doc-card:hover {
        border-color: #3b82f6;
    }
    
    /* 状态徽章 */
    .badge-success {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
    }
    
    .badge-warning {
        background-color: rgba(234, 179, 8, 0.2);
        color: #eab308;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
    }
    
    .badge-error {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_kb" not in st.session_state:
        st.session_state.current_kb = None

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        # Logo 和标题
        st.markdown("## 🧠 DocuMind AI")
        st.markdown("---")

        # 知识库选择
        st.markdown("### 📚 知识库")

        # TODO: 从 API 获取知识库列表
        kb_options = ["请选择知识库", "技术文档库", "产品手册库"]
        selected_kb = st.selectbox(
            "选择知识库",
            kb_options,
            label_visibility="collapsed",
        )

        if selected_kb != "请选择知识库":
            st.session_state.current_kb = selected_kb

        st.markdown("---")

        # 文档列表
        st.markdown("### 📄 文档列表")

        if st.session_state.current_kb:
            # TODO: 从 API 获取文档列表
            docs = [
                {"name": "技术手册.pdf", "status": "completed"},
                {"name": "API文档.docx", "status": "completed"},
                {"name": "使用说明.md", "status": "processing"},
            ]

            for doc in docs:
                status_class = (
                    "badge-success" if doc["status"] == "completed" else "badge-warning"
                )
                status_text = "✓ 完成" if doc["status"] == "completed" else "⏳ 处理中"
                st.markdown(
                    f"""
                <div class="doc-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #f8fafc;">📄 {doc["name"]}</span>
                        <span class="{status_class}">{status_text}</span>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("请先选择知识库")

        st.markdown("---")

        # 上传按钮
        uploaded_file = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "txt", "md"],
            help="支持 PDF、Word、TXT、Markdown 格式",
        )

        if uploaded_file:
            st.success(f"已选择: {uploaded_file.name}")
            if st.button("📤 开始上传", use_container_width=True):
                # TODO: 调用上传 API
                st.info("上传功能将在后续实现")


def render_chat():
    """渲染聊天区域"""
    # 欢迎消息
    if not st.session_state.messages:
        st.markdown(
            """
        <div style="
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin: 2rem 0;
        ">
            <h2 style="color: #f8fafc; margin-bottom: 1rem;">👋 欢迎使用 DocuMind AI</h2>
            <p style="color: #94a3b8; margin-bottom: 1rem;">
                上传您的文档，然后开始提问。<br>
                我会基于文档内容为您提供精准的回答。
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                <span style="background: #1e293b; padding: 0.5rem 1rem; border-radius: 8px; color: #94a3b8;">
                    📄 支持 PDF、Word、TXT、Markdown
                </span>
                <span style="background: #1e293b; padding: 0.5rem 1rem; border-radius: 8px; color: #94a3b8;">
                    🔍 智能语义检索
                </span>
                <span style="background: #1e293b; padding: 0.5rem 1rem; border-radius: 8px; color: #94a3b8;">
                    💬 多轮对话
                </span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # 显示历史消息
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f"""
            <div class="user-message">
                {message["content"]}
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="assistant-message">
                {message["content"]}
            </div>
            """,
                unsafe_allow_html=True,
            )

            # 显示来源引用
            if "sources" in message and message["sources"]:
                with st.expander("📚 查看来源引用"):
                    for source in message["sources"]:
                        st.markdown(
                            f"""
                        <div class="source-card">
                            <div style="color: #60a5fa; font-weight: 500;">
                                📄 {source.get("filename", "未知文档")}
                            </div>
                            <div style="color: #94a3b8; font-size: 0.875rem; margin-top: 0.5rem;">
                                {source.get("content", "")[:200]}...
                            </div>
                            <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">
                                相关度: {source.get("score", 0) * 100:.1f}%
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )


def render_input():
    """渲染输入区域"""
    # 创建输入表单
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input(
                "问题",
                placeholder="请输入您的问题...",
                label_visibility="collapsed",
            )

        with col2:
            submit_button = st.form_submit_button("发送 ➤", use_container_width=True)

        if submit_button and user_input:
            if not st.session_state.current_kb:
                st.warning("请先选择知识库")
            else:
                # 添加用户消息
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

                # TODO: 调用问答 API
                # 这里先添加占位回复
                assistant_response = {
                    "role": "assistant",
                    "content": f"您好！您的问题是：「{user_input}」\n\n这是一个占位响应。检索和 LLM 生成功能将在后续阶段实现。",
                    "sources": [
                        {
                            "filename": "示例文档.pdf",
                            "content": "这是一个示例来源引用，实际内容将在检索模块完成后显示。",
                            "score": 0.95,
                        }
                    ],
                }
                st.session_state.messages.append(assistant_response)

                # 刷新页面
                st.rerun()


def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()

    # 渲染侧边栏
    render_sidebar()

    # 主内容区
    st.markdown("# 💬 智能问答")

    # 渲染聊天区域
    render_chat()

    # 渲染输入区域
    render_input()


if __name__ == "__main__":
    main()
