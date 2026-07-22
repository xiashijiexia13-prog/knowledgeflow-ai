"""Streamlit user interface for the KnowledgeFlow AI HTTP API."""

import os
from typing import Any

import httpx
import streamlit as st


API_BASE_URL = os.getenv(
    "KNOWLEDGEFLOW_API_URL",
    "http://localhost:8000/api/v1",
).rstrip("/")


def main() -> None:
    """Render document management and grounded question-answering controls."""

    st.set_page_config(page_title="KnowledgeFlow AI", page_icon="📚", layout="wide")
    st.title("📚 KnowledgeFlow AI")
    st.caption("基于本地文档、向量检索和 Ollama Qwen 的知识库问答系统")

    documents_tab, chat_tab = st.tabs(["知识库管理", "文档问答"])
    with documents_tab:
        _render_document_management()
    with chat_tab:
        _render_chat()


def _render_document_management() -> None:
    uploads = st.file_uploader(
        "上传 PDF、TXT 或 Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("上传文档", disabled=not uploads):
        for upload in uploads:
            response = _request(
                "POST",
                "/documents/upload",
                files={"file": (upload.name, upload.getvalue(), upload.type)},
            )
            if response is not None:
                st.success(f"已上传：{response['document_name']}")

    col_build, col_refresh = st.columns(2)
    if col_build.button("重建知识库", type="primary"):
        response = _request("POST", "/knowledge-base/build", json={"reset": True})
        if response is not None:
            st.success(
                f"已处理 {response['documents']} 个文档、{response['chunks']} 个片段。"
            )
    refresh = col_refresh.button("刷新文档列表")

    if refresh or "documents" not in st.session_state:
        st.session_state.documents = _request("GET", "/documents") or []
    documents = st.session_state.documents
    if documents:
        st.dataframe(documents, use_container_width=True, hide_index=True)
    else:
        st.info("当前还没有已上传文档。")


def _render_chat() -> None:
    question = st.text_area(
        "问题",
        placeholder="例如：公司的年假申请流程是什么？",
        height=100,
    )
    if st.button("提交问题", type="primary", disabled=not question.strip()):
        response = _request("POST", "/chat", json={"question": question})
        if response is None:
            return
        st.subheader("回答")
        st.write(response["answer"])
        sources = response.get("sources", [])
        if sources:
            st.subheader("引用来源")
            for index, source in enumerate(sources, start=1):
                page = source.get("page_number") or "无页码"
                with st.expander(
                    f"来源{index}：{source['document_name']} · 第 {page} 页 · "
                    f"相似度 {source['score']:.3f}"
                ):
                    st.write(source["excerpt"])


def _request(method: str, path: str, **kwargs: Any) -> Any | None:
    """Call the backend and render a readable error instead of a traceback."""

    try:
        response = httpx.request(
            method,
            f"{API_BASE_URL}{path}",
            timeout=180.0,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        detail = error.response.json().get("detail", error.response.text)
        st.error(f"请求失败：{detail}")
    except (httpx.HTTPError, ValueError) as error:
        st.error(f"无法连接后端服务：{error}")
    return None


if __name__ == "__main__":
    main()
