import os
import tempfile
import streamlit as st
from rag_module import process_pdf_and_get_chain

st.set_page_config(page_title="AI 문서 Q&A 에이전트", page_icon="📄")
st.title("📄 AI 문서 Q&A 에이전트")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바: PDF 업로드
with st.sidebar:
    st.header("문서 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if uploaded_file is not None and st.session_state.chain is None:
        with st.spinner("문서를 분석하는 중입니다..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            st.session_state.chain = process_pdf_and_get_chain(tmp_path)
            os.remove(tmp_path)

        st.success("✅ 문서 분석 완료")

# 채팅 이력 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 채팅 입력창
if question := st.chat_input("문서 내용에 대해 질문해보세요"):
    if st.session_state.chain is None:
        st.warning("먼저 사이드바에서 PDF 파일을 업로드해주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중입니다..."):
                answer = st.session_state.chain.invoke(question)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        