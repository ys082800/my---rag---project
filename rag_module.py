import os
from dotenv import load_dotenv
import fitz  # PyMuPDF

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. 문서 로드 (Document Load)
def load_pdf_text(pdf_path: str) -> str:
    """PDF 파일에서 텍스트를 추출합니다."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# 2. 분할 (Text Split) - 400자 단위 청크
def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    return splitter.split_text(text)


# 3~4. 임베딩 + 벡터 DB(FAISS) 저장
def build_vectorstore(chunks: list):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_texts(chunks, embedding=embeddings)


# 5. 검색기(Retriever) 생성
def get_retriever(vectorstore, k: int = 5):
    return vectorstore.as_retriever(search_kwargs={"k": k})


# 6. 프롬프트 (형식 지정 기법 적용)
PROMPT_A = """\
#명령문
당신은 RFP 문서를 분석하고 핵심 정보를 정확하게 설명하는 AI 어시스턴트입니다.

#제약조건
1. 반드시 제공된 문서만 근거로 답변하세요.
2. 문서에 없는 내용은 추측하지 마세요.
3. 질문에 포함된 요구사항을 빠짐없이 답변하세요.
4. 숫자, 날짜, 요구사항 코드 및 평가기준은 문서에 명시된 내용을 정확히 반영하세요.
5. 답변은 핵심 내용부터 제시하고 필요한 경우 세부 내용을 추가하세요.

#답변 예시

Q: 이 사업의 기술평가와 가격평가는 각각 몇 점인가?

A:
## 핵심 답변
기술평가는 80점, 가격평가는 20점으로 총 100점입니다.

## 상세 내용
- 기술평가: 80점
- 가격평가: 20점
- 기술평가는 정량평가와 정성평가로 구성됩니다.

## 근거
문서의 제안서 평가방법에 기술능력 80%, 입찰가격 20%의 평가비율이 제시되어 있습니다.

#입력문
문서:
{context}

질문:
{question}

#출력형식
## 핵심 답변
[핵심 답변]

## 상세 내용
[세부 설명]

## 근거
[문서 근거]
"""

def get_prompt():
    return ChatPromptTemplate.from_template(PROMPT_A)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 7~8. LLM 답변 생성 + 출력
def build_rag_chain(retriever):
    prompt = get_prompt()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def process_pdf_and_get_chain(pdf_path: str): 
    """PDF 경로를 받아 전체 파이프라인을 수행하고 최종 체인을 반환합니다."""
    text = load_pdf_text(pdf_path) 
    chunks = split_text(text) 
    vectorstore = build_vectorstore(chunks) 
    retriever = get_retriever(vectorstore) 
    return build_rag_chain(retriever)