import streamlit as st
import pickle
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import base64
from pathlib import Path
import re
from langchain_community.retrievers import BM25Retriever

load_dotenv()

def extract_regulation(source_info):
    match = re.search(r'\[([^\]]+)\]\s+([^:]+):\s*"([^"]+)"', source_info)
    if match:
        return {
            'document': match.group(1),
            'section': match.group(2),
            'quote': match.group(3)
        }
    return None

def find_matching_document(documents, regulation):
    for doc in documents:
        if regulation['quote'] in doc.page_content:
            return doc
    return None

def load_resources():
    # Vector DB 로드
    persist_directory = "./chroma_db"
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    
    # BM25Retriever 로드
    with open('bm25_retriever.pkl', 'rb') as f:
        bm25_retriever = pickle.load(f)
    
    # Retriever 설정
    chroma_retriever = vector_db.as_retriever(search_kwargs={'k':5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True,
    )
    
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm
    )
    
    return vector_db, multi_query_retriever, llm, chroma_retriever, ensemble_retriever


def load_pdf_metadata():
    metadata_path = Path('pdf_metadata.pkl')
    if not metadata_path.exists():
        st.error("PDF 메타데이터 파일을 찾을 수 없습니다. 메타데이터를 다시 생성해주세요.")
        return {}
    
    try:
        with open(metadata_path, 'rb') as f:
            pdf_metadata = pickle.load(f)
        return pdf_metadata
    except Exception as e:
        st.error(f"PDF 메타데이터 로딩 중 오류 발생: {str(e)}")
        return {}

def get_conversational_rag_chain(llm, retriever, vector_db):
    def _get_context_retriever_chain(vector_db, ensemble_retriever, llm):
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
        ])
        retriever_chain = create_history_aware_retriever(llm, ensemble_retriever, prompt)
        return retriever_chain

    retriever_chain = _get_context_retriever_chain(vector_db, retriever, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are an assistant designed specifically for answering queries based on company regulations. Always respond strictly according to the company's internal regulations, ensuring your answers are aligned with these rules. 
        When providing an answer, first cite the most relevant regulation in detail, including chapter and section numbers if applicable. If multiple regulations apply, list all relevant ones before giving your response. 
        Your goal is to provide the user with clear guidance based on the regulations, so be as specific as possible with the details of the rules and regulations before proceeding with the final answer.
        If no regulation directly applies, inform the user and give your best guidance based on your knowledge of the company's practices.
        
        After your explanation, provide the exact quotes from the relevant regulations under a "Source Regulations:" section. Format each quote as follows:
        Document Name, Chapter X, Section Y: "Exact quote from the regulation"

        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

try:
    import fitz  # PyMuPDF
except ImportError:
    st.error("PyMuPDF를 찾을 수 없습니다. 'pip install PyMuPDF'를 실행하여 설치해주세요.")
    fitz = None

# PDF 렌더링 함수 수정
def render_pdf_page(file_path, page_number):
    if fitz is None:
        st.error("PyMuPDF가 설치되지 않아 PDF를 렌더링할 수 없습니다.")
        return None
    
    try:
        doc = fitz.open(file_path)
        page = doc.load_page(page_number)  # 페이지 번호는 0부터 시작하므로 1을 빼줍니다
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2배 확대
        img_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        return base64_image
    except Exception as e:
        st.error(f"PDF 페이지 렌더링 중 오류 발생: {str(e)}")
        return None

# Streamlit 앱에서 사용 예시
pdf_metadata = load_pdf_metadata()

# Streamlit 앱 설정
st.title("한국가스공사 사규 질의응답 시스템")

# 리소스 로드
vector_db, multi_query_retriever, llm, chroma_retriever, ensemble_retriever = load_resources()

# 대화 체인 생성
conversation_rag_chain = get_conversational_rag_chain(llm, multi_query_retriever, vector_db)

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in conversation_rag_chain.pick("answer").stream({
            "messages": [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]],
            "input": prompt
        }):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
        # Source Regulations 정보 추출
        if "Source Regulations:" in full_response:
            source_info = full_response.split("Source Regulations:")[1].strip()

            # 인용된 텍스트 추출
            quoted_text = re.search(r'"(.+?)"', source_info)
            if quoted_text:
                query_text = quoted_text.group(1)
            else:
                query_text = source_info

            # 원본 문서 검색
            documents = ensemble_retriever.get_relevant_documents(source_info)
            
            if documents:
                # 가장 관련성 높은 문서 선택 (첫 번째 결과)
                top_doc = documents[0]
                
                # Streamlit 컬럼 생성
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 가장 관련성 높은 원본 문서 정보")
                    st.markdown(f"**파일명:** {top_doc.metadata.get('source', '알 수 없음')}")
                    st.markdown(f"**페이지:** {top_doc.metadata.get('page', '알 수 없음')}")
                    st.markdown(f"**내용:**\n{top_doc.page_content}")
                
                with col2:
                    st.markdown("### PDF 원본 내용")
                    file_path = top_doc.metadata.get('source', '')
                    page_number = top_doc.metadata.get('page', 1)
                    
                    if Path(file_path).exists():
                        if fitz is not None:
                            base64_image = render_pdf_page(file_path, page_number)
                            if base64_image:
                                st.markdown(f'<img src="data:image/png;base64,{base64_image}" style="width:100%">', unsafe_allow_html=True)
                            else:
                                st.write("PDF 페이지를 렌더링할 수 없습니다.")
                        else:
                            st.write("PyMuPDF가 설치되지 않아 PDF를 표시할 수 없습니다.")
                    else:
                        st.write(f"PDF 파일을 찾을 수 없습니다: {file_path}")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})