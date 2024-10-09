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

load_dotenv()

@st.cache_resource
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
    
    return vector_db, multi_query_retriever, llm

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
        [Document Name] Chapter X, Section Y: "Exact quote from the regulation"

        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Streamlit 앱 설정
st.title("한국가스공사 사규 질의응답 시스템")

# 리소스 로드
vector_db, multi_query_retriever, llm = load_resources()

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
            # 원본 문서 검색
            documents = multi_query_retriever.get_relevant_documents(source_info)
            
            if documents:
                # 관련성 점수 계산 및 정렬
                scored_documents = [(doc, vector_db.similarity_search_with_score(doc.page_content, k=1)[0][1]) for doc in documents]
                scored_documents.sort(key=lambda x: x[1])  # 점수가 낮을수록 더 관련성이 높음
                
                # 가장 관련성 높은 문서 선택
                top_doc, score = scored_documents[0]
                
                st.markdown("### 가장 관련성 높은 원본 문서 정보")
                st.markdown(f"**파일명:** {top_doc.metadata.get('source', '알 수 없음')}")
                st.markdown(f"**페이지:** {top_doc.metadata.get('page', '알 수 없음')}")
                st.markdown(f"**내용:**\n{top_doc.page_content}")
                st.markdown(f"**관련성 점수:** {score}")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})