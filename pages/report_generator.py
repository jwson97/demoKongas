import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
import matplotlib.font_manager as fm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import os
import json
from dotenv import load_dotenv

# 폰트 경로 설정
FONT_PATH_REGULAR = os.path.join(os.path.dirname(__file__), 'NanumGothic-Regular.ttf')
FONT_PATH_BOLD = os.path.join(os.path.dirname(__file__), 'NanumGothic-Bold.ttf')
FONT_PATH_EXTRABOLD = os.path.join(os.path.dirname(__file__), 'NanumGothic-ExtraBold.ttf')

# ReportLab에 폰트 등록
pdfmetrics.registerFont(TTFont('NanumGothic', FONT_PATH_REGULAR))
pdfmetrics.registerFont(TTFont('NanumGothic-Bold', FONT_PATH_BOLD))
pdfmetrics.registerFont(TTFont('NanumGothic-ExtraBold', FONT_PATH_EXTRABOLD))

# Matplotlib 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
fm.fontManager.addfont(FONT_PATH_REGULAR)
fm.fontManager.addfont(FONT_PATH_BOLD)
fm.fontManager.addfont(FONT_PATH_EXTRABOLD)
plt.rcParams['font.family'] = 'NanumGothic'

def set_custom_style():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #007bff;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
        color: #333333;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border-left: 5px solid #007bff;
    }
    .chat-message.bot {
        background-color: #f8f9fa;
        border-left: 5px solid #28a745;
    }
    .chat-message .content {
        width: 100%;
    }
    .chat-message .content p {
        margin: 0;
        color: #333333;
    }
    h1, h2, h3 {
        color: #007bff;
    }
    .stAlert {
        background-color: #fff3cd;
        color: #856404;
    }
    .stFileUploader {
        background-color: #ffffff;
        border: 1px dashed #007bff;
        border-radius: 5px;
        padding: 10px;
    }
    .stFileUploader > div > div > div > div {
        color: #333333;
    }
    .stAlert > div {
        background-color: #e2e3e5;
        color: #383d41;
        border: 1px solid #d6d8db;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_chart(data, chart_type='bar', title=''):
    fig, ax = plt.subplots(figsize=(6, 4))
    if chart_type == 'bar':
        ax.bar(data.keys(), data.values(), color='#1f77b4')
        ax.set_ylabel("통지갯수", fontsize=10)
        for i, v in enumerate(data.values()):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
        plt.xticks(rotation=0, ha='center', fontsize=9)
    elif chart_type == 'line':
        ax.plot(list(data.keys()), list(data.values()), marker='o', color='#1f77b4')
        ax.set_ylabel("통지갯수", fontsize=10)
        for i, v in enumerate(data.values()):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
        plt.xticks(rotation=0, ha='center', fontsize=9)
    elif chart_type == 'pie':
        wedges, texts, autotexts = ax.pie(data.values(), autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.legend(wedges, data.keys(), title="사유", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
        plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def get_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

def create_pdf(data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20, bottomMargin=20, leftMargin=30, rightMargin=30)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Korean', fontName='NanumGothic', fontSize=10, leading=14))
    styles.add(ParagraphStyle(name='KoreanTitle', fontName='NanumGothic', fontSize=16, leading=20, alignment=1))
    styles.add(ParagraphStyle(name='KoreanSubtitle', fontName='NanumGothic', fontSize=14, leading=18))
    
    elements = []
    
    # PDF 생성 로직 (이전과 동일)
    # ...

    doc.build(elements)
    buffer.seek(0)
    return buffer

def app():
    set_custom_style()
    
    st.title("생성형 AI를 활용한 보고서 생성")
    st.write("생성형 AI를 활용해 보고서, 공문, 엑셀 파일 등을 자동 생성하는 기능입니다.")

    # 세션 상태 초기화
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'show_preview_button' not in st.session_state:
        st.session_state.show_preview_button = False

    # 파일 업로드
    uploaded_file = st.file_uploader("JSON 파일을 업로드하세요", type="json")    
    
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            st.session_state.file_uploaded = True
            st.session_state.data = data
            st.success("JSON 파일이 성공적으로 업로드되었습니다.")
        except json.JSONDecodeError:
            st.error("올바른 JSON 형식이 아닙니다. 파일을 확인해 주세요.")
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
    
    # 채팅 인터페이스
    if st.session_state.file_uploaded:
        user_input = st.text_input("보고서 작성 지시를 입력하세요:")
        if st.button("전송"):
            if user_input:
                st.write(f"입력받은 내용: {user_input}")
                st.session_state.show_preview_button = True
            else:
                st.warning("입력 내용이 없습니다. 보고서 작성 지시를 입력해주세요.")
    
    # 보고서 미리보기 버튼
    if st.session_state.show_preview_button:
        if st.button("보고서 미리보기"):
            data = st.session_state.data
            st.subheader(data['title'])
            st.write(data['subtitle'])
            st.write(data['report_title'])
            st.write(data['date'])
            
            for section in data['sections']:
                st.subheader(section['title'])
                if 'content' in section:
                    if isinstance(section['content'], list):
                        for content in section['content']:
                            st.write(content)
                    else:
                        st.write(section['content'])
                
                if 'subsections' in section:
                    for subsection in section['subsections']:
                        st.write(subsection['title'])
                        if 'table' in subsection:
                            st.table(pd.DataFrame(subsection['table'][1:], columns=subsection['table'][0]))
                        
                        if 'chart' in subsection:
                            chart = create_chart(subsection['chart']['data'], subsection['chart']['type'], subsection['chart']['title'])
                            st.pyplot(chart)

            # PDF 다운로드 버튼
            pdf_buffer = create_pdf(data)
            pdf_bytes = pdf_buffer.getvalue()
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="report.pdf">PDF로 저장하기</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    app()