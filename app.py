import streamlit as st
from multiapp.MultiApp import MultiApp
from pages import chatbot, scheduler, report_generator

from dotenv import load_dotenv

# 페이지 설정 (상단의 'app'과 같은 기본 요소를 없애는 코드)
st.set_page_config(page_title="KOGAS 업무용 AI 비서", layout="wide")

__import__('pysqlite3')
import sys



sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()  # 이 함수를 호출하면 .env 파일의 내용이 환경 변수로 로드됩니다


# 멀티페이지 앱 실행
app = MultiApp()

# 각 페이지 추가
app.add_app("공사 챗봇 서비스", chatbot.app)
app.add_app("공사 특화 스케쥴러", scheduler.app)
app.add_app("생성형 AI를 활용한 보고서 생성", report_generator.app)

app.run()
