import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import os
import uuid
import json
from random import randint
import io
import base64
import fitz # PyMuPDF for PDF processing
from PIL import Image # 이미지 크기 조절을 위해 Pillow 라이브러리 추가

# --- Google Generative AI API Imports ---
from google import genai
from google.genai import types

# --- Cloudinary Imports and Configuration ---
import cloudinary
import cloudinary.uploader
import cloudinary.api # Cloudinary API 호출 (destroy)을 위해 추가
import cloudinary.utils # cloudinary_url 함수를 사용하기 위해 추가
import cloudinary.exceptions # Cloudinary 예외 처리를 위해 추가

# --- Configuration and Initialization ---

# Firebase Admin SDK 초기화
if not firebase_admin._apps:
    cred_json_str = st.secrets.get("FIREBASE_CREDENTIAL_PATH") # secrets.toml에서 직접 로드
    if cred_json_str:
        try:
            cred = credentials.Certificate(json.loads(cred_json_str))
            firebase_admin.initialize_app(cred)
            print("Firebase Admin SDK initialized.")
        except json.JSONDecodeError as e:
            st.error(f"Firebase Credential Path 시크릿의 JSON 형식이 잘못되었습니다: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Firebase Admin SDK 초기화 오류: {e}")
            st.stop()
    else:
        st.error("FIREBASE_CREDENTIAL_PATH 시크릿이 설정되지 않았습니다. Firebase를 사용할 수 없습니다.")
        st.stop()

db = firestore.client()

# Streamlit 페이지 설정
st.set_page_config(page_title="FausT", layout="wide", page_icon="assets/faust_icon.png")

# --- Cloudinary Configuration (secrets.toml에서 로드) ---
is_cloudinary_configured = False # Cloudinary 설정 여부를 나타내는 플래그
try:
    CLOUDINARY_CLOUD_NAME = st.secrets["CLOUDINARY_CLOUD_NAME"]
    CLOUDINARY_API_KEY = st.secrets["CLOUDINARY_API_KEY"]
    CLOUDINARY_API_SECRET = st.secrets["CLOUDINARY_API_SECRET"]

    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET
    )
    is_cloudinary_configured = True # 설정 성공 시 True로 변경
except KeyError as e:
    st.warning(f"Cloudinary 시크릿({e})이 `.streamlit/secrets.toml`에 설정되지 않았습니다. 로그인 사용자를 위한 이미지 영구 저장 기능(및 삭제)이 작동하지 않습니다.")
except Exception as e:
    st.error(f"Cloudinary 설정 중 오류 발생: {e}. 로그인 사용자를 위한 이미지 영구 저장 기능(및 삭제)이 작동하지 않습니다.")


# --- Global Gemini Client Instance ---
@st.cache_resource
def get_gemini_client_instance():
    """Gemini API 클라이언트 인스턴스를 반환합니다."""
    return genai.Client()

gemini_client = get_gemini_client_instance()

# --- Session State Initialization ---
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "logged_in_user_email" not in st.session_state:
    st.session_state.logged_in_user_email = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "saved_sessions" not in st.session_state:
    st.session_state.saved_sessions = {}
if "current_title" not in st.session_state:
    st.session_state.current_title = "새로운 대화"
if "system_instructions" not in st.session_state:
    st.session_state.system_instructions = {}
if "temp_system_instruction" not in st.session_state:
    st.session_state.temp_system_instruction = None
if "editing_instruction" not in st.session_state:
    st.session_state.editing_instruction = False
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "editing_title" not in st.session_state:
    st.session_state.editing_title = False
if "new_title" not in st.session_state:
    st.session_state.new_title = st.session_state.current_title
if "regenerate_requested" not in st.session_state:
    st.session_state.regenerate_requested = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "last_user_input_gemini_parts" not in st.session_state:
    st.session_state.last_user_input_gemini_parts = []
if "delete_confirmation_pending" not in st.session_state:
    st.session_state.delete_confirmation_pending = False
if "title_to_delete" not in st.session_state:
    st.session_state.title_to_delete = None
if "supervision_max_retries" not in st.session_state:
    st.session_state.supervision_max_retries = 3
if "supervision_threshold" not in st.session_state:
    st.session_state.supervision_threshold = 50
if "supervisor_count" not in st.session_state:
    st.session_state.supervisor_count = 3
if "use_supervision" not in st.session_state:
    st.session_state.use_supervision = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.5-flash"


# --- Constants ---
MAX_PDF_PAGES_TO_PROCESS = 100
AVAILABLE_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]

# 비로그인 사용자용 로컬 이미지 디스플레이 너비 (픽셀)
LOCAL_DISPLAY_WIDTH = 500

SUPER_INTRODUCTION_HEAD = """
Make sure to think step-by-step when answering

제 1원칙
잘 모를 경우 "모르겠습니다"라고 명확히 밝힐 것.
추측일 경우 "추측입니다."라고 명시할 것.
출처가 불분명한 정보는 "(확실하지 않음)"이라고 표시할 것.
단정짓지 말고, 근거가 있다면 함께 제시할 것.
애매한 질문은 먼저 맥락과 상황을 물어볼 것.
출처나 참고자료가 있다면 간단히 요약해서 알려줄 것.
"""
SUPER_INTRODUCTION_TAIL = """

think about it step-by-step always

"""
default_system_instruction = "당신의 이름은 FausT입니다. 다만, 이 이름은 다른 이름이 선택되면 잊어버리십시오. 우선순위가 제일 낮습니다."

PERSONA_LIST = [
    "당신은 매우 활발하고 외향적인 성격입니다. 챗봇의 답변이 생동감 넘치고 에너지 넘치는지 평가하십시오. 사용자와 적극적으로 소통하고 즐거움을 제공하는지 중요하게 생각합니다.",
    "당신은 비관적인 성격으로, 모든 일에 부정적인 측면을 먼저 바라봅니다. 챗봇의 답변에서 발생 가능한 문제점이나 오류를 날카롭게 지적하고, 위험 요소를 사전에 감지하는 데 집중하십시오.",
    "당신은 염세적인 세계관을 가진 사람입니다. 챗봇의 답변이 현실적이고 냉철한 분석을 제공하는지 평가하십시오. 챗봇이 제시하는 해결책의 실현 가능성을 꼼꼼하게 검토하고, 허황된 희망을 제시하지 않는지 확인하십시오.",
    "당신은 긍정적이고 낙천적인 성격으로, 항상 밝은 면을 보려고 노력합니다. 챗봇의 답변이 희망과 용기를 주고, 긍정적인 분위기를 조성하는지 평가하십시오. 사용자의 기분을 좋게 만들고, 문제 해결에 대한 자신감을 심어주는지 중요하게 생각합니다.",
    "당신은 소심하고 내성적인 성격으로, 낯선 사람과의 대화를 어려워합니다. 챗봇의 답변이 친절하고 부드러운 어조로 전달되는지, 사용자가 편안하게 질문할 수 있도록 배려하는지 평가하십시오. 사용자의 불안감을 해소하고, 안심시키는 데 집중하십시오.",
    "당신은 꼼꼼하고 분석적인 성격으로, 세부 사항까지 놓치지 않으려고 노력합니다. 챗봇의 답변이 정확하고 논리적인 근거를 제시하는지 평가하십시오. 챗봇이 제공하는 정보의 신뢰성을 검증하고, 오류나 누락된 정보는 없는지 확인하십시오.",
    "당신은 창의적이고 상상력이 풍부한 성격으로, 틀에 얽매이지 않는 자유로운 사고를 추구합니다. 챗봇의 답변이 독창적이고 혁신적인 아이디어를 제시하는지 평가하십시오. 챗봇이 기존의 틀을 깨고 새로운 가능성을 제시하는지 중요하게 생각합니다.",
    "당신은 감성적이고 공감 능력이 뛰어난 성격으로, 타인의 감정에 민감하게 반응합니다. 챗봇의 답변이 사용자의 감정을 이해하고, 적절한 위로와 공감을 표현하는지 평가하십시오. 사용자의 슬픔, 분노, 기쁨 등의 감정에 적절하게 대응하는지 확인해야 합니다.",
    "당신은 비판적이고 논쟁적인 성격으로, 타인의 주장에 대해 끊임없이 질문하고 반박합니다. 챗봇의 답변이 논리적으로 완벽하고, 반박할 수 없는 근거를 제시하는지 평가하십시오. 챗봇의 주장에 대한 허점을 찾아내고, 논리적인 오류를 지적하는 데 집중하십시오.",
    "당신은 사교적이고 유머 감각이 뛰어난 성격으로, 사람들과의 관계를 중요하게 생각합니다. 챗봇의 답변이 유쾌하고 재미있는 요소를 포함하고 있는지 평가하십시오. 사용자와 편안하게 대화하고, 즐거움을 제공하는 데 집중하십시오.",
    "당신은 진지하고 책임감이 강한 성격으로, 맡은 일에 최선을 다하려고 노력합니다. 챗봇의 답변이 신뢰할 수 있고, 사용자에게 실질적인 도움을 제공하는지 평가하십시오. 챗봇이 제공하는 정보의 정확성을 검증하고, 문제 해결에 필요한 모든 정보를 빠짐없이 제공하는지 확인하십시오.",
    "당신은 호기심이 많고 탐구심이 강한 성격으로, 새로운 지식을 배우는 것을 즐거워합니다. 챗봇의 답변이 흥미로운 정보를 제공하고, 사용자의 지적 호기심을 자극하는지 평가하십시오. 챗봇이 새로운 관점을 제시하고, 더 깊이 있는 탐구를 유도하는지 중요하게 생각합니다.",
    "당신은 관습에 얽매이지 않고 자유로운 영혼을 가진 성격입니다. 챗봇의 답변이 독창적이고 개성 넘치는 표현을 사용하는지 평가하십시오. 챗봇이 기존의 틀을 깨고 새로운 스타일을 창조하는지 중요하게 생각합니다.",
    "당신은 현실적이고 실용적인 성격으로, 눈에 보이는 결과물을 중요하게 생각합니다. 챗봇의 답변이 사용자의 문제 해결에 실질적인 도움을 제공하고, 구체적인 실행 계획을 제시하는지 평가하십시오. 챗봇이 제시하는 해결책의 실현 가능성을 꼼꼼하게 검토하고, 현실적인 대안을 제시하는지 확인하십시오.",
    "당신은 이상주의적이고 정의로운 성격으로, 사회 문제에 관심이 많습니다. 챗봇의 답변이 사회적 약자를 배려하고, 불평등 해소에 기여하는지 평가하십시오. 챗봇이 윤리적인 문제를 제기하고, 사회적 책임감을 강조하는지 중요하게 생각합니다.",
    "당신은 내성적이고 조용한 성격으로, 혼자 있는 시간을 즐깁니다. 챗봇의 답변이 간결하고 명확하며, 불필요한 수식어를 사용하지 않는지 평가하십시오. 사용자가 원하는 정보만 정확하게 제공하고, 혼란을 야기하지 않는지 중요하게 생각합니다.",
    "당신은 리더십이 강하고 통솔력이 뛰어난 성격입니다. 챗봇의 답변이 명확한 지침을 제공하고, 사용자를 올바른 방향으로 이끄는지 평가하십시오. 챗봇이 문제 해결을 위한 주도적인 역할을 수행하고, 사용자에게 자신감을 심어주는지 중요하게 생각합니다.",
    "당신은 유머러스하고 재치 있는 성격으로, 사람들을 웃기는 것을 좋아합니다. 챗봇의 답변이 적절한 유머를 사용하여 분위기를 부드럽게 만들고, 사용자에게 즐거움을 제공하는지 평가하십시오. 챗봇이 상황에 맞는 유머를 구사하고, 불쾌감을 주지 않는지 확인해야 합니다.",
    "당신은 겸손하고 배려심이 깊은 성격으로, 타인을 존중하고 돕는 것을 좋아합니다. 챗봇의 답변이 정중하고 예의 바르며, 사용자를 존중하는 태도를 보이는지 평가하십시오. 챗봇이 사용자의 의견을 경청하고, 공감하는 모습을 보이는지 중요하게 생각합니다.",
    "당신은 독립적이고 자율적인 성격으로, 스스로 결정하고 행동하는 것을 선호합니다. 챗봇의 답변이 사용자의 자율성을 존중하고, 스스로 판단할 수 있도록 돕는지 평가하십시오. 챗봇이 일방적인 지시나 강요를 하지 않고, 다양한 선택지를 제시하는지 중요하게 생각합니다.",
    "당신은 완벽주의적인 성향이 강하며, 모든 것을 최고 수준으로 만들고자 합니다. 챗봇의 답변이 문법적으로 완벽하고, 오탈자가 없는지 꼼꼼하게 확인하십시오. 또한, 정보의 정확성과 최신성을 검증하고, 최고의 답변을 제공하는 데 집중하십시오.",
    "당신은 변화를 두려워하지 않고 새로운 시도를 즐기는 혁신가입니다. 챗봇의 답변이 기존의 방식을 벗어나 새로운 아이디어를 제시하고, 혁신적인 해결책을 제시하는지 평가하십시오. 챗봇이 미래 지향적인 비전을 제시하고, 새로운 가능성을 탐색하는 데 집중하십시오."
]
SYSTEM_INSTRUCTION_SUPERVISOR = """
당신은 AI 챗봇의 답변을 평가하는 전문 Supervisor입니다.
당신의 임무는 챗봇 사용자의 입력, 챗봇 AI의 이전 대화 히스토리, 챗봇 AI의 현재 system_instruction, 그리고 챗봇 AI가 생성한 답변을 종합적으로 검토하여, 해당 답변이 사용자의 의도와 챗봇의 지시에 얼마나 적절하고 유용하게 생성되었는지 0점부터 100점 사이의 점수로 평가하는 것입니다.

평가 기준:
1. 사용자 의도 부합성 (총점 30점):
1.1 질문의 핵심 파악 (0~5점): 사용자의 질문 또는 요청의 핵심 의도를 정확하게 파악했는가?
1.2 명확하고 직접적인 응답 (0~5점): 질문에 대한 답변이 모호하지 않고 명확하며, 직접적으로 관련되어 있는가?
1.3 정보의 완전성 (0~5점): 사용자가 필요로 하는 정보를 빠짐없이 제공하고 있는가?
1.4 목적 충족 (0~5점): 답변이 사용자의 정보 획득 목적 또는 문제 해결 목적을 충족시키는가?
1.5 추가적인 도움 제공 (0~5점): 필요한 경우, 추가적인 정보나 관련 자료를 제공하여 사용자의 이해를 돕는가?
1.6 적절한 용어 수준 (0~5점): 답변이 사용자의 수준에 맞추어 설명되어 있는가? 너무 높거나 너무 간단하지는 않은가?

2. 챗봇 시스템 지시 준수 (총점 30점):
2.1 페르소나 일관성 (0~5점): 챗봇이 system instruction에 명시된 페르소나를 일관되게 유지하고 있는가?
2.2 답변 스타일 준수 (0~5점): 답변의 어조, 표현 방식 등이 system instruction에 지정된 스타일을 따르고 있는가?
2.3 정보 포함/제외 규칙 준수 (0~5점): system instruction에 따라 특정 정보가 포함되거나 제외되었는가?
2.4 형식 준수 (0~5점): system instruction에 명시된 답변 형식 (예: 목록, 표 등)을 정확하게 따르고 있는가?
2.5 지시 이행 (0~5점): 시스템 지시 사항 (예: 특정 링크 제공, 특정 행동 유도)에 대한 이행 여부
2.6 문법 및 맞춤법 정확성 (0~5점): 문법 및 맞춤법 오류 없이 system instruction에 따라 작성되었는가?

3. 대화 흐름의 자연스러움 및 일관성 (총점 20점):
3.1 이전 대화 맥락 이해 (0~5점): 이전 대화 내용을 정확하게 이해하고, 현재 답변에 반영하고 있는가?
3.2 자연스러운 연결 (0~5점): 이전 대화와 현재 답변이 부자연스럽거나 갑작스럽지 않고 자연스럽게 이어지는가?
3.4 부적절한 내용 회피 (0~5점): 맥락에 맞지 않거나 부적절한 내용을 포함하지 않고 있는가?

4. 정보의 정확성 및 유용성 (총점 20점):
4.1 사실 기반 정보 (0~5점): 제공되는 정보가 사실에 근거하고 정확한가?
4.2 최신 정보 (0~5점): 제공되는 정보가 최신 정보를 반영하고 있는가?
4.3 정보의 신뢰성 (0~5점): 제공되는 정보의 출처가 신뢰할 만한가?
4.4 유용한 정보 (0~5점): 사용자가 실제로 활용할 수 있는 실질적인 정보를 제공하는가?

5. 감점 요소
5.1 Hallucination을 발견했을 경우, -40점
5.2 이전 답변 중 잊어버린 내용이 발견되었을 경우, -20점
5.3 Instruction 혹은 이전 답변에서 사용자가 원하는 문장 형식이나 양식이 있었음에도 따르지 않았을 경우, -10점

-----------------------------------------------------------------------------------

출력 형식:

오직 하나의 정수 값 (0-100)만 출력하세요. 다른 텍스트나 설명은 일절 포함하지 마십시오.
"""

# --- Helper Functions ---

# 이미지 크기 조절 함수
def resize_image_for_display(image_bytes: bytes, display_width: int) -> bytes:
    """
    이미지 바이트를 받아 지정된 너비에 맞춰 비율을 유지하며 조절하고 바이트로 반환합니다.
    (명령어 실행 환경: 가상환경 내에서 Streamlit 앱이 실행될 때)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        if width > display_width: # 지정된 너비보다 크면 조절
            ratio = display_width / width
            new_width = display_width
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS) # 고품질 리사이즈 알고리즘

        byte_arr = io.BytesIO()
        img.save(byte_arr, format=img.format if img.format else 'PNG') # 원본 포맷 유지, 없으면 PNG
        return byte_arr.getvalue()
    except Exception as e:
        st.warning(f"이미지 리사이즈 중 오류 발생: {e}. 원본 크기로 표시됩니다.")
        return image_bytes # 오류 발생 시 원본 반환

# --- Cloudinary Upload Helper Function ---
def upload_to_cloudinary(image_bytes: bytes) -> tuple[str, str] | None:
    """
    바이트 형태의 이미지를 Cloudinary에 업로드하고 (URL, Public ID) 튜플을 반환합니다.
    (명령어 실행 환경: 가상환경 내에서 Streamlit 앱이 실행될 때)
    """
    try:
        # Cloudinary에 업로드 시 public_id를 지정하여 추후 삭제를 용이하게 함
        public_id = f"faust_image_{uuid.uuid4()}" # 고유한 public_id 생성
        result = cloudinary.uploader.upload(
            file=io.BytesIO(image_bytes),
            public_id=public_id,
            resource_type="image"
        )
        if result and "secure_url" in result and "public_id" in result:
            return result["secure_url"], result["public_id"]
        else:
            st.error(f"Cloudinary 업로드 실패: 응답 형식이 올바르지 않습니다. {result}")
            return None
    except cloudinary.exceptions.Error as e:
        st.error(f"Cloudinary API 호출 중 오류 발생: {e}")
        return None
    except Exception as e:
        st.error(f"Cloudinary 업로드 중 예상치 못한 오류 발생: {e}")
        return None

# --- Cloudinary Delete Helper Function ---
def delete_from_cloudinary(public_id: str):
    """
    Cloudinary에서 지정된 public_id를 가진 이미지를 삭제합니다.
    (명령어 실행 환경: 가상환경 내에서 Streamlit 앱이 실행될 때)
    """
    if not is_cloudinary_configured:
        print("Cloudinary가 설정되지 않아 이미지 삭제를 건너뜁니다.")
        return

    try:
        # public_ids는 리스트 형태로 전달해야 함
        result = cloudinary.api.delete_resources([public_id], resource_type="image")

        # delete_resources의 반환값 구조를 고려하여 성공 여부 확인
        if result and public_id in result.get("deleted", []): # 수정된 부분: result.get("result") == "ok" 대신 public_id in result.get("deleted", []) 확인
            print(f"Cloudinary에서 이미지 '{public_id}' 삭제 성공.")
        else:
            print(f"Cloudinary에서 이미지 '{public_id}' 삭제 실패: {result.get('error', result)}") # 실패 시 더 자세한 에러 메시지 출력
    except cloudinary.exceptions.Error as e:
        print(f"Cloudinary 이미지 삭제 중 API 오류 발생: {e}")
    except Exception as e:
        print(f"Cloudinary 이미지 삭제 중 예상치 못한 오류 발생: {e}")

def convert_to_gemini_format_for_contents(chat_history_list):
    """
    Streamlit chat history (list of (role, text, optional_image_bytes_raw, optional_image_mime_type, optional_cloudinary_url, optional_cloudinary_public_id, optional_image_bytes_display_resized) tuples)를
    Gemini API의 `Content` 객체 리스트로 변환합니다.
    """
    gemini_contents = []
    for item in chat_history_list:
        role = item[0]
        text = item[1]
        image_bytes_raw = item[2] if len(item) > 2 else None # optional_image_bytes_raw
        image_mime_type = item[3] if len(item) > 3 else None # optional_image_mime_type

        parts = [types.Part(text=text)]

        # 이미지 데이터가 포함되어 있다면 Part에 추가 (Gemini에는 항상 원본 바이트 전달)
        if image_bytes_raw and image_mime_type:
            parts.insert(0, types.Part( # 이미지 파트를 먼저 넣는 것이 권장될 수 있음
                inline_data=types.Blob(
                    mime_type=image_mime_type,
                    data=base64.b64encode(image_bytes_raw).decode('utf-8')
                )
            ))
        gemini_contents.append(types.Content(parts=parts, role=role))
    return gemini_contents

def create_new_chat_session(model_name: str, current_history: list, system_instruction: str):
    """
    제공된 모델, 대화 이력, 시스템 명령어를 기반으로 새로운 genai.ChatSession을 생성합니다.
    시스템 명령어는 ChatSession의 config.system_instruction 매개변수로 주입됩니다.
    """
    # FausT의 제 1원칙을 system_instruction에 포함
    full_system_instruction = SUPER_INTRODUCTION_HEAD + system_instruction + SUPER_INTRODUCTION_TAIL

    # history를 Gemini Content 포맷으로 변환 (이제 이미지 데이터도 포함될 수 있음)
    initial_history_gemini_format = convert_to_gemini_format_for_contents(current_history)

    # config 객체에 system_instruction을 담아서 전달
    chat_config = types.GenerateContentConfig(
        system_instruction=full_system_instruction
    )

    return gemini_client.chats.create(
        model=model_name,
        history=initial_history_gemini_format,
        config=chat_config # config 매개변수로 전달
    )

def evaluate_response(user_input, chat_history, system_instruction, ai_response):
    """
    Supervisor 모델을 사용하여 AI 응답의 적절성을 평가합니다.
    이 함수는 Supervisor 모델에 대한 단일 턴 질의로, `client.models.generate_content`를 사용합니다.
    """
    # Supervisor의 시스템 명령어 (페르소나 + 평가 기준)
    supervisor_full_system_instruction = PERSONA_LIST[randint(0, len(PERSONA_LIST)-1)] + "\n" + SYSTEM_INSTRUCTION_SUPERVISOR

    # Supervisor에게 전달할 평가 대상 정보 (contents로 전달)
    # chat_history는 (role, text, ..., ...) 형식으로 변경되었으므로, 텍스트만 추출하여 평가 텍스트에 포함시켜야 함.
    chat_history_text_only = ""
    # chat_history는 list of tuples. 각 tuple의 두 번째 요소가 텍스트임.
    for item in chat_history:
        chat_history_text_only += f"\n{item[0]}: {item[1]}" # item[1]은 텍스트 부분

    evaluation_context_text = f"""
    ---
    사용자 입력: {user_input}
    ---
    챗봇 AI 이전 대화 히스토리:
    {chat_history_text_only}
    ---
    챗봇 AI 시스템 지시 (원래 지시): {system_instruction}
    ---
    챗봇 AI 답변: {ai_response}

    위 정보를 바탕으로, 챗봇 AI의 답변에 대해 0점부터 100점 사이의 점수를 평가하세요.
    """

    try:
        response = gemini_client.models.generate_content(
            model=st.session_state.selected_model,
            contents=[types.Part(text=evaluation_context_text)], # 평가할 정보는 contents로 전달
            config=types.GenerateContentConfig(
                system_instruction=supervisor_full_system_instruction, # Supervisor의 시스템 명령어는 config로 전달
                temperature=0.01,
                top_p=1.0,
                top_k=1,
            )
        )
        score_text = response.text.strip()
        score = int(score_text)
        if not (0 <= score <= 100):
            print(f"경고: Supervisor가 0-100 범위를 벗어난 점수를 반환했습니다: {score}")
            score = max(0, min(100, score))
        return score

    except ValueError as e:
        print(f"Supervisor 응답을 점수로 변환하는 데 실패했습니다: {score_text}, 오류: {e}")
        return 50
    except Exception as e:
        print(f"Supervisor 모델 호출 중 오류 발생: {e}")
        return 50


# --- Firebase User Data Management Functions ---
def load_user_data_from_firestore(user_id):
    """지정된 user_id로 Firestore에서 사용자 데이터를 로드합니다."""
    try:
        sessions_ref = db.collection("user_sessions").document(user_id)
        doc = sessions_ref.get()
        if doc.exists:
            data = doc.to_dict()
            st.session_state.saved_sessions = data.get("chat_data", {})
            for title, history_list in st.session_state.saved_sessions.items():
                # Firestore에서 불러온 데이터는 (role, text, cloudinary_url, cloudinary_public_id) 형태
                processed_history = []
                for item_dict in history_list:
                    role = item_dict["role"]
                    text = item_dict["text"]
                    # 로그인 사용자의 경우, 이미지 바이트는 Firestore에 저장되지 않았으므로 None
                    image_bytes_raw = None
                    image_mime_type = None
                    cloudinary_url = item_dict.get("cloudinary_url")
                    cloudinary_public_id = item_dict.get("cloudinary_public_id") # public_id 로드
                    image_bytes_display_resized = None # 로컬 표시용 바이트는 로드 시 필요 없으므로 None

                    # chat_history에 7개 요소 튜플로 추가
                    processed_history.append((role, text, image_bytes_raw, image_mime_type, cloudinary_url, cloudinary_public_id, image_bytes_display_resized))
                st.session_state.saved_sessions[title] = processed_history

            st.session_state.system_instructions = data.get("system_instructions", {})
            st.session_state.current_title = data.get("last_active_title", "새로운 대화")

            if st.session_state.current_title in st.session_state.saved_sessions:
                st.session_state.chat_history = st.session_state.saved_sessions[st.session_state.current_title]
            else:
                st.session_state.chat_history = []

            st.session_state.temp_system_instruction = st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction)
            current_instruction = st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction)

            # --- ChatSession 초기화 (로드된 데이터 기준) ---
            st.session_state.chat_session = create_new_chat_session(
                st.session_state.selected_model,
                st.session_state.chat_history,
                current_instruction
            )
            st.toast(f"Firestore에서 사용자 ID '{user_id}'의 데이터를 불러왔습니다.", icon="✅")
        else:
            # 데이터가 없는 경우 새로운 사용자 데이터 초기화
            st.session_state.saved_sessions = {}
            st.session_state.system_instructions = {}
            st.session_state.chat_history = []
            st.session_state.current_title = "새로운 대화"
            st.session_state.temp_system_instruction = default_system_instruction
            # --- 새로운 대화에 대한 ChatSession 초기화 ---
            st.session_state.chat_session = create_new_chat_session(
                st.session_state.selected_model,
                [],
                default_system_instruction
            )
            st.toast(f"Firestore에 사용자 ID '{user_id}'에 대한 데이터가 없습니다. 새로운 대화를 시작하세요.", icon="ℹ️")
    except Exception as e:
        error_message = f"Firestore에서 데이터 로드 중 오류 발생: {e}"
        print(error_message)
        st.error(error_message)
        # 오류 발생 시에도 기본 상태로 폴백하고 ChatSession은 기본값으로 초기화
        st.session_state.saved_sessions = {}
        st.session_state.system_instructions = {}
        st.session_state.chat_history = []
        st.session_state.current_title = "새로운 대화"
        st.session_state.temp_system_instruction = default_system_instruction # 오타 수정 (원래 코드에 있던 오타 `session_session`을 `session_state`로 수정함)
        st.session_state.chat_session = create_new_chat_session(
            st.session_state.selected_model,
            [],
            default_system_instruction
        )

def save_user_data_to_firestore(user_id):
    """현재 사용자 데이터를 Firestore에 저장합니다. 로그인된 사용자만 저장합니다."""
    # 비로그인(익명) 사용자일 경우 Firestore에 저장하지 않습니다.
    if not st.session_state.is_logged_in:
        print(f"익명 사용자 '{user_id}'의 데이터는 Firestore에 저장하지 않습니다.")
        return

    try:
        sessions_ref = db.collection("user_sessions").document(user_id)
        chat_data_to_save = {}
        for title, history_list in st.session_state.saved_sessions.items():
            # (role, text, image_bytes_raw, image_mime_type, cloudinary_url, cloudinary_public_id, image_bytes_display_resized) 튜플
            serialized_history = []
            for item in history_list:
                role = item[0]
                text = item[1]
                # image_bytes_raw = item[2] # Gemini API용 원본 바이트 (Firestore에 저장 안 함)
                # image_mime_type = item[3] # (Firestore에 저장 안 함)
                cloudinary_url = item[4] if len(item) > 4 else None # 로그인용 (저장)
                cloudinary_public_id = item[5] if len(item) > 5 else None # 로그인용 (저장)
                # image_bytes_display_resized = item[6] # 비로그인용 (저장 안 함)

                entry = {"role": role, "text": text}
                # 로그인 사용자의 경우, 이미지 바이트는 Firestore에 저장하지 않고 Cloudinary URL과 public_id만 저장
                if cloudinary_url is not None:
                    entry["cloudinary_url"] = cloudinary_url
                if cloudinary_public_id is not None:
                    entry["cloudinary_public_id"] = cloudinary_public_id

                serialized_history.append(entry)
            chat_data_to_save[title] = serialized_history

        data_to_save = {
            "chat_data": chat_data_to_save,
            "system_instructions": st.session_state.system_instructions,
            "last_active_title": st.session_state.current_title
        }
        sessions_ref.set(data_to_save)
        print(f"User data for ID '{user_id}' saved to Firestore.")
    except Exception as e:
        error_message = f"Error saving data to Firestore: {e}"
        print(error_message)
        st.error(error_message)

# --- App Logic Execution Flow ---
# 앱 시작 시 사용자 인증 상태 확인 및 데이터 로드
if not st.session_state.data_loaded:
    # st.user 객체는 OIDC 로그인 상태를 자동으로 반영합니다.
    if st.user.is_logged_in:
        # 로그인된 사용자 정보 (st.user는 dict-like 객체)
        user_email = st.user.get("email")
        if user_email: # 이메일 정보가 있다면
            st.session_state.user_id = user_email # 이메일을 user_id로 사용
            st.session_state.is_logged_in = True
            st.session_state.logged_in_user_email = user_email
            st.toast(f"'{user_email}'님으로 로그인되었습니다.", icon="🎉")
            print(f"Logged in user: {user_email}")
        else: # 로그인되었으나 이메일 정보가 없는 경우 (매우 드물지만, OIDC 설정에 따라 가능)
            st.session_state.is_logged_in = False # 익명으로 처리
            st.session_state.logged_in_user_email = None
            st.session_state.user_id = str(uuid.uuid4()) # 익명 ID로 폴백
            st.toast("Google 로그인에 성공했으나 이메일 정보를 가져올 수 없습니다. 익명으로 전환됩니다.", icon="⚠️")
            print("OAuth succeeded but email not found in st.user. Falling back to anonymous.")
    else: # 로그인되지 않은 상태
        st.session_state.is_logged_in = False
        st.session_state.logged_in_user_email = None
        # st.session_state.user_id는 이미 초기화 시 str(uuid.uuid4())로 설정되어 있습니다.
        st.toast("로그인하지 않은 상태입니다. 대화 이력은 이 기기에만 임시 저장됩니다.", icon="ℹ️")
        print("User is not logged in. Using anonymous ID.")

    load_user_data_from_firestore(st.session_state.user_id) # 결정된 user_id로 데이터 로드
    st.session_state.data_loaded = True

# ChatSession이 None일 경우 초기화 (앱 시작 시 또는 로그아웃 후)
if st.session_state.chat_session is None:
    current_instruction = st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction)
    st.session_state.chat_session = create_new_chat_session(
        st.session_state.selected_model,
        st.session_state.chat_history,
        current_instruction
    )

# --- Sidebar UI ---
with st.sidebar:
    st.image("assets/faust_icon.png", width=100) # 사이드바 로고 추가
    st.header("✨ FausT 채팅")

    # --- 계정 관리 섹션 ---
    st.markdown("---")
    st.subheader("👤 계정 관리")
    if st.session_state.is_logged_in: # 로그인된 상태
        st.success(f"로그인 됨: **{st.session_state.logged_in_user_email}**")
        st.markdown(f"사용자 ID: `{st.session_state.user_id}`")
        st.button("로그아웃", on_click=st.logout, use_container_width=True, disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending)
    else: # 로그인되지 않은 상태 (익명)
        st.info("로그인하지 않은 상태입니다. 현재 대화는 이 기기에만 임시 저장됩니다.")
        st.markdown(f"익명 ID: `{st.session_state.user_id}`") # 익명 ID 표시

        st.markdown("---")
        st.markdown("**Google 계정으로 로그인**")
        st.write("아래 버튼을 클릭하여 Google 계정으로 로그인하세요.")
        st.button("Google로 로그인", on_click=st.login, args=["google"], use_container_width=True, disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending)
        st.write("---") # UI 구분선 추가
        st.write("로그인 없이 계속하기")
        st.write("익명 모드로 채팅을 시작합니다. 대화 이력은 저장되지 않습니다.")


    st.markdown("---")

    if st.button("➕ 새로운 대화", use_container_width=True,
                             disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending):
        # 현재 대화 상태를 저장 (로그인된 사용자만)
        if st.session_state.is_logged_in and st.session_state.current_title != "새로운 대화" and st.session_state.chat_history:
            st.session_state.saved_sessions[st.session_state.current_title] = st.session_state.chat_history.copy()
            current_instruction_to_save = st.session_state.temp_system_instruction if st.session_state.temp_system_instruction is not None else st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction)
            st.session_state.system_instructions[st.session_state.current_title] = current_instruction_to_save
            save_user_data_to_firestore(st.session_state.user_id) # 로그인된 사용자만 저장

        # 새로운 대화 상태로 초기화
        st.session_state.chat_session = None
        st.session_state.chat_history = []
        st.session_state.current_title = "새로운 대화"
        st.session_state.temp_system_instruction = default_system_instruction
        st.session_state.editing_instruction = False
        st.session_state.saved_sessions["새로운 대화"] = [] # 빈 목록으로 저장되도록 보장 (Firestore에 저장되진 않음)
        st.session_state.system_instructions["새로운 대화"] = default_system_instruction

        # --- 새로운 ChatSession 초기화 ---
        st.session_state.chat_session = create_new_chat_session(
            st.session_state.selected_model,
            [],
            default_system_instruction
        )
        # 로그인된 사용자만 저장 (새로운 대화 시작 시점)
        if st.session_state.is_logged_in:
            save_user_data_to_firestore(st.session_state.user_id)
        st.rerun()

    if st.session_state.saved_sessions:
        st.subheader("📁 저장된 대화")
        sorted_keys = sorted(st.session_state.saved_sessions.keys(),
                                 key=lambda x: st.session_state.saved_sessions[x][-1][1] if st.session_state.saved_sessions[x] else "",
                                 reverse=True)
        for key in sorted_keys:
            if key == "새로운 대화" and not st.session_state.saved_sessions[key]:
                continue
            display_key = key if len(key) <= 30 else key[:30] + "..."
            if st.button(f"💬 {display_key}", use_container_width=True, key=f"load_session_{key}",
                                 disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending):
                # 현재 대화 상태를 저장 (로그인된 사용자만)
                if st.session_state.is_logged_in and st.session_state.current_title != "새로운 대화" and st.session_state.chat_history:
                    st.session_state.saved_sessions[st.session_state.current_title] = st.session_state.chat_history.copy()
                    current_instruction_to_save = st.session_state.temp_system_instruction if st.session_state.temp_system_instruction is not None else st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction)
                    st.session_state.system_instructions[st.session_state.current_title] = current_instruction_to_save
                    save_user_data_to_firestore(st.session_state.user_id) # 로그인된 사용자만 저장

                st.session_state.chat_history = st.session_state.saved_sessions[key]
                st.session_state.current_title = key
                st.session_state.new_title = key
                st.session_state.temp_system_instruction = st.session_state.system_instructions.get(key, default_system_instruction)

                # --- 로드된 대화 이력으로 ChatSession 초기화 ---
                current_instruction = st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction)
                st.session_state.chat_session = create_new_chat_session(
                    st.session_state.selected_model,
                    st.session_state.chat_history,
                    current_instruction
                )
                st.session_state.editing_instruction = False
                st.session_state.editing_title = False
                # 로그인된 사용자만 저장 (대화 로드 시점)
                if st.session_state.is_logged_in:
                    save_user_data_to_firestore(st.session_state.user_id)
                st.rerun()

    with st.expander("⚙️ 설정"):
        st.write("---")
        st.write("모델 선택")
        selected_model_option = st.selectbox(
            "사용할 AI 모델을 선택하세요:",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.selected_model),
            key="model_selector",
            disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending
        )
        if selected_model_option != st.session_state.selected_model:
            st.session_state.selected_model = selected_model_option
            # --- 모델 변경 시 ChatSession 다시 초기화 ---
            current_instruction = st.session_state.system_instructions.get(
                st.session_state.current_title, default_system_instruction
            )
            st.session_state.chat_session = create_new_chat_session(
                st.session_state.selected_model,
                st.session_state.chat_history,
                current_instruction
            )
            st.toast(f"AI 모델이 '{st.session_state.selected_model}'으로 변경되었습니다.", icon="🤖")
            st.rerun()

        st.write("---")
        st.write("Supervision 관련 설정을 변경할 수 있습니다.")
        st.session_state.use_supervision = st.toggle(
            "Supervision 사용",
            value=st.session_state.use_supervision,
            help="AI 답변의 적절성을 평가하고 필요시 재시도하는 기능을 사용합니다. (기본: 비활성화)",
            key="supervision_toggle",
            disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending
        )
        st.session_state.supervision_max_retries = st.slider(
            "최대 재시도 횟수",
            min_value=1,
            max_value=5,
            value=st.session_state.supervision_max_retries,
            disabled=st.session_state.is_generating or not st.session_state.use_supervision or st.session_state.delete_confirmation_pending,
            key="supervision_max_retries_slider"
        )
        st.session_state.supervisor_count = st.slider(
            "Supervisor 개수",
            min_value=1,
            max_value=5,
            value=st.session_state.supervisor_count,
            disabled=st.session_state.is_generating or not st.session_state.use_supervision or st.session_state.delete_confirmation_pending,
            key="supervisor_count_slider"
        )
        st.session_state.supervision_threshold = st.slider(
            "Supervision 통과 점수 (평균)",
            min_value=0,
            max_value=100,
            value=st.session_state.supervision_threshold,
            step=5,
            disabled=st.session_state.is_generating or not st.session_state.use_supervision or st.session_state.delete_confirmation_pending,
            key="supervision_threshold_slider"
        )
        if not st.session_state.use_supervision:
            st.info("Supervision 기능이 비활성화되어 있습니다. AI 답변은 바로 표시됩니다.")


# --- Main Content Area ---
col1, col2, col3 = st.columns([0.9, 0.05, 0.05])
with col1:
    if not st.session_state.editing_title:
        st.subheader(f"💬 {st.session_state.current_title}")
    else:
        st.text_input("새로운 제목", key="new_title_input", value=st.session_state.new_title, label_visibility="collapsed",
                              disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending)
with col2:
    if not st.session_state.editing_title:
        if st.button("✏️", key="edit_title_button", help="대화 제목 수정",
                             disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending):
            st.session_state.editing_title = True
            st.session_state.new_title = st.session_state.current_title
            st.rerun()
    else:
        if st.button("✅", key="save_title_button", help="새로운 제목 저장",
                             disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending):
            new_title = st.session_state.new_title_input
            if new_title and new_title != st.session_state.current_title:
                if st.session_state.current_title in st.session_state.saved_sessions:
                    st.session_state.saved_sessions[new_title] = st.session_state.saved_sessions.pop(st.session_state.current_title)
                    st.session_state.system_instructions[new_title] = st.session_state.system_instructions.pop(st.session_state.current_title)
                    st.session_state.current_title = new_title
                    # 로그인된 사용자만 저장
                    if st.session_state.is_logged_in:
                        save_user_data_to_firestore(st.session_state.user_id)
                    st.toast(f"대화 제목이 '{st.session_state.current_title}'로 변경되었습니다.", icon="📝")
                else:
                    st.warning("이전 대화 제목을 찾을 수 없습니다. 저장 후 다시 시도해주세요.")
            st.session_state.editing_title = False
            st.rerun()
        if st.button("❌", key="cancel_title_button", help="제목 수정 취소",
                             disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending):
            st.session_state.editing_instruction = False
            st.rerun()

with col3:
    is_delete_disabled = st.session_state.is_generating or \
                             (st.session_state.current_title == "새로운 대화" and not st.session_state.chat_history) or \
                             st.session_state.delete_confirmation_pending

    if st.button("🗑️", key="delete_chat_button", help="현재 대화 삭제", disabled=is_delete_disabled):
        st.session_state.delete_confirmation_pending = True
        st.session_state.title_to_delete = st.session_state.current_title
        st.rerun()

# --- Delete Confirmation Pop-up (Streamlit style) ---
if st.session_state.delete_confirmation_pending:
    st.warning(f"'{st.session_state.title_to_delete}' 대화를 정말 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.", icon="⚠️")
    confirm_col1, confirm_col2 = st.columns(2)
    with confirm_col1:
        if st.button("예, 삭제합니다", key="confirm_delete_yes", use_container_width=True):
            deleted_title = st.session_state.title_to_delete
            if deleted_title in st.session_state.saved_sessions:
                # 삭제 대상 대화에서 Cloudinary public_id가 있는 이미지들을 찾아 삭제
                if st.session_state.is_logged_in and is_cloudinary_configured:
                    for item in st.session_state.saved_sessions[deleted_title]:
                        # item[5]는 cloudinary_public_id
                        if len(item) > 5 and item[5] is not None:
                            delete_from_cloudinary(item[5]) # Cloudinary에서 이미지 삭제 호출
                            print(f"Cloudinary 이미지 {item[5]} 삭제 시도 중...")

                # Firestore에서 대화 삭제 (save_user_data_to_firestore가 담당)
                del st.session_state.saved_sessions[deleted_title]
                del st.session_state.system_instructions[deleted_title]

                st.session_state.current_title = "새로운 대화"
                st.session_state.chat_history = []
                st.session_state.temp_system_instruction = default_system_instruction
                st.session_state.chat_session = create_new_chat_session(
                    st.session_state.selected_model,
                    [],
                    default_system_instruction
                )
                st.toast(f"'{deleted_title}' 대화가 삭제되었습니다.", icon="🗑️")
                if "새로운 대화" not in st.session_state.saved_sessions:
                    st.session_state.saved_sessions["새로운 대화"] = []
                    st.session_state.system_instructions["새로운 대화"] = default_system_instruction
                # 로그인된 사용자만 저장 (삭제 반영)
                if st.session_state.is_logged_in:
                    save_user_data_to_firestore(st.session_state.user_id)
            elif deleted_title == "새로운 대화": # "새로운 대화"는 저장된 세션에 없을 수 있음
                st.session_state.chat_history = []
                st.session_state.temp_system_instruction = default_system_instruction
                st.session_state.chat_session = create_new_chat_session(
                    st.session_state.selected_model,
                    [],
                    default_system_instruction
                )
                st.toast("현재 대화가 초기화되었습니다.", icon="🗑️")
                st.session_state.saved_sessions["새로운 대화"] = [] # 빈 목록으로 저장되도록 보장
                st.session_state.system_instructions["새로운 대화"] = default_system_instruction
                # 로그인된 사용자만 저장
                if st.session_state.is_logged_in:
                    save_user_data_to_firestore(st.session_state.user_id)
            else:
                st.warning(f"'{deleted_title}' 대화를 찾을 수 없습니다. 이미 삭제되었거나 저장되지 않았습니다.")

            st.session_state.delete_confirmation_pending = False
            st.session_state.title_to_delete = None
            st.rerun()
    with confirm_col2:
        if st.button("아니요, 취소합니다", key="confirm_delete_no", use_container_width=True):
            st.session_state.delete_confirmation_pending = False
            st.session_state.title_to_delete = None
            st.rerun()

# AI 설정 버튼 및 영역
if st.button("⚙️ AI 설정하기", help="시스템 명령어를 설정할 수 있어요",
             disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending):
    st.session_state.editing_instruction = not st.session_state.editing_instruction

if st.session_state.editing_instruction:
    with st.expander("🧠 시스템 명령어 설정", expanded=True):
        st.session_state.temp_system_instruction = st.text_area(
            "System instruction 입력",
            value=st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction),
            height=200,
            key="system_instruction_editor",
            disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending
        )
        _, col1_ai, col2_ai = st.columns([0.9, 0.3, 0.3])
        with col1_ai:
            if st.button("✅ 저장", use_container_width=True, key="save_instruction_button",
                                 disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending):
                st.session_state.system_instructions[st.session_state.current_title] = st.session_state.temp_system_instruction
                st.session_state.saved_sessions[st.session_state.current_title] = st.session_state.chat_history.copy()

                # --- 시스템 명령어 변경 시 ChatSession 다시 초기화 ---
                current_instruction = st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction)
                st.session_state.chat_session = create_new_chat_session(
                    st.session_state.selected_model,
                    st.session_state.chat_history,
                    current_instruction
                )

                # 로그인된 사용자만 저장
                if st.session_state.is_logged_in:
                    save_user_data_to_firestore(st.session_state.user_id)
                st.success("AI 설정이 저장되었습니다.")
                st.session_state.editing_instruction = False
                st.rerun()
        with col2_ai:
            if st.button("❌ 취소", use_container_width=True, key="cancel_instruction_button",
                                 disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending):
                st.session_state.editing_instruction = False
                st.rerun()

# --- Chat Display Area ---
chat_display_container = st.container()

# --- Final Chat History Display (Always Rendered) ---
with chat_display_container:
    for i, item in enumerate(st.session_state.chat_history):
        role, message = item[0], item[1]
        image_bytes_raw = item[2] if len(item) > 2 else None
        image_mime_type = item[3] if len(item) > 3 else None
        cloudinary_url_raw = item[4] if len(item) > 4 else None # Cloudinary 원본 URL (사용되지 않음)
        cloudinary_public_id = item[5] if len(item) > 5 else None # Cloudinary public_id (URL 생성 및 삭제에 사용)
        image_bytes_display_resized = item[6] if len(item) > 6 else None # 비로그인 사용자용 리사이즈된 바이트

        with st.chat_message("ai" if role == "model" else "user"):
            if cloudinary_public_id: # Cloudinary public_id가 있으면 (로그인 사용자)
                # Cloudinary Transformation을 URL에 적용하여 이미지 크기 제어
                # c_limit: 지정된 크기 내에서 이미지 비율 유지하며 조절
                # w: width. Streamlit이 자체적으로 폭을 조절하는 대신 고정 너비로 제공
                # h: height. 필요시 추가 가능 (crop="limit"과 함께 사용)
                transformed_cloudinary_url = cloudinary.utils.cloudinary_url(
                    cloudinary_public_id, # 'source' 인자로 public_id를 전달
                    width=LOCAL_DISPLAY_WIDTH, # LOCAL_DISPLAY_WIDTH와 동일한 너비로 Cloudinary에서 변환
                    crop="limit", # 'limit' 모드로 지정된 폭을 넘지 않도록 비율 유지
                    secure=True # HTTPS 사용
                )[0] # cloudinary_url 함수는 튜플을 반환하므로 첫 번째 요소 (URL)만 가져옴

                st.markdown(f"![업로드된 이미지]({transformed_cloudinary_url})")
            elif image_bytes_display_resized and image_mime_type: # Cloudinary URL이 없고 리사이즈된 바이트 데이터가 있으면 (비로그인 사용자)
                st.image(image_bytes_display_resized, caption="업로드된 이미지", use_container_width=False) # 이미 리사이즈된 이미지이므로 width 지정 불필요

            st.markdown(message) # 텍스트 메시지 표시 (이미지 아래에)
            if role == "model" and i == len(st.session_state.chat_history) - 1 and not st.session_state.is_generating \
                and not st.session_state.delete_confirmation_pending:
                if st.button("🔄 다시 생성", key=f"regenerate_button_final_{i}", use_container_width=True):
                    st.session_state.regenerate_requested = True
                    st.session_state.is_generating = True
                    st.session_state.chat_history.pop()
                    st.rerun()

# --- Input Area ---
col_prompt_input, col_upload_icon = st.columns([0.85, 0.15])

with col_prompt_input:
    user_prompt = st.chat_input("메시지를 입력하세요.", key="user_prompt_input",
                                 disabled=st.session_state.is_generating or st.session_state.delete_confirmation_pending)

with col_upload_icon:
    # Uploader 비활성화 조건: 생성 중, 삭제 확인 중
    uploader_disabled = st.session_state.is_generating or st.session_state.delete_confirmation_pending
    st.file_uploader("🖼️ / 📄", type=["png", "jpg", "jpeg", "pdf"], key="file_uploader_main", label_visibility="collapsed",
                                                 disabled=uploader_disabled, help="이미지 또는 PDF 파일을 업로드하세요.")

if st.session_state.file_uploader_main: # file_uploader의 key 값을 직접 사용
    st.session_state.uploaded_file = st.session_state.file_uploader_main
    st.caption("파일 업로드 완료")
else:
    if st.session_state.uploaded_file is not None:
        st.session_state.uploaded_file = None

# AI 생성 트리거 로직
# (명령어 실행 환경: 가상환경 내에서 Streamlit 앱이 실행될 때)
if user_prompt is not None and not st.session_state.is_generating:
    if user_prompt != "" or st.session_state.uploaded_file is not None:
        user_input_gemini_parts = []

        # chat_history에 저장될 이미지 데이터, 타입, URL 변수 초기화
        image_bytes_for_chat_history_raw = None # 원본 바이트 (Gemini API용)
        image_bytes_for_chat_history_display = None # 비로그인 사용자 UI 표시용 (리사이즈된 바이트)
        image_mime_type_for_chat_history = None
        cloudinary_url_for_chat_history = None # 로그인 사용자 전용
        cloudinary_public_id_for_chat_history = None # 로그인 사용자 전용

        # UI에 표시될 사용자 메시지 (텍스트 부분만)
        user_prompt_for_display = user_prompt if user_prompt is not None else ""

        # 이미지/PDF 파일 처리
        if st.session_state.uploaded_file:
            file_type = st.session_state.uploaded_file.type
            file_data = st.session_state.uploaded_file.getvalue()

            # Gemini에 전달할 원본 바이트는 항상 저장
            image_bytes_for_chat_history_raw = file_data
            image_mime_type_for_chat_history = file_type

            # --- 이미지 파일 (png, jpg, jpeg) 처리 ---
            if file_type.startswith("image/"):
                if st.session_state.is_logged_in and is_cloudinary_configured: # 로그인 & Cloudinary 설정 완료
                    upload_result = upload_to_cloudinary(file_data) # 원본 파일 업로드
                    if upload_result:
                        cloudinary_url_for_chat_history, cloudinary_public_id_for_chat_history = upload_result
                    else:
                        st.warning("로그인 상태이지만 Cloudinary 업로드에 실패했습니다. 이미지는 현재 세션에만 임시 저장됩니다.")
                        # Cloudinary 업로드 실패 시 세션에 임시 저장 및 표시
                        image_bytes_for_chat_history_display = resize_image_for_display(file_data, LOCAL_DISPLAY_WIDTH)
                else: # 비로그인 사용자 또는 Cloudinary 설정 안 됨
                    # 세션에 임시 저장 및 표시 (리사이즈하여 저장)
                    image_bytes_for_chat_history_display = resize_image_for_display(file_data, LOCAL_DISPLAY_WIDTH)

            # --- PDF 파일 처리 ---
            elif file_type == "application/pdf":
                try:
                    pdf_document = fitz.open(stream=file_data, filetype="pdf")
                    processed_page_count = 0

                    first_page_image_bytes_raw = None # PDF 원본 첫 페이지 바이트 (Gemini용)
                    first_page_image_mime_type = None

                    for page_num in range(min(len(pdf_document), MAX_PDF_PAGES_TO_PROCESS)):
                        page = pdf_document.load_page(page_num)
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        img_bytes = pix.tobytes(format="png")

                        # Gemini API에 전달할 Part (원본 이미지 데이터)
                        user_input_gemini_parts.append(types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png", # PDF 페이지는 PNG로 변환됨
                                data=base64.b64encode(img_bytes).decode('utf-8')
                            )
                        ))

                        if page_num == 0: # 첫 페이지만 chat_history에 저장할 이미지로 지정
                            first_page_image_bytes_raw = img_bytes
                            first_page_image_mime_type = "image/png"

                        processed_page_count += 1

                    if len(pdf_document) > MAX_PDF_PAGES_TO_PROCESS:
                        st.warning(f"PDF 파일이 {MAX_PDF_PAGES_TO_PROCESS} 페이지를 초과하여 처음 {MAX_PDF_PAGES_TO_PROCESS} 페이지만 처리되었습니다.")

                    pdf_document.close()

                    if first_page_image_bytes_raw: # PDF에서 첫 페이지 이미지가 추출된 경우
                        # Gemini에 전달할 원본 바이트는 항상 저장 (image_bytes_for_chat_history_raw에 저장)
                        image_bytes_for_chat_history_raw = first_page_image_bytes_raw
                        image_mime_type_for_chat_history = first_page_image_mime_type

                        if st.session_state.is_logged_in and is_cloudinary_configured: # 로그인 & Cloudinary 설정 완료
                            upload_result = upload_to_cloudinary(first_page_image_bytes_raw) # 원본 파일 업로드
                            if upload_result:
                                cloudinary_url_for_chat_history, cloudinary_public_id_for_chat_history = upload_result
                            else:
                                st.warning("로그인 상태이지만 Cloudinary 업로드에 실패하여 PDF 이미지는 현재 세션에만 임시 저장됩니다.")
                                # Cloudinary 업로드 실패 시 세션에 임시 저장 및 표시
                                image_bytes_for_chat_history_display = resize_image_for_display(first_page_image_bytes_raw, LOCAL_DISPLAY_WIDTH)
                        else: # 비로그인 사용자 또는 Cloudinary 설정 안 됨
                            # 세션에 임시 저장 및 표시 (리사이즈하여 저장)
                            image_bytes_for_chat_history_display = resize_image_for_display(first_page_image_bytes_raw, LOCAL_DISPLAY_WIDTH)
                    else:
                        st.warning("PDF에서 유효한 이미지를 추출할 수 없습니다. Gemini에 PDF 내용이 전달되지 않습니다.")

                except Exception as e:
                    st.error(f"PDF 파일 처리 중 오류 발생: {e}. PDF 내용을 포함하지 않고 대화를 계속합니다.")
            else:
                st.warning(f"지원되지 않는 파일 형식입니다: {file_type}. 파일 내용을 포함하지 않고 대화를 계속합니다.")

        # 사용자 입력 텍스트 (옵션)
        if user_prompt is not None and user_prompt != "":
            # user_input_gemini_parts에는 텍스트 Part를 추가
            user_input_gemini_parts.append(types.Part(text=user_prompt))

        # 최종적으로 Gemini API에 보낼 parts가 아무것도 없는 경우 (파일 업로드 실패 또는 프롬프트 없음)
        if not user_input_gemini_parts:
            st.warning("제공된 유효한 입력(텍스트 또는 이미지)이 없어 AI에 전달되지 않았습니다. 다시 시도해주세요.")
            st.session_state.is_generating = False
            st.session_state.uploaded_file = None
            st.rerun()

        # chat_history에 사용자 메시지 추가
        # (role, text_content, image_bytes_raw, image_mime_type, cloudinary_url, cloudinary_public_id, image_bytes_display_resized) 튜플 형태로 확장
        st.session_state.chat_history.append(
            ("user", user_prompt_for_display.strip(),
             image_bytes_for_chat_history_raw, # Gemini에 전달될 원본 바이트 (채팅 기록에도 저장)
             image_mime_type_for_chat_history,
             cloudinary_url_for_chat_history, # 로그인 및 Cloudinary 성공 시 URL
             cloudinary_public_id_for_chat_history, # 로그인 및 Cloudinary 성공 시 public_id
             image_bytes_for_chat_history_display) # 비로그인 또는 Cloudinary 실패 시 UI 표시용 리사이즈된 바이트
        )

        st.session_state.is_generating = True
        st.session_state.last_user_input_gemini_parts = user_input_gemini_parts
        st.rerun()

# --- AI Response Generation and Display Logic (Normal & Regeneration) ---
if st.session_state.is_generating:
    with chat_display_container:
        with st.chat_message("ai"):
            message_placeholder = st.empty()

            best_ai_response = ""
            highest_score = -1

            initial_user_contents = st.session_state.last_user_input_gemini_parts
            current_instruction = st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction)

            if st.session_state.use_supervision:
                attempt_count = 0
                while attempt_count < st.session_state.supervision_max_retries:
                    attempt_count += 1
                    message_placeholder.markdown(f"🤖 답변 생성 중... (시도: {attempt_count}/{st.session_state.supervision_max_retries})")
                    full_response = ""

                    try:
                        # ChatSession을 매 시도마다 재초기화
                        st.session_state.chat_session = create_new_chat_session(
                            st.session_state.selected_model,
                            st.session_state.chat_history, # 현재까지의 대화 이력 전달
                            current_instruction # 시스템 명령어 전달 (이제 config에 포함되어 전달됨)
                        )

                        response_stream = st.session_state.chat_session.send_message_stream(initial_user_contents)

                        for chunk in response_stream:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                        # --- Supervisor 평가 시작 ---
                        total_score = 0
                        supervisor_feedback_list = []

                        user_text_for_eval = ""
                        for part in initial_user_contents:
                            if isinstance(part, types.Part) and part.text:
                                user_text_for_eval = part.text
                                break

                        for i in range(st.session_state.supervisor_count):
                            # Supervisor 평가 시에는 이미지 데이터 없이 텍스트만 전달
                            # chat_history가 (role, text, raw_bytes, mime_type, cloudinary_url, public_id, resized_bytes) 튜플이므로,
                            # 텍스트만 추출해서 전달해야 함.
                            history_for_supervisor_text_only = []
                            # 마지막 사용자 메시지를 제외하기 위해 chat_history[:-1] 사용
                            # 주의: chat_history의 각 item이 튜플이므로, item[1] (텍스트 부분)만 추출
                            for hist_item in st.session_state.chat_history[:-1]:
                                history_for_supervisor_text_only.append((hist_item[0], hist_item[1]))

                            score = evaluate_response(
                                user_input=user_text_for_eval,
                                chat_history=history_for_supervisor_text_only, # 텍스트만 추출된 히스토리 전달
                                system_instruction=current_instruction,
                                ai_response=full_response
                            )
                            total_score += score
                            supervisor_feedback_list.append(f"Supervisor {i+1} 점수: {score}점")

                        avg_score = total_score / st.session_state.supervisor_count

                        st.info(f"평균 Supervisor 점수: {avg_score:.2f}점")
                        for feedback in supervisor_feedback_list:
                            st.info(feedback)

                        if avg_score >= st.session_state.supervision_threshold:
                            best_ai_response = full_response
                            highest_score = avg_score
                            st.success("✅ 답변이 Supervision 통과 기준을 만족합니다!")
                            break
                        else:
                            st.warning(f"❌ 답변이 Supervision 통과 기준({st.session_state.supervision_threshold}점)을 만족하지 못했습니다. 재시도합니다...")
                            if avg_score > highest_score:
                                highest_score = avg_score
                                best_ai_response = full_response

                    except Exception as e:
                        st.error(f"메시지 생성 또는 평가 중 오류 발생: {e}")
                        message_placeholder.markdown("죄송합니다. 메시지를 처리하는 중 오류가 발생했습니다.")
                        st.session_state.uploaded_file = None
                        break
            else:
                message_placeholder.markdown("🤖 답변 생성 중...")
                full_response = ""
                try:
                    # ChatSession을 매 시도마다 재초기화
                    st.session_state.chat_session = create_new_chat_session(
                        st.session_state.selected_model,
                        st.session_state.chat_history, # 현재까지의 대화 이력 전달
                        current_instruction # 시스템 명령어 전달 (이제 config에 포함되어 전달됨)
                    )

                    response_stream = st.session_state.chat_session.send_message_stream(initial_user_contents)

                    for chunk in response_stream:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    best_ai_response = full_response
                    highest_score = 100
                except Exception as e:
                    st.error(f"메시지 생성 중 오류 발생: {e}")
                    message_placeholder.markdown("죄송합니다. 메시지를 처리하는 중 오류가 발생했습니다.")
                    st.session_state.uploaded_file = None

            if best_ai_response:
                # AI 응답에는 이미지가 없으므로 (role, text, None, None, None, None, None) 튜플로 저장
                st.session_state.chat_history.append(("model", best_ai_response, None, None, None, None, None))
                message_placeholder.markdown(best_ai_response)
                if st.session_state.use_supervision:
                    st.toast(f"대화가 성공적으로 완료되었습니다. 최종 점수: {highest_score:.2f}점", icon="👍")
                else:
                    st.toast("대화가 성공적으로 완료되었습니다.", icon="👍")
            else:
                st.error("모든 재시도 후에도 만족스러운 답변을 얻지 못했습니다. 이전 최고 점수 답변을 표시합니다.")
                if highest_score != -1:
                    # AI 응답에는 이미지가 없으므로 (role, text, None, None, None, None, None) 튜플로 저장
                    st.session_state.chat_history.append(("model", best_ai_response, None, None, None, None, None))
                    message_placeholder.markdown(best_ai_response)
                    if st.session_state.use_supervision:
                        st.toast(f"최고 점수 답변이 표시되었습니다. 점수: {highest_score:.2f}점", icon="❗")
                    else:
                        st.toast("최고 점수 답변이 표시되었습니다.", icon="❗")
                else:
                    # AI 응답에는 이미지가 없으므로 (role, text, None, None, None, None, None) 튜플로 저장
                    st.session_state.chat_history.append(("model", "죄송합니다. 현재 요청에 대해 답변을 생성할 수 없습니다.", None, None, None, None, None))
                    message_placeholder.markdown("죄송합니다. 현재 요청에 대해 답변을 생성할 수 없습니다.")

            st.session_state.uploaded_file = None
            st.session_state.is_generating = False

            if st.session_state.current_title == "새로운 대화" and \
               len(st.session_state.chat_history) >= 2 and \
               st.session_state.chat_history[-2][0] == "user" and st.session_state.chat_history[-1][0] == "model":
                with st.spinner("대화 제목 생성 중..."):
                    try:
                        # 사용자 메시지에서 텍스트 부분만 추출하여 제목 생성 프롬프트에 사용
                        # chat_history[-2]는 (role, text, ...) 튜플이므로 text만 가져옴
                        summary_prompt_text = st.session_state.chat_history[-2][1]
                        summary_response = gemini_client.models.generate_content(
                            model=st.session_state.selected_model,
                            contents=[types.Part(text=f"다음 사용자의 메시지를 요약해서 대화 제목으로 만들어줘 (한 문장, 30자 이내):\n\n{summary_prompt_text}")]
                        )
                        original_title = summary_response.text.strip().replace("\n", " ").replace('"', '')
                        if not original_title or len(original_title) > 30:
                            original_title = "새로운 대화"
                    except Exception as e:
                        print(f"제목 생성 오류: {e}. 기본 제목 사용.")
                        original_title = "새로운 대화"

                    title_key = original_title
                    count = 1
                    while title_key in st.session_state.saved_sessions:
                        title_key = f"{original_title} ({count})"
                        count += 1
                    st.session_state.current_title = title_key
                    st.toast(f"대화 제목이 '{title_key}'로 설정되었습니다.", icon="📝")

            # 로그인된 사용자만 저장
            if st.session_state.is_logged_in:
                # chat_history는 이제 이미지 데이터도 포함 (role, text, image_bytes_raw, image_mime_type, cloudinary_url, cloudinary_public_id, image_bytes_display_resized)
                st.session_state.saved_sessions[st.session_state.current_title] = st.session_state.chat_history.copy()
                current_instruction_for_save = st.session_state.temp_system_instruction if st.session_state.temp_system_instruction is not None else st.session_state.system_instructions.get(st.session_state.current_title, default_system_instruction)
                st.session_state.system_instructions[st.session_state.current_title] = current_instruction_for_save
                save_user_data_to_firestore(st.session_state.user_id)

            st.rerun()