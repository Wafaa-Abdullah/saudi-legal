"""
Saudi Legal AI API — v3.0
FastAPI + RAG + Multi-model fallback
"""
import os, gc, re, time, logging
from collections import deque, defaultdict
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from groq import Groq
from openai import OpenAI
from huggingface_hub import InferenceClient, login
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
import chromadb

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# Config من Environment Variables
# ══════════════════════════════════════════════════════════
GROQ_API_KEY       = os.getenv('GROQ_API_KEY', '')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
HF_TOKEN           = os.getenv('HF_TOKEN', '')
API_SECRET_KEY     = os.getenv('API_SECRET_KEY', 'saudi-legal-2024')
HF_REPO_ID         = os.getenv('HF_REPO_ID', 'WafaaFraih/saudi-legal-moj')
CHROMA_PATH        = os.getenv('CHROMA_PATH', './chroma_db')

# ══════════════════════════════════════════════════════════
# Global State
# ══════════════════════════════════════════════════════════
vectorstore    = None
bm25_index     = None
bm25_texts     = []
bm25_metadatas = []
embeddings     = None
ACTIVE_MODELS  = []
hf_client      = None
working_or_models = []
groq_client    = None
or_client      = None
request_log    = deque(maxlen=500)
stats          = {'total': 0, 'success': 0, 'blocked': 0, 'errors': 0}
active_ips     = {}
_expansion_cache = {}
_rewrite_cache   = {}

# ══════════════════════════════════════════════════════════
# Startup: تحميل كل حاجة
# ══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    logger.info("Shutting down...")

async def startup():
    global vectorstore, bm25_index, bm25_texts, bm25_metadatas
    global embeddings, ACTIVE_MODELS, hf_client, working_or_models
    global groq_client, or_client

    logger.info("🚀 Starting Saudi Legal AI...")

    # ── Clients ───────────────────────────────────────────
    groq_client = Groq(api_key=GROQ_API_KEY)
    or_client   = OpenAI(api_key=OPENROUTER_API_KEY, base_url='https://openrouter.ai/api/v1')

    # ── Groq Models ───────────────────────────────────────
    for m in [
        {'client': 'groq', 'model': 'llama-3.3-70b-versatile', 'name': 'Groq llama-3.3'},
        {'client': 'groq', 'model': 'llama3-70b-8192',          'name': 'Groq llama3-70b'},
        {'client': 'groq', 'model': 'llama-3.1-8b-instant',     'name': 'Groq llama-3.1-8b'},
    ]:
        try:
            groq_client.chat.completions.create(
                model=m['model'], messages=[{'role': 'user', 'content': 'hi'}],
                max_tokens=3, timeout=10)
            ACTIVE_MODELS.append(m)
            logger.info(f"✅ {m['name']}")
        except Exception as e:
            logger.warning(f"❌ {m['name']}: {str(e)[:30]}")

    # ── Qwen HF ───────────────────────────────────────────
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        client = InferenceClient(model='Qwen/Qwen2.5-72B-Instruct', token=HF_TOKEN)
        client.chat_completion(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=3)
        hf_client = client
        logger.info("✅ Qwen 72B HF")
    except Exception as e:
        logger.warning(f"❌ Qwen HF: {str(e)[:40]}")

    # ── OpenRouter ────────────────────────────────────────
    for model in ['qwen/qwen3-32b:free', 'meta-llama/llama-3.3-70b-instruct:free',
                  'deepseek/deepseek-v3:free', 'google/gemma-3-12b-it:free']:
        try:
            or_client.chat.completions.create(
                model=model, messages=[{'role': 'user', 'content': 'hi'}],
                max_tokens=3, timeout=10)
            working_or_models.append(model)
            logger.info(f"✅ OR: {model}")
            if len(working_or_models) >= 2: break
        except: pass

    # ── Embeddings ────────────────────────────────────────
    logger.info("🔄 Loading embeddings...")
    embeddings = SentenceTransformerEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # ── ChromaDB (Persistent) ─────────────────────────────
    chroma_client_persist = chromadb.PersistentClient(path=CHROMA_PATH)

    # لو الـ DB موجودة خلاص
    try:
        collection = chroma_client_persist.get_collection('saudi_legal_v3')
        if collection.count() > 100:
            logger.info(f"✅ ChromaDB loaded from disk: {collection.count()} chunks")
            vectorstore = Chroma(
                client=chroma_client_persist,
                collection_name='saudi_legal_v3',
                embedding_function=embeddings
            )
        else:
            raise Exception("Empty collection")
    except:
        logger.info("📥 Loading dataset from HuggingFace...")
        dataset = load_dataset(HF_REPO_ID, token=HF_TOKEN, split='train')
        logger.info(f"✅ {len(dataset)} articles")

        docs = [
            Document(
                page_content=item['text'],
                metadata={
                    'article_number': item.get('article_number', ''),
                    'law_name':       item.get('law_name', ''),
                    'law_type':       item.get('law_type', ''),
                    'source':         item.get('source', ''),
                }
            )
            for item in dataset if len(item.get('text', '')) > 30
        ]

        # قوانين إضافية
        for a in EXTRA_LAWS:
            docs.append(Document(page_content=a['text'], metadata={
                'article_number': a['article_number'],
                'law_name': a['law_name'],
                'law_type': a['law_type'],
                'source': a['source'],
            }))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks   = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=chroma_client_persist,
            collection_name='saudi_legal_v3'
        )
        logger.info(f"✅ ChromaDB created: {vectorstore._collection.count()} chunks")

    # ── BM25 ──────────────────────────────────────────────
    logger.info("🔄 Building BM25...")
    all_chunks     = vectorstore.get()
    bm25_texts     = all_chunks['documents']
    bm25_metadatas = all_chunks['metadatas']
    stop_words     = {'من','في','على','إلى','عن','مع','هي','هو','ما','لا','أن','إن'}

    def tokenize(text):
        return [w for w in text.split() if len(w) > 2 and w not in stop_words]

    bm25_index = BM25Okapi([tokenize(t) for t in bm25_texts])
    logger.info(f"✅ BM25: {len(bm25_texts)} docs")
    logger.info("✅ Saudi Legal AI Ready!")


# ══════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════
EXTRA_LAWS = [
    {'text': 'المادة الثالثة والثمانون: عند انتهاء عقد العمل يستحق العامل مكافأة عن مدة خدمته تحسب على أساس أجر نصف شهر عن كل سنة من السنوات الخمس الأولى، وأجر شهر عن كل سنة بعد ذلك.',
     'article_number': 'المادة الثالثة والثمانون', 'law_name': 'نظام العمل', 'law_type': 'نظام', 'source': 'hrsd.gov.sa'},
    {'text': 'المادة الثامنة والثمانون: إذا أنهى صاحب العمل عقد العمل دون سبب مشروع وجب عليه دفع تعويض يعادل أجر خمسة عشر يوماً عن كل سنة خدمة ولا يقل عن أجر شهرين.',
     'article_number': 'المادة الثامنة والثمانون', 'law_name': 'نظام العمل', 'law_type': 'نظام', 'source': 'hrsd.gov.sa'},
    {'text': 'المادة الخامسة والستون: لا يجوز تشغيل العامل أكثر من ثماني ساعات يومياً وثمان وأربعين ساعة في الأسبوع. وفي رمضان تنخفض إلى ست ساعات.',
     'article_number': 'المادة الخامسة والستون', 'law_name': 'نظام العمل', 'law_type': 'نظام', 'source': 'hrsd.gov.sa'},
    {'text': 'المادة الثالثة والستون: للعامل الذي أمضى سنة كاملة إجازة سنوية واحد وعشرون يوماً تزداد إلى ثلاثين يوماً إذا أمضى عشر سنوات.',
     'article_number': 'المادة الثالثة والستون', 'law_name': 'نظام العمل', 'law_type': 'نظام', 'source': 'hrsd.gov.sa'},
    {'text': 'المادة الثالثة والعشرون: لا يجوز فصل العامل بسبب تقدمه بشكوى. ويعد الفصل تعسفياً ويحق للعامل التعويض.',
     'article_number': 'المادة الثالثة والعشرون', 'law_name': 'نظام العمل', 'law_type': 'نظام', 'source': 'hrsd.gov.sa'},
    {'text': 'المادة الثالثة: يعاقب بالسجن مدة لا تزيد على سنة وبغرامة لا تزيد على خمسمائة ألف ريال كل شخص يرتكب جريمة الدخول غير المشروع لموقع إلكتروني.',
     'article_number': 'المادة الثالثة', 'law_name': 'نظام مكافحة الجرائم المعلوماتية', 'law_type': 'نظام', 'source': 'boe.gov.sa'},
    {'text': 'المادة الرابعة: لا يجوز معالجة البيانات الشخصية إلا لتحقيق الغرض المشروع مع الحصول على موافقة صريحة من صاحب البيانات.',
     'article_number': 'المادة الرابعة', 'law_name': 'نظام حماية البيانات الشخصية', 'law_type': 'نظام', 'source': 'boe.gov.sa'},
]

SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في الأنظمة والتشريعات السعودية.

⚠️ تنبيه مهم: هذه المعلومات للاستئناس فقط وليست استشارة قانونية معتمدة. يُنصح بمراجعة محامٍ مختص.

تعامل مع كل أنواع الأسئلة (عربي، إنجليزي، عامية).

طريقة الإجابة:
📋 المرجع: [اسم النظام] — [رقم المادة]
✅ الإجابة: [إجابة مباشرة]
📝 التفاصيل: [شرح مختصر]

قواعد:
1. العربية الفصحى فقط في الإجابة
2. اذكر رقم المادة دايماً
3. لا تخترع معلومات
4. للأسئلة العملية: أجب بنعم/لا أولاً"""

OUT_OF_SCOPE = """أنا مساعد قانوني متخصص في الأنظمة السعودية.

سؤالك خارج نطاق اختصاصي. ممكن أساعدك في:
• أنظمة وزارة العدل
• نظام العمل السعودي
• نظام مكافحة الجرائم المعلوماتية
• نظام الشركات وحماية البيانات

⚠️ تنبيه: المعلومات للاستئناس فقط وليست استشارة قانونية معتمدة."""

LAW_KEYWORDS = {
    'توثيق':'نظام التوثيق','كاتب عدل':'نظام التوثيق','موثق':'نظام التوثيق',
    'محامي':'نظام المحاماة','محاماة':'نظام المحاماة',
    'مزاولة مهنة':'نظام المحاماة',
    'إفلاس':'نظام الإفلاس','تحكيم':'نظام التحكيم',
    'إثبات':'نظام الإثبات','تنفيذ':'نظام التنفيذ',
    'متهم':'نظام الإجراءات الجزائية',
    'زواج':'نظام الأحوال الشخصية','طلاق':'نظام الأحوال الشخصية',
    'نفقة':'نظام الأحوال الشخصية','حضانة':'نظام الأحوال الشخصية',
    'عقار':'نظام التسجيل العيني للعقار',
    'قضاء':'نظام القضاء','قاضي':'نظام القضاء',
    'غسل أموال':'نظام مكافحة غسل الأموال',
    'أركان العقد':'نظام المعاملات المدنية',
    'موظف':'نظام العمل','عامل':'نظام العمل',
    'فصل':'نظام العمل','إجازة':'نظام العمل',
    'نهاية خدمة':'نظام العمل','صاحب عمل':'نظام العمل',
    'اشتغلت':'نظام العمل','مكافأة':'نظام العمل',
    'جرائم معلوماتية':'نظام مكافحة الجرائم المعلوماتية',
    'بيانات شخصية':'نظام حماية البيانات الشخصية',
}

QUERY_EXPANSION = {
    'شروط رخصة الموثق':           'يشترط في الموثق ما يأتي',
    'شروط مزاولة مهنة المحاماة':  'يشترط فيمن يزاول مهنة المحاماة مقيداً جدول ممارسين',
    'أركان العقد':                 'أركان العقد الإيجاب والقبول',
    'عقوبات غسل الأموال':         'يعاقب على جريمة غسل الأموال',
    'أحكام الطلاق':               'الطلاق حل عقد الزواج رجعي بائن',
    'حقوق المتهم':                 'يحق للمتهم الاستعانة بمحامي',
    'هل لي مكافأة':               'يستحق العامل مكافأة نهاية الخدمة',
    'اشتغلت':                     'مكافأة نهاية الخدمة يستحق العامل سنوات',
    'فصلوني':                     'إنهاء عقد العمل تعويض تعسف',
    'ساعات العمل':                'لا يجوز تشغيل العامل أكثر من ثماني ساعات',
}

SPELL_CORRECTIONS = {
    'مزاولت':'مزاولة','مهنه':'مهنة','المحاماه':'المحاماة',
    'عقوبت':'عقوبة','رخصه':'رخصة','السعوديه':'السعودية',
    'الجزائيه':'الجزائية','مكافاه':'مكافأة',
    'اجرات':'إجراءات','الاموال':'الأموال',
    'احكام':'أحكام','القاضى':'القاضي','فى':'في',
    'تعين':'تعيين','ساعت':'ساعات','غسيل':'غسل',
    '٣':'3','٤':'4','٥':'5','١':'1','٢':'2',
    'penality':'penalty','laudering':'laundering',
    'calculat':'calculate','servise':'service',
    'lawer':'lawyer','condtions':'conditions',
}

LEGAL_TERMS_AR = [
    'إجراءات','محاماة','توثيق','إفلاس','تحكيم',
    'مكافأة','الغرامة','العقوبة','السجن','غسل الأموال',
    'القاضي','المحكمة','الزواج','الطلاق','الحضانة',
    'ساعات العمل','مكافأة نهاية الخدمة',
]

ENGLISH_TO_ARABIC = {
    'end of service':        'مكافأة نهاية الخدمة',
    'end of servise':        'مكافأة نهاية الخدمة',
    'money laundering':      'غسل الأموال',
    'money laudering':       'غسل الأموال',
    'wrongfully terminated': 'فصل تعسفي',
    'wrongful termination':  'فصل تعسفي',
    'if fired':              'عند الفصل',
    'if terminated':         'عند الفصل',
    'i was fired':           'تم فصلي',
    'am i entitled':         'هل يحق لي',
    'what are my rights':    'ما هي حقوقي',
    'my rights':             'حقوقي',
    'how to calculate':      'كيف تحسب',
    'how is calculated':     'كيف تحسب',
    'working hours':         'ساعات العمل',
    'annual leave':          'الإجازة السنوية',
    'lawyer license':        'رخصة المحامي',
    'lawer license':         'رخصة المحامي',
    'data protection':       'حماية البيانات الشخصية',
    'cybercrime':            'الجرائم المعلوماتية',
    'my employer':           'صاحب العمل',
    'i worked':              'اشتغلت',
    'labor law':             'نظام العمل',
    'labour law':            'نظام العمل',
    'saudi arabia':          'المملكة العربية السعودية',
    'penalty for':           'عقوبة',
    'penality for':          'عقوبة',
    'conditions for':        'شروط',
    'what r ':               'ما هي ',
    'labor':'عمل','labour':'عمل','arbitration':'تحكيم',
    'bankruptcy':'إفلاس','lawyer':'محامي','judge':'قاضي',
    'marriage':'زواج','divorce':'طلاق','custody':'حضانة',
    'salary':'الأجر','employee':'عامل','employer':'صاحب عمل',
    'penalty':'عقوبة','fine':'غرامة','rights':'حقوق',
    'terminated':'فُصلت','dismissed':'فُصلت',
}

COLLOQUIAL = {
    'ايه':'ما','إيه':'ما','ايش':'ما','شو':'ما',
    'اللي':'الذي','عشان':'لأن','ازاي':'كيف',
    'امتى':'متى','فين':'أين','مين':'من','ليه':'لماذا',
}

ARABIC_PRACTICAL = [
    'كم ساعة','كم يوم','كم مدة','كم سنة','كم راتب',
    'ساعات العمل','أنا موظف','أنا عامل','اشتغلت',
    'فُصلت','فصلوني','صاحب العمل','هل لي','هل يحق',
    'هل أستحق','حقوقي','مستحقاتي',
]

LEGAL_KEYWORDS_ALL = list(LAW_KEYWORDS.keys()) + [
    'نظام','قانون','مادة','عقوبة','غرامة','سجن',
    'حق','شرط','إجراء','محكمة','دعوى','ساعة',
    'إجازة','مكافأة','تعويض','فصل','عقد عمل',
    'غسل الأموال','جرائم معلوماتية','بيانات شخصية',
    'أجر','راتب','دوام',
]

LEGAL_ENGLISH_ALL = [
    'law','legal','court','judge','regulation','article',
    'penalty','fine','imprisonment','right','obligation',
    'contract','labor','labour','employment',
    'arbitration','bankruptcy','notary','lawyer','attorney',
    'cybercrime','data protection','money laundering',
    'saudi','marriage','divorce','custody','salary',
    'wage','employee','employer','annual leave',
    'terminated','dismissed','wrongful','working hours',
    'end of service','maternity','overtime',
]


# ══════════════════════════════════════════════════════════
# Rate Limiter
# ══════════════════════════════════════════════════════════
class SmartRateLimiter:
    def __init__(self):
        self.requests  = {'groq': deque(), 'qwen_hf': deque(), 'or': deque()}
        self.limits    = {'groq': 28, 'qwen_hf': 25, 'or': 18}
        self.last_used = {'groq': 0, 'qwen_hf': 0, 'or': 0}

    def _clean(self, c):
        now = time.time()
        while self.requests[c] and now - self.requests[c][0] > 60:
            self.requests[c].popleft()

    def can_use(self, c):
        self._clean(c)
        if time.time() - self.last_used[c] < 0.5: return False
        return len(self.requests[c]) < self.limits[c]

    def record(self, c):
        self.requests[c].append(time.time())
        self.last_used[c] = time.time()

    def wait_time(self, c):
        self._clean(c)
        if len(self.requests[c]) < self.limits[c]: return 0
        return max(0, 60 - (time.time() - self.requests[c][0]))

rate_limiter = SmartRateLimiter()


# ══════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════
def _call_groq(messages):
    for m in sorted([m for m in ACTIVE_MODELS if m['client'] == 'groq'],
                    key=lambda x: 0 if '70b' in x['model'] else 1):
        try:
            r = groq_client.chat.completions.create(
                model=m['model'], max_tokens=1000, temperature=0.1, messages=messages)
            answer = r.choices[0].message.content
            arabic = sum(1 for c in answer if '\u0600' <= c <= '\u06ff')
            if arabic / max(len([c for c in answer if c.strip()]), 1) < 0.6: continue
            return answer, m['name']
        except Exception as e:
            if '429' in str(e): continue
    return None, None

def _call_qwen(messages):
    if not hf_client: return None, None
    r = hf_client.chat_completion(messages=messages, max_tokens=800)
    return r.choices[0].message.content, 'Qwen 72B HF'

def _call_or(messages):
    for model in working_or_models:
        try:
            r = or_client.chat.completions.create(model=model, max_tokens=1000, messages=messages)
            return r.choices[0].message.content, f'OR-{model.split("/")[1]}'
        except: continue
    return None, None

def generate_with_fallback(messages):
    for client_name, call_fn in [('groq', lambda: _call_groq(messages)),
                                  ('qwen_hf', lambda: _call_qwen(messages)),
                                  ('or', lambda: _call_or(messages))]:
        if rate_limiter.can_use(client_name):
            try:
                result, model = call_fn()
                if result and len(result.strip()) > 50:
                    rate_limiter.record(client_name)
                    return result, model
            except Exception as e:
                if '429' in str(e) or 'rate' in str(e).lower():
                    for _ in range(rate_limiter.limits[client_name]):
                        rate_limiter.requests[client_name].append(time.time())
                    continue
        else:
            wait = rate_limiter.wait_time(client_name)
            if wait > 0:
                time.sleep(min(wait + 1, 10))
                result, model = call_fn()
                if result:
                    rate_limiter.record(client_name)
                    return result, model
    return None, None


# ══════════════════════════════════════════════════════════
# Pipeline: Spell → Translate → Normalize → RAG
# ══════════════════════════════════════════════════════════
def correct_spelling(text: str) -> str:
    words, result = text.split(), []
    for word in words:
        clean = word.strip('؟،.')
        if clean in SPELL_CORRECTIONS:
            result.append(SPELL_CORRECTIONS[clean] + word[len(clean):])
            continue
        if len(word) >= 4 and any('\u0600' <= c <= '\u06ff' for c in word):
            match = process.extractOne(word, LEGAL_TERMS_AR, scorer=fuzz.ratio, score_cutoff=75)
            if match:
                result.append(match[0])
                continue
        result.append(word)
    return ' '.join(result)

def translate_to_arabic(question: str) -> str:
    q_lower = question.lower()
    result  = question
    for eng, ar in sorted(ENGLISH_TO_ARABIC.items(), key=lambda x: -len(x[0])):
        if eng.lower() in q_lower:
            result  = re.sub(re.escape(eng), ar, result, flags=re.IGNORECASE)
            q_lower = result.lower()
    remaining = [w for w in question.split() if any(c.isascii() and c.isalpha() for c in w) and len(w) > 3]
    for eng_word in remaining:
        match = process.extractOne(eng_word.lower(), list(ENGLISH_TO_ARABIC.keys()), scorer=fuzz.ratio, score_cutoff=70)
        if match:
            result = re.sub(re.escape(eng_word), ENGLISH_TO_ARABIC[match[0]], result, flags=re.IGNORECASE)
    return result.strip()

def full_pipeline_normalize(question: str):
    log = []
    corrected = correct_spelling(question)
    if corrected != question:
        log.append(f'spell: {corrected}')
        question = corrected
    if any(c.isascii() and c.isalpha() for c in question):
        translated = translate_to_arabic(question)
        if translated != question:
            log.append(f'translated: {translated}')
            question = translated
    question = ' '.join(question.split())
    question = re.sub(r'[؟?]+', '؟', question)
    for col, formal in COLLOQUIAL.items():
        question = re.sub(rf'\b{col}\b', formal, question, flags=re.IGNORECASE)
    question = question.strip()
    if question and not question.endswith('؟'):
        question += '؟'
    return question, log

def is_legal_question(question: str) -> bool:
    q_lower = question.lower()
    if any(p in question for p in ARABIC_PRACTICAL): return True
    if any(kw in question for kw in LEGAL_KEYWORDS_ALL): return True
    if any(kw in q_lower for kw in LEGAL_ENGLISH_ALL): return True
    practical_en = ['my employer','i work','i worked','i was fired','am i entitled',
                    'my rights','terminated','dismissed','wrongfully','working hours']
    if any(p in q_lower for p in practical_en): return True
    return False

def detect_question_type(question: str) -> str:
    q_lower = question.lower()
    practical = ['أنا موظف','أنا عامل','اشتغلت','فصلوني','هل لي','هل يحق',
                 'my employer','i worked','am i entitled','if fired']
    if any(p in q_lower for p in practical): return 'practical'
    if any(p in q_lower for p in ['penalty','عقوبة','غرامة','سجن','fine']): return 'penalty'
    return 'general'

def expand_query(question: str) -> list:
    if question in _expansion_cache: return _expansion_cache[question]
    result = [question]
    for pattern, expansion in QUERY_EXPANSION.items():
        if pattern in question:
            result = [question, expansion]
            _expansion_cache[question] = result
            return result
    cleaned = re.sub(r'^(ما هي|ما هو|هل|كيف|متى)\s+', '', question).replace('؟', '').strip()
    if cleaned and cleaned != question: result.append(cleaned)
    words = [w for w in question.split() if len(w) > 3 and w not in {'هي','هو','ما','في','على','من','إلى'}]
    if words: result.append(' '.join(words[:4]))
    _expansion_cache[question] = result
    return result

def bm25_search_fn(query, k=5, target_law=None):
    stop_words = {'من','في','على','إلى','عن','مع','هي','هو','ما','لا','أن','إن'}
    tokens  = [w for w in query.split() if len(w) > 2 and w not in stop_words]
    scores  = bm25_index.get_scores(tokens)
    results = []
    for idx in scores.argsort()[::-1]:
        if len(results) >= k or scores[idx] < 0.1: break
        meta = bm25_metadatas[idx]
        if target_law and meta.get('law_name') != target_law: continue
        results.append(Document(page_content=bm25_texts[idx], metadata=meta))
    return results

def rerank_docs(docs, question, target_law=None):
    qwords = [w for w in question.split() if len(w) > 2]
    scored = []
    for doc in docs:
        score = 0
        if target_law and doc.metadata.get('law_name') == target_law: score += 8
        score += sum(2 for w in qwords if w in doc.page_content)
        if 'المادة' in doc.metadata.get('article_number', ''): score += 3
        score += min(len(doc.page_content) // 200, 3)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]

def calculate_coverage(question, docs):
    if not docs: return 0.0
    words    = [w for w in question.split() if len(w) > 3]
    if not words: return 1.0
    all_text = ' '.join(d.page_content for d in docs)
    return sum(1 for w in words if w in all_text or (len(w) >= 4 and w[:4] in all_text)) / len(words)

def build_context(docs):
    parts = []
    for i, doc in enumerate(docs):
        law     = doc.metadata.get('law_name', '')
        article = doc.metadata.get('article_number', '')
        parts.append(f'[{"الأكثر صلة" if i==0 else f"مرجع {i+1}"}] {law} — {article}\n{doc.page_content}\n{"─"*40}')
    return '\n\n'.join(parts)

def post_process(answer, docs):
    answer = answer.strip()
    lines  = answer.split('\n')
    clean  = [l for l in lines
              if sum(1 for c in l if '\u0600' <= c <= '\u06ff') / max(len(l.replace(' ','')), 1) > 0.3
              or any(s in l for s in ['📋','✅','📝','⚠️','•','-','─'])]
    return '\n'.join(clean).strip() if clean else answer


def ask_legal_core(question: str) -> dict:
    original = question
    question, log = full_pipeline_normalize(question)

    if not is_legal_question(question):
        test_docs = vectorstore.similarity_search(question, k=3)
        if calculate_coverage(question, test_docs) < 0.4:
            return {'answer': OUT_OF_SCOPE, 'sources': [], 'coverage': 0, 'model': 'out_of_scope', 'log': log}

    queries = expand_query(question)
    q_type  = detect_question_type(question)

    target_law = None
    for keyword, law in sorted(LAW_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in question:
            target_law = law
            break

    top_k       = 6 if len(question.split()) <= 5 else 8 if len(question.split()) <= 10 else 10
    k_per_query = max(3, top_k // len(queries))
    all_docs    = []

    for q in queries:
        if target_law:
            docs = vectorstore.similarity_search(q, k=k_per_query, filter={'law_name': target_law})
            if len(docs) < 2:
                extra = vectorstore.similarity_search(q, k=2)
                docs += [d for d in extra if d not in docs]
        else:
            docs = vectorstore.similarity_search(q, k=k_per_query)
        all_docs.extend(docs)

    bm25_docs    = []
    keyword_docs = []
    for q in queries: bm25_docs.extend(bm25_search_fn(q, k=5, target_law=target_law))

    if target_law:
        all_in_law = vectorstore.get(where={'law_name': target_law})
        all_words  = set(w for q in queries for w in q.split() if len(w) > 2)
        scored_kw  = [(sum(1 for w in all_words if w in dt), dt, dm)
                      for dt, dm in zip(all_in_law['documents'], all_in_law['metadatas'])
                      if sum(1 for w in all_words if w in dt) >= 1]
        scored_kw.sort(reverse=True)
        keyword_docs = [Document(page_content=dt, metadata=dm) for _, dt, dm in scored_kw[:5]]

    seen, combined = set(), []
    for d in keyword_docs + bm25_docs + all_docs:
        key = d.page_content[:50]
        if key not in seen: seen.add(key); combined.append(d)

    final_docs = rerank_docs(combined, question, target_law)[:top_k]
    coverage   = calculate_coverage(question, final_docs)

    if coverage < 0.5 and target_law:
        for d in vectorstore.similarity_search(question, k=top_k) + bm25_search_fn(question, k=5):
            key = d.page_content[:50]
            if key not in seen: seen.add(key); combined.append(d)
        final_docs = rerank_docs(combined, question, target_law)[:top_k]
        coverage   = calculate_coverage(question, final_docs)

    if len(final_docs) == 0 or coverage < 0.35:
        if q_type == 'practical':
            docs = vectorstore.similarity_search(question, k=8, filter={'law_name': 'نظام العمل'})
            if docs:
                context = build_context(docs[:5])
                answer, model = generate_with_fallback([
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'المواد:\n{context}\n\nالسؤال: {question}\nملاحظة: سؤال عملي.'}
                ])
                if answer:
                    return {'answer': post_process(answer, docs[:5]),
                            'sources': [{'law': d.metadata.get('law_name',''), 'article': d.metadata.get('article_number','')} for d in docs[:3]],
                            'coverage': 50, 'model': model, 'log': log}
        return {'answer': OUT_OF_SCOPE, 'sources': [], 'coverage': 0, 'model': 'quality_check', 'log': log}

    context   = build_context(final_docs)
    type_hint = '\nملاحظة: سؤال عملي — أجب بنعم/لا ثم اشرح الحق.' if q_type == 'practical' else \
                '\nملاحظة: اذكر العقوبة بدقة مع رقم المادة.' if q_type == 'penalty' else ''

    answer, model_used = generate_with_fallback([
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': f'المواد:\n{context}\n\nالسؤال: {question}{type_hint}'}
    ])

    if not answer:
        return {'answer': 'كل الموديلات محجوزة حالياً. حاول مرة أخرى.', 'sources': [], 'coverage': 0, 'model': '', 'log': log}

    return {
        'answer':   post_process(answer, final_docs),
        'sources':  [{'law': d.metadata.get('law_name',''), 'article': d.metadata.get('article_number','')} for d in final_docs[:3]],
        'coverage': round(coverage * 100),
        'model':    model_used,
        'log':      log
    }


# ══════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════
app = FastAPI(
    title='⚖️ Saudi Legal AI',
    description='نظام الذكاء الاصطناعي للقانون السعودي — وزارة العدل',
    version='3.0.0',
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


# ── Auth ──────────────────────────────────────────────────
def verify_api_key(request: Request):
    api_key = request.headers.get('X-API-Key') or request.query_params.get('api_key')
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail='API Key غلط أو ناقص')
    return api_key


# ── Models ────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    include_sources: bool = True

class QuestionResponse(BaseModel):
    answer:   str
    sources:  list
    coverage: int
    model:    str
    duration_ms: int
    disclaimer: str = "⚠️ هذه المعلومات للاستئناس فقط وليست استشارة قانونية معتمدة. يُنصح بمراجعة محامٍ مختص."


# ── Endpoints ─────────────────────────────────────────────
@app.get('/')
def root():
    return {
        'name':    'Saudi Legal AI ⚖️',
        'version': '3.0.0',
        'status':  'running',
        'docs':    '/docs'
    }

@app.get('/health')
def health():
    return {
        'status':  'healthy',
        'chunks':  vectorstore._collection.count() if vectorstore else 0,
        'models':  [m['name'] for m in ACTIVE_MODELS],
        'qwen':    bool(hf_client),
        'or_models': len(working_or_models),
        'version': '3.0.0'
    }

@app.post('/ask', response_model=QuestionResponse)
async def ask(
    req: QuestionRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail='السؤال فاضي!')

    if len(req.question) > 1000:
        raise HTTPException(status_code=400, detail='السؤال طويل جداً (الحد 1000 حرف)')

    ip = request.headers.get('X-Forwarded-For', 'unknown').split(',')[0].strip()
    t0 = time.time()

    try:
        result = ask_legal_core(req.question)
    except Exception as e:
        logger.error(f"Error: {e}")
        stats['errors'] += 1
        raise HTTPException(status_code=500, detail='خطأ في المعالجة')

    ms     = int((time.time() - t0) * 1000)
    status = 'blocked' if result['model'] in ['out_of_scope', 'quality_check'] else 'success'

    # Log
    request_log.appendleft({
        'time': datetime.now().strftime('%H:%M:%S'),
        'ip': ip, 'question': req.question[:60],
        'model': result['model'], 'status': status, 'ms': ms
    })
    stats['total']  += 1
    stats[status if status in stats else 'errors'] += 1
    active_ips[ip]  = active_ips.get(ip, 0) + 1

    return QuestionResponse(
        answer   = result['answer'],
        sources  = result['sources'] if req.include_sources else [],
        coverage = result['coverage'],
        model    = result['model'],
        duration_ms = ms
    )

@app.get('/stats')
def get_stats(api_key: str = Depends(verify_api_key)):
    return {
        'total':      stats['total'],
        'success':    stats['success'],
        'blocked':    stats['blocked'],
        'errors':     stats['errors'],
        'unique_ips': len(active_ips),
        'top_users':  sorted(active_ips.items(), key=lambda x: -x[1])[:5],
        'recent':     list(request_log)[:10]
    }


# ══════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
