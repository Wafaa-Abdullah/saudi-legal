"""
Saudi Legal AI — Production v10.1
وزارة العدل السعودية | Ministry of Justice
Fixed: Mobile Responsive UI, Deduplicated Sources (Law + Article), Removed Slang Question.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import re, time, json, hashlib
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from groq import Groq
from huggingface_hub import login
import google.generativeai as genai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import chromadb

# ══════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_TOKEN     = os.getenv("HF_TOKEN", "").strip()
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "").strip()
HF_REPO_ID   = os.getenv("HF_REPO_ID", "WafaaFraih/saudi-legal-moj").strip()

vectorstore:      Optional[Chroma]    = None
bm25_index:       Optional[BM25Okapi] = None
bm25_texts:       list = []
bm25_metas:       list = []
ACTIVE_MODELS:    list = []
GEMINI_AVAILABLE: bool = False
gemini_clients:   dict = {}
groq_client:      Optional[Groq] = None
unique_items:     list = []

request_log   = deque(maxlen=500)
api_stats     = defaultdict(int)
user_ips      = defaultdict(list)
_answer_cache = {}

GEMINI_PRIORITY = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemma-3-27b-it",
]

GEMINI_SAFETY = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
]

# ══════════════════════════════════════════════════════════
# Dictionaries
# ══════════════════════════════════════════════════════════
COLLOQUIAL = {
    "ايه":           "ما",
    "إيه":           "ما",
    "ايش":           "ما",
    "وش":            "ما",
    "اللي":          "الذي",
    "عشان":          "لأن",
    "ازاي":          "كيف",
    "امتى":          "متى",
    "فين":           "أين",
    "مين":           "من",
    "ليه":           "لماذا",
    "عندي":          "لدي",
    "اشتغلت":        "عملت",
    "فصلوني":        "تم فصلي",
    "طردوني":        "تم فصلي",
    "اتطردت":        "تم فصلي",
    "اتفصلت":        "تم فصلي",
    "رفدوني":        "تم فصلي",
    "اترفدت":        "تم فصلي",
    "رفدي":          "تم فصلي",
    "مش":            "لا",
    "كمان":          "أيضاً",
    "مكافاه":        "مكافأة",
    "مكافأه":        "مكافأة",
    "رخصه":          "رخصة",
    "مزاوله":        "مزاولة",
    "المحاماه":      "المحاماة",
    "ليا":           "لي",
    "هل ليا":        "هل يحق لي",
    "الشغل":         "العمل",
    "شغل":           "عمل",
    "شغلي":          "عملي",
    "فلوس":          "مستحقات مالية",
    "عاوز":          "أريد",
    "عايز":          "أريد",
    "عاوزة":         "أريد",
    "عايزة":         "أريد",
    "الحبس":         "السجن",
    "قبضوا علي":     "تم توقيفي",
    "وقفوني":        "تم توقيفي",
    "اتهمت":         "تم اتهامي",
    "هستقيل":        "استقالة",
    "استقيل":        "استقالة",
    "اطلق مراتي":    "أريد طلاق زوجتي",
    "اطلق جوزي":     "أريد الطلاق من زوجي",
    "عايز اطلق":     "أريد الطلاق",
    "عاوز اطلق":     "أريد الطلاق",
    "عايزة اطلق":    "أريد الطلاق",
    "عاوزة اطلق":    "أريد الطلاق",
    "اتطلق":         "طلاق",
    "اتطلقنا":       "طلقنا",
    "مراتي":         "زوجتي",
    "جوزي":          "زوجي",
    "جوزها":         "زوجها",
}

LAW_KEYWORDS = {
    "توثيق":                    "نظام التوثيق",
    "كاتب عدل":                 "نظام التوثيق",
    "موثق":                     "نظام التوثيق",
    "محاماة":                   "نظام المحاماة",
    "محامي":                    "نظام المحاماة",
    "ترخيص محاماة":             "نظام المحاماة",
    "مزاولة مهنة المحاماة":    "نظام المحاماة",
    "إفلاس":                    "نظام الإفلاس",
    "إعسار":                    "نظام الإفلاس",
    "تحكيم":                    "نظام التحكيم",
    "إثبات":                    "نظام الإثبات",
    "شاهد":                     "نظام الإثبات",
    "بينة":                     "نظام الإثبات",
    "تنفيذ حكم":                "نظام التنفيذ",
    "تنفيذ":                    "نظام التنفيذ",
    "حقوق الموقوف":             "نظام الإجراءات الجزائية",
    "متهم":                     "نظام الإجراءات الجزائية",
    "توقيف":                    "نظام الإجراءات الجزائية",
    "تهمة":                     "نظام الإجراءات الجزائية",
    "محتجز":                    "نظام الإجراءات الجزائية",
    "جريمة":                    "نظام الإجراءات الجزائية",
    "جرائم":                    "نظام الإجراءات الجزائية",
    "قبض":                      "نظام الإجراءات الجزائية",
    "قتل":                      "نظام الإجراءات الجزائية",
    "تم اتهامي":                "نظام الإجراءات الجزائية",
    "تم توقيفي":                "نظام الإجراءات الجزائية",
    "النيابة العامة":           "نظام الإجراءات الجزائية",
    "ضبط جنائي":                "نظام الإجراءات الجزائية",
    "زواج":                     "نظام الأحوال الشخصية",
    "طلاق":                     "نظام الأحوال الشخصية",
    "أريد الطلاق":              "نظام الأحوال الشخصية",
    "أريد طلاق":                "نظام الأحوال الشخصية",
    "نفقة":                     "نظام الأحوال الشخصية",
    "حضانة":                    "نظام الأحوال الشخصية",
    "خلع":                      "نظام الأحوال الشخصية",
    "مهر":                      "نظام الأحوال الشخصية",
    "عدة":                      "نظام الأحوال الشخصية",
    "زوجتي":                    "نظام الأحوال الشخصية",
    "زوجي":                     "نظام الأحوال الشخصية",
    "زوجها":                    "نظام الأحوال الشخصية",
    "عقار":                     "نظام التسجيل العيني للعقار",
    "تسجيل عقار":               "نظام التسجيل العيني للعقار",
    "قضاء":                     "نظام القضاء",
    "قاضي":                     "نظام القضاء",
    "تعيين القضاة":             "نظام القضاء",
    "غسل الأموال":              "نظام مكافحة غسل الأموال",
    "غسيل الأموال":             "نظام مكافحة غسل الأموال",
    "تبييض الأموال":            "نظام مكافحة غسل الأموال",
    "أركان العقد":              "نظام المعاملات المدنية",
    "عقد مدني":                 "نظام المعاملات المدنية",
    "مكافأة نهاية الخدمة":     "نظام العمل",
    "نهاية الخدمة":             "نظام العمل",
    "صاحب عمل":                 "نظام العمل",
    "عقد عمل":                  "نظام العمل",
    "ساعات العمل":              "نظام العمل",
    "ساعة العمل":               "نظام العمل",
    "كم ساعة":                  "نظام العمل",
    "إجازة سنوية":              "نظام العمل",
    "إجازة الأمومة":            "نظام العمل",
    "إجازة أمومة":              "نظام العمل",
    "الأمومة":                  "نظام العمل",
    "علاوة":                    "نظام العمل",
    "العلاوة":                  "نظام العمل",
    "فصل تعسفي":                "نظام العمل",
    "تم فصلي":                  "نظام العمل",
    "فُصلت":                    "نظام العمل",
    "موظف":                     "نظام العمل",
    "عامل":                     "نظام العمل",
    "مكافأة":                   "نظام العمل",
    "استقالة":                  "نظام العمل",
    "راتب":                     "نظام العمل",
    "أجر":                      "نظام العمل",
    "شهادة خبرة":               "نظام العمل",
    "عمل إضافي":                "نظام العمل",
    "ساعات إضافية":             "نظام العمل",
    "مستحقات مالية":            "نظام العمل",
    "جرائم معلوماتية":          "نظام مكافحة الجرائم المعلوماتية",
    "اختراق إلكتروني":          "نظام مكافحة الجرائم المعلوماتية",
    "تشهير إلكتروني":           "نظام مكافحة الجرائم المعلوماتية",
    "قرصنة":                    "نظام مكافحة الجرائم المعلوماتية",
    "بيانات شخصية":             "نظام حماية البيانات الشخصية",
    "حماية البيانات":           "نظام حماية البيانات الشخصية",
    "خصوصية":                   "نظام حماية البيانات الشخصية",
    "مرافعات":                  "نظام المرافعات الشرعية",
    "دعوى":                     "نظام المرافعات الشرعية",
}

LEGAL_KEYWORDS = {
    "نظام","قانون","مادة","عقوبة","غرامة","سجن","محكمة","قاضي",
    "دعوى","متهم","عقد","زواج","طلاق","حضانة","إفلاس","تحكيم",
    "توثيق","محامي","عقار","موظف","عامل","مكافأة","تعويض",
    "إجازة","أجر","فصل","بيانات","راتب","مستحقات","ترخيص",
    "إثبات","تنفيذ","لائحة","موقوف","احتجاز","دين","سلف",
    "علاوة","العلاوة","استقالة","نفقة","مهر","خلع","شاهد",
    "حقوق","مستحقات","اتهام","توقيف","قضية","جريمة",
    "أريد الطلاق","طلاق","زوجتي","زوجي","مراتي","جوزي",
    "تم فصلي","فُصلت","شهادة خبرة","صاحب عمل","عقد عمل",
}

PRACTICAL_PATTERNS = {
    "تم فصلي", "فصلوني", "طردوني", "اتفصلت", "اترفدت", "رفدوني",
    "استقيل", "هستقيل", "استقالة", "صاحب عمل", "شهادة خبرة",
    "حقي", "حقوقي", "مستحقاتي", "هل يحق", "هل لي", "هل أستحق",
    "اشتغلت", "عملت", "اتطردت", "ما بتدفش", "ما دفعوا", "ما عطوني", 
    "ما عندي", "ليا فلوس", "هل ليا", "راتب", "علاوة", "العلاوة",
    "اطلق", "اتطلق", "عايزة اطلق", "عاوز اطلق",
    "مراتي", "جوزي", "جوزها", "زوجتي", "زوجي",
    "اتهمت", "اتهمني", "قبضوا علي", "وقفوني", "موقوف",
    "التحقيق", "تحقيق", "محتجز", "قضيتي", "قضيه", "حقوق",
}

NON_LEGAL = {
    "كورة", "فريق", "لاعب", "مباراة", "دوري",
    "طبخ", "أكل", "وصفة", "وصفه",
    "فيلم", "مسلسل", "ممثل",
    "سعر الذهب", "سعر الدولار", "بورصة", "سهم",
    "طقس", "حرارة", "مطر",
}

STOP_WORDS = {
    "من","في","على","إلى","عن","مع","هي","هو","ما","لا","أن","إن",
    "كل","هذا","ذلك","التي","الذي","هذه","تلك","قد","إذا","إذ",
}

QUERY_EXPANSION = {
    "مكافأة نهاية الخدمة": "يستحق العامل مكافأة عن مدة خدمته نصف شهر سنة",
    "نهاية الخدمة":        "يستحق العامل مكافأة عن مدة خدمته نصف شهر سنة",
    "فصلوني":              "إنهاء عقد العمل تعسفي تعويض فصل",
    "تم فصلي":             "إنهاء عقد العمل دون سبب مشروع تعويض",
    "حقوقي":               "يستحق العامل مكافأة تعويض نظام العمل",
    "ساعات العمل":         "لا يجوز تشغيل العامل أكثر من ثماني ساعات يومياً",
    "اتهمت":               "حق المتهم في محامٍ أثناء التحقيق الجزائي",
    "شروط مزاولة":         "يشترط فيمن يزاول المحاماة سعودي جدول المحامين",
    "أريد الطلاق":         "أحكام الطلاق الرجعي البائن نظام الأحوال الشخصية",
    "طلاق":                "أحكام الطلاق الرجعي البائن نظام الأحوال الشخصية",
    "زوجتي":               "أحكام الطلاق نظام الأحوال الشخصية الفرقة بين الزوجين",
    "زوجي":                "أحكام الطلاق الخلع نظام الأحوال الشخصية",
    "علاوة":               "أجر العامل راتب إجازة نظام العمل",
    "العلاوة":             "أجر العامل راتب إجازة نظام العمل",
    "استقالة":             "إنهاء عقد العمل بإرادة العامل نظام العمل",
}

MANUAL_DOCS = [
    {"text":"المادة الثالثة والثمانون: عند انتهاء عقد العمل يستحق العامل مكافأة عن مدة خدمته تحسب على أساس أجر نصف شهر عن كل سنة من السنوات الخمس الأولى، وأجر شهر عن كل سنة بعد ذلك. وتحسب المكافأة على أساس آخر أجر تقاضاه العامل.","article_number":"المادة الثالثة والثمانون","law_name":"نظام العمل"},
    {"text":"المادة الثامنة والثمانون: إذا أنهى صاحب العمل عقد العمل دون سبب مشروع وجب عليه دفع تعويض يعادل أجر خمسة عشر يوماً عن كل سنة خدمة ولا يقل عن أجر شهرين.","article_number":"المادة الثامنة والثمانون","law_name":"نظام العمل"},
    {"text":"المادة التاسعة والثمانون: لا يجوز لصاحب العمل فصل العامل بسبب تقدمه بشكوى أو مطالبته بحقوقه. وإذا أثبت العامل أن الفصل كان تعسفياً وجب دفع تعويض عادل إضافة إلى مكافأة نهاية الخدمة.","article_number":"المادة التاسعة والثمانون","law_name":"نظام العمل"},
    {"text":"المادة الخامسة والستون: لا يجوز تشغيل العامل أكثر من ثماني ساعات في اليوم أو ثماني وأربعين ساعة في الأسبوع. وفي شهر رمضان تنخفض ساعات العمل للمسلمين إلى ست ساعات يومياً.","article_number":"المادة الخامسة والستون","law_name":"نظام العمل"},
    {"text":"المادة الثامنة والستون: يستحق العامل عن ساعات العمل الإضافية أجراً إضافياً لا يقل عن أجره الأصلي مضافاً إليه خمسون بالمئة من ذلك الأجر.","article_number":"المادة الثامنة والستون","law_name":"نظام العمل"},
    {"text":"المادة الثالثة والستون: تُحدَّد مدة الإجازة السنوية بواحد وعشرين يوماً لكل سنة عمل، وتُزاد إلى ثلاثين يوماً إذا أمضى العامل خمس سنوات متواصلة في خدمة صاحب عمل واحد.","article_number":"المادة الثالثة والستون","law_name":"نظام العمل"},
    {"text":"المادة الثالثة والثلاثون: تستحق العاملة إجازة وضع بأجر كامل مدتها عشرة أسابيع قبل الوضع وبعده. ويُحظر تشغيل المرأة في الأسابيع الستة التالية للوضع مباشرة.","article_number":"المادة الثالثة والثلاثون","law_name":"نظام العمل"},
    {"text":"المادة الخامسة والسبعون: إذا أراد أحد طرفي عقد العمل غير المحدد المدة إنهاءه وجب عليه إخطار الطرف الآخر كتابةً قبل الإنهاء بستين يوماً للعمال الشهريين وثلاثين يوماً لغيرهم.","article_number":"المادة الخامسة والسبعون","law_name":"نظام العمل"},
    {"text":"المادة السابعة والعشرون: عند انتهاء عقد العمل يلتزم صاحب العمل بإعطاء العامل شهادة خبرة مجانية تبين فيها مدة خدمته ونوع عمله ومقدار أجره.","article_number":"المادة السابعة والعشرون","law_name":"نظام العمل"},
    {"text":"المادة الثالثة عشرة: يلتزم صاحب العمل بدفع الأجر المتفق عليه في مواعيده النظامية. ولا يجوز خصم أي مبالغ من أجر العامل إلا في الحالات المحددة نظاماً.","article_number":"المادة الثالثة عشرة","law_name":"نظام العمل"},
    {"text":"المادة الثانية: لا يجوز القبض على أي إنسان أو توقيفه أو سجنه أو تفتيشه إلا في الأحوال المنصوص عليها نظاماً، ولا يُعامل المقبوض عليه إلا بما يحفظ كرامته الإنسانية.","article_number":"المادة الثانية","law_name":"نظام الإجراءات الجزائية"},
    {"text":"المادة التاسعة والستون: يحق للمتهم الاستعانة بمحامٍ أو وكيل خلال مراحل التحقيق والمحاكمة. ولا يجوز التحقيق مع المتهم في الجرائم الكبيرة إلا بحضور محامٍ.","article_number":"المادة التاسعة والستون","law_name":"نظام الإجراءات الجزائية"},
    {"text":"المادة الحادية والسبعون: يجب على المحقق إخبار المتهم عند مثوله أمامه بالتهمة المنسوبة إليه. وللمتهم رفض الإجابة حتى يحضر محاميه.","article_number":"المادة الحادية والسبعون","law_name":"نظام الإجراءات الجزائية"},
    {"text":"المادة الثالثة: يعاقب بالسجن مدة لا تزيد على سنة وبغرامة لا تزيد على خمسمائة ألف ريال كل شخص يرتكب جريمة الدخول غير المشروع إلى الأنظمة المعلوماتية أو التشهير عبر الفضاء الإلكتروني.","article_number":"المادة الثالثة","law_name":"نظام مكافحة الجرائم المعلوماتية"},
    {"text":"المادة السادسة: يعاقب بالسجن مدة لا تزيد على خمس سنوات وبغرامة لا تزيد على ثلاثة ملايين ريال كل شخص يرتكب جريمة اختراق الأنظمة الحاسوبية الحكومية أو يتسبب في تعطيلها.","article_number":"المادة السادسة","law_name":"نظام مكافحة الجرائم المعلوماتية"},
    {"text":"المادة الرابعة: لا يجوز معالجة البيانات الشخصية إلا لتحقيق الغرض المشروع الذي جُمعت من أجله مع الحصول على موافقة صريحة من صاحب البيانات قبل معالجتها.","article_number":"المادة الرابعة","law_name":"نظام حماية البيانات الشخصية"},
    {"text":"المادة التاسعة والعشرون: يعاقب على الإفصاح عن البيانات الشخصية دون وجه حق بالسجن مدة لا تزيد على سنتين وبغرامة لا تزيد على ثلاثة ملايين ريال.","article_number":"المادة التاسعة والعشرون","law_name":"نظام حماية البيانات الشخصية"},
    {"text":"المادة الثالثة: يشترط فيمن يزاول مهنة المحاماة: أن يكون سعودي الجنسية بالأصل، حاصلاً على شهادة البكالوريوس في الشريعة الإسلامية أو الأنظمة، مقيداً اسمه في جدول المحامين الممارسين، غير محكوم بعقوبة مخلة بالشرف.","article_number":"المادة الثالثة","law_name":"نظام المحاماة"},
    {"text":"المادة الثانية والثلاثون: يشترط فيمن يُعيَّن في وظائف القضاء أن يكون سعودي الجنسية بالأصل، مسلماً، حاصلاً على شهادة الأهلية في الشريعة الإسلامية أو ما يعادلها، وأن ينجح في الامتحان المقرر.","article_number":"المادة الثانية والثلاثون","law_name":"نظام القضاء"},
]

# ══════════════════════════════════════════════════════════
# Rate Limiter
# ══════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, rpm=28):
        self._q = deque(); self._rpm = rpm
    def _clean(self):
        now = time.time()
        while self._q and now - self._q[0] > 60: self._q.popleft()
    def can_use(self):  self._clean(); return len(self._q) < self._rpm
    def record(self):   self._q.append(time.time())
    def wait_secs(self):
        self._clean()
        if len(self._q) < self._rpm: return 0.0
        return max(0.0, 60 - (time.time() - self._q[0]))
groq_limiter = RateLimiter(rpm=28)

# ══════════════════════════════════════════════════════════
# Text Helpers
# ══════════════════════════════════════════════════════════
def clean_arabic(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(r"ه\b", "ة", text)
    text = re.sub(r"[أإآ]", "ا", text)
    return text

def normalize_question(question: str) -> str:
    question = clean_arabic(question)
    for dial, formal in sorted(COLLOQUIAL.items(), key=lambda x: -len(x[0])):
        question = re.sub(rf"\b{re.escape(dial)}\b", formal, question, flags=re.IGNORECASE)
    question = re.sub(r"[؟?]+", "؟", question).strip()
    if not question.endswith("؟"):
        question += "؟"
    return question

def eng_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha: return 0.0
    return sum(1 for c in alpha if c.isascii()) / len(alpha)

def dict_translate(q: str) -> str:
    ENG = {
        "conditions for lawyer": "شروط مزاولة مهنة المحاماة", "lawyer license": "ترخيص محاماة",
        "end of service": "مكافأة نهاية الخدمة", "end of servise": "مكافأة نهاية الخدمة",
        "money laundering": "غسل الأموال", "wrongful termination": "فصل تعسفي",
        "unfair dismissal": "فصل تعسفي", "i was dismissed": "تم فصلي",
        "i was fired": "تم فصلي", "i got fired": "تم فصلي", "my rights": "حقوقي",
        "working hours": "ساعات العمل", "annual leave": "الإجازة السنوية",
        "maternity leave": "إجازة الأمومة", "overtime": "العمل الإضافي",
        "data protection": "حماية البيانات الشخصية", "cybercrime": "الجرائم المعلوماتية",
        "hacking": "اختراق إلكتروني", "bankruptcy": "إفلاس", "arbitration": "تحكيم",
        "divorce": "طلاق", "custody": "حضانة", "alimony": "نفقة", "labor law": "نظام العمل",
        "labour law": "نظام العمل", "salary": "الأجر", "employee": "عامل",
        "employer": "صاحب عمل", "penalty": "عقوبة", "fine": "غرامة", "imprisonment": "سجن",
        "evidence": "إثبات", "notary": "توثيق", "witness": "شاهد", "fraud": "احتيال",
        "bribery": "رشوة", "am i entitled": "هل يحق لي", "what are my rights": "ما هي حقوقي",
        "what is the penalty": "ما هي عقوبة", "how is calculated": "كيف تحسب",
        "gratuity": "مكافأة نهاية الخدمة",
    }
    lower = q.lower()
    for eng, ar in sorted(ENG.items(), key=lambda x: -len(x[0])):
        if eng in lower:
            q = re.sub(re.escape(eng), ar, q, flags=re.IGNORECASE)
            lower = q.lower()
    return q

# ══════════════════════════════════════════════════════════
# Core Pipeline Functions
# ══════════════════════════════════════════════════════════
def is_legal(original: str, normalized: str) -> bool:
    for text in [original, normalized]:
        t = clean_arabic(text.lower())
        if any(kw in t for kw in NON_LEGAL): return False
    all_legal_kws = [clean_arabic(kw) for kw in LEGAL_KEYWORDS | set(LAW_KEYWORDS.keys())]
    for text in [original, normalized]:
        t = clean_arabic(text)
        if any(kw in t for kw in all_legal_kws): return True
        if any(p in t for p in PRACTICAL_PATTERNS): return True
    return False

def detect_law(text: str) -> Optional[str]:
    t = clean_arabic(text)
    for kw, law in sorted(LAW_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if clean_arabic(kw) in t: return law
    return None

def fast_rewrite(question: str) -> list:
    queries = [question]
    for pat, exp in QUERY_EXPANSION.items():
        if pat in question:
            queries.append(exp)
            break
    words = [w for w in question.split() if len(w) > 3 and w not in STOP_WORDS]
    if words: queries.append(" ".join(words[:6]))
    return list(dict.fromkeys(queries))

def tokenize(text: str) -> list:
    return [w for w in text.split() if len(w) > 2 and w not in STOP_WORDS]

def bm25_search(query: str, k: int = 5, target_law: str = None) -> list:
    scores = bm25_index.get_scores(tokenize(query))
    docs   = []
    for idx in scores.argsort()[::-1]:
        if len(docs) >= k or scores[idx] < 0.05: break
        meta = bm25_metas[idx]
        if target_law and meta.get("law_name") != target_law: continue
        docs.append(Document(page_content=bm25_texts[idx], metadata=meta))
    return docs

PRIMARY_PREFIXES = [
    "نظام الإجراءات الجزائية", "نظام المحاماة", "نظام العمل",
    "نظام التوثيق", "نظام الأحوال الشخصية", "نظام الإفلاس",
    "نظام التحكيم", "نظام الإثبات", "نظام التنفيذ",
    "نظام المرافعات", "نظام القضاء", "نظام مكافحة",
    "نظام المعاملات", "نظام المحاكم", "نظام حماية", "نظام التسجيل",
]

def rerank_docs(docs: list, question: str, target_law: str = None) -> list:
    qwords = [w for w in clean_arabic(question).split() if len(w) > 2]
    scored = []
    for doc in docs:
        s = 0
        law_name = doc.metadata.get("law_name", "")
        content  = clean_arabic(doc.page_content)
        src      = doc.metadata.get("_src", "")
        
        if target_law:
            if law_name == target_law: s += 12
            elif law_name.startswith(target_law): s += 9
            
        if any(law_name.startswith(p) for p in PRIMARY_PREFIXES): s += 4
        if "لائحة تنفيذية" in law_name: s -= 2
        if "أدلة إجرائية" in law_name: s -= 2
        
        s += sum(2 for w in qwords if w in content)
        art = doc.metadata.get("article_number", "")
        if "المادة" in art and "فقرة" not in art and "نص كامل" not in art: s += 3
        elif "فقرة" in art or "نص كامل" in art: s -= 1
        
        s += min(len(doc.page_content) // 200, 3)
        if src == "manual": s += 3
        elif src == "hf": s += 1
        scored.append((s, doc))
    return [d for _, d in sorted(scored, key=lambda x: x[0], reverse=True)]

def calc_coverage(question: str, docs: list) -> float:
    if not docs: return 0.0
    laws = {d.metadata.get("law_name", "") for d in docs}
    base = 0.5 if any(law in laws for law in LAW_KEYWORDS.values()) else 0.0
    words = [w for w in question.split() if len(w) > 3]
    if not words: return base
    text = " ".join(d.page_content for d in docs)
    m = sum(1 for w in words if w in text or (len(w) >= 4 and w[:4] in text))
    return max(base, m / len(words))

def build_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs):
        label = "الأكثر صلة" if i == 0 else f"مرجع {i+1}"
        law   = doc.metadata.get("law_name", "")
        art   = doc.metadata.get("article_number", "")
        parts.append(f"[{label}] {law} — {art}\n{doc.page_content}\n{'─'*40}")
    return "\n\n".join(parts)

def post_process(answer: str, docs: list) -> str:
    answer = answer.strip()
    if docs and "📋" not in answer and "لم أجد" not in answer:
        law = docs[0].metadata.get("law_name", "")
        art = docs[0].metadata.get("article_number", "")
        if law:
            answer = f"📋 المرجع: {law} — {art}\n\n{answer}"
    return answer

# ══════════════════════════════════════════════════════════
# LLM Generation
# ══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """أنت مستشار قانوني سعودي محترف متخصص في أنظمة وزارة العدل ووزارة الموارد البشرية.
نطاق عملك:
• نظام العمل (مكافأة نهاية الخدمة، ساعات العمل، الفصل، الإجازات، الاستقالة...)
• أنظمة وزارة العدل (محاماة، توثيق، أحوال شخصية، إفلاس، تنفيذ، مرافعات، إثبات...)
• نظام الإجراءات الجزائية (حقوق المتهم، التوقيف، التحقيق...)
• نظام مكافحة الجرائم المعلوماتية
• نظام حماية البيانات الشخصية
صيغة الإجابة الإلزامية:
📋 المرجع: [اسم النظام] — [رقم المادة]
✅ الإجابة: [خلاصة مباشرة — استخدم نعم/لا للأسئلة العملية]
📝 التفاصيل: [شرح وافٍ ومترابط بناءً على النصوص فقط — لا تبتر الجمل أبدًا]
قواعد صارمة:
١. العربية الفصحى حصراً.
٢. استند للنصوص المرفقة فقط — لا تخترع معلومات.
٣. لا تنهِ إجابتك بكلمة ناقصة أو جملة مقطوعة أبداً.
٤. ابدأ بـ 📋 مباشرة بلا مقدمات.
٥. راجع إجابتك قبل الإرسال لتتأكد أنها مكتملة وواضحة.
⚠️ هذه المعلومات للاستئناس فقط وليست استشارة قانونية معتمدة."""

OUT_OF_SCOPE = """أنا مساعد قانوني متخصص في الأنظمة السعودية.
يمكنني مساعدتك في:
• نظام العمل (مكافأة نهاية الخدمة، ساعات العمل، الفصل، الاستقالة...)
• أنظمة وزارة العدل (محاماة، توثيق، أحوال شخصية، طلاق، إفلاس...)
• نظام الإجراءات الجزائية (حقوق المتهم، التوقيف...)
• نظام مكافحة الجرائم المعلوماتية
• نظام حماية البيانات الشخصية
⚠️ سؤالك خارج نطاق اختصاصي."""

def generate_answer(messages: list) -> tuple:
    prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    if GEMINI_AVAILABLE:
        for mn, client in gemini_clients.items():
            try:
                r = client.generate_content(
                    prompt,
                    safety_settings=GEMINI_SAFETY,
                    generation_config={"max_output_tokens": 2048, "temperature": 0.15},
                )
                text = (r.text or "").strip()
                if text: return text, mn.split("/")[-1]
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower(): continue
                print(f"  ⚠️  Gemini {mn}: {err[:60]}")
                continue
    wait = groq_limiter.wait_secs()
    if wait > 0: time.sleep(min(wait + 1, 30))
    if groq_limiter.can_use():
        for m in sorted(ACTIVE_MODELS, key=lambda x: 0 if "70b" in x["model"] else 1):
            try:
                r = groq_client.chat.completions.create(
                    model=m["model"], max_tokens=2048, temperature=0.15, messages=messages
                )
                groq_limiter.record()
                return r.choices[0].message.content, m["name"]
            except Exception as e:
                if "429" in str(e): continue
                raise
    return None, None

# ══════════════════════════════════════════════════════════
# Main ask()
# ══════════════════════════════════════════════════════════
def ask_legal(question: str) -> dict:
    cache_key = hashlib.md5(question.strip().encode()).hexdigest()
    if cache_key in _answer_cache:
        return {**_answer_cache[cache_key], "cache_hit": True}

    original   = question.strip()
    normalized = normalize_question(original)

    if eng_ratio(original) > 0.40:
        translated = dict_translate(original)
        if translated != original:
            normalized = normalize_question(translated)
            
    if not is_legal(original, normalized):
        return {"answer": OUT_OF_SCOPE, "sources": [], "coverage": 0, "model": "out_of_scope", "normalized": normalized}

    queries    = fast_rewrite(normalized)
    target_law = detect_law(normalized) or detect_law(original)

    labor_clues = {
        "مكافأة","نهاية الخدمة","راتب","أجر","ساعات","موظف","عامل",
        "فصلوني","اتفصلت","اترفدت","رفدوني","إجازة","أمومة",
        "شهادة خبرة","العمل","استقالة","مستحقات مالية","علاوة","العلاوة",
        "تم فصلي","فصل تعسفي","عمل إضافي",
    }
    anti_labor = {
        "محاماة","محامي","رخصة","إفلاس","قتل","جريمة",
        "طلاق","أريد الطلاق","زواج","نفقة","عقار","زوجتي","زوجي",
    }

    if not target_law:
        has_labor = any(w in normalized or w in original for w in labor_clues)
        has_anti  = any(w in normalized or w in original for w in anti_labor)
        if has_labor and not has_anti: target_law = "نظام العمل"

    seen, result_docs = set(), []
    def add_docs(new_docs):
        for d in new_docs:
            key = d.page_content[:60]
            if key not in seen:
                seen.add(key)
                result_docs.append(d)

    for q in queries:
        filt = {"law_name": target_law} if target_law else None
        add_docs(vectorstore.similarity_search(q, k=4, filter=filt) if filt else vectorstore.similarity_search(q, k=4))
        add_docs(bm25_search(q, k=4, target_law=target_law))

    if target_law:
        hits = sum(1 for d in result_docs if d.metadata.get("law_name") == target_law)
        if hits < 2:
            try:
                raw = vectorstore.get(where={"law_name": target_law})
                qw  = set(w for q in queries for w in q.split() if len(w) > 2)
                scored = sorted(
                    [(sum(1 for w in qw if w in dt), dt, dm)
                     for dt, dm in zip(raw["documents"], raw["metadatas"])
                     if sum(1 for w in qw if w in dt) >= 1],
                    reverse=True,
                )
                add_docs([Document(page_content=dt, metadata=dm) for _, dt, dm in scored[:5]])
            except Exception: pass

    final = rerank_docs(result_docs, normalized, target_law)[:6]
    cov   = calc_coverage(normalized, final)

    if not final:
        return {"answer": OUT_OF_SCOPE, "sources": [], "coverage": 0, "model": "low_coverage", "normalized": normalized}

    context  = build_context(final)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"المواد القانونية:\n\n{context}\n\nالسؤال: {normalized}"},
    ]

    ans, model_used = generate_answer(messages)
    if not ans:
        return {"answer": "الخدمة مشغولة — يرجى المحاولة مجدداً.", "sources": [], "coverage": 0, "model": "unavailable", "normalized": normalized}

    ans = post_process(ans, final)

    # ── FIX v10.1: Clean specific sources showing Law + Article properly
    seen_srcs = set()
    unique_sources = []
    for d in final:
        law = d.metadata.get("law_name", "").strip()
        art = d.metadata.get("article_number", "").strip()
        if not law: continue
        
        display_art = art.split("\n")[0][:40] if "المادة" in art else "نص قانوني"
        label = f"{law} — {display_art}"
        
        if label not in seen_srcs:
            seen_srcs.add(label)
            unique_sources.append({"law": law, "article": display_art, "label": label})
            if len(unique_sources) >= 3: break

    result = {
        "answer":     ans,
        "sources":    unique_sources,
        "coverage":   round(cov * 100),
        "model":      model_used,
        "normalized": normalized,
    }
    if result["model"] not in {"out_of_scope", "low_coverage", "unavailable", ""}:
        _answer_cache[cache_key] = result
    return result

# ══════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield

async def startup():
    global vectorstore, bm25_index, bm25_texts, bm25_metas
    global ACTIVE_MODELS, GEMINI_AVAILABLE, gemini_clients, groq_client, unique_items

    print("🚀 Saudi Legal AI v10.1 — Starting...")

    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        for mn in GEMINI_PRIORITY:
            try:
                client = genai.GenerativeModel(mn)
                client.generate_content("test", generation_config={"max_output_tokens": 1})
                gemini_clients[mn] = client
                GEMINI_AVAILABLE   = True
                print(f"  ✅ Gemini: {mn.split('/')[-1]}")
            except Exception as e:
                print(f"  ⚠️  Gemini {mn.split('/')[-1]}: {str(e)[:50]}")

    groq_client = Groq(api_key=GROQ_API_KEY)
    for m in [{"model": "llama-3.3-70b-versatile", "name": "Llama-3.3-70B"}, {"model": "llama-3.1-8b-instant", "name": "Llama-3.1-8B"}]:
        try:
            groq_client.chat.completions.create(model=m["model"], messages=[{"role": "user", "content": "hi"}], max_tokens=1)
            ACTIVE_MODELS.append(m)
            print(f"  ✅ Groq: {m['name']}")
        except Exception as e: print(f"  ⚠️  Groq {m['name']}: {str(e)[:50]}")

    hf_items = []
    print("📥 Loading HuggingFace dataset...")
    if HF_TOKEN: login(token=HF_TOKEN, add_to_git_credential=False)
    try:
        ds = load_dataset(HF_REPO_ID, token=HF_TOKEN, split="train")
        hf_items = [{"text": item.get("text", ""), "law_name": item.get("law_name", ""), "article_number": item.get("article_number", ""), "is_exec_reg": 1 if "لائحة" in item.get("law_name", "") else 0, "_src": "hf"} for item in ds if len(item.get("text", "")) > 30]
        print(f"  ✅ HF: {len(hf_items)} articles")
    except Exception as e: print(f"  ⚠️  HF load failed: {e}")

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "train.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f: train_raw = json.load(f)
        added = 0
        for record in train_raw:
            try:
                conv     = record.get("conversations", [])
                gpt_turn = next((c for c in conv if c.get("from") == "gpt"), None)
                if not gpt_turn: continue
                val    = json.loads(gpt_turn["value"])
                output = val.get("output", val)
                text   = (output.get("full_text", "") or "").strip()
                subj   = (output.get("subject", "") or "وزارة العدل").strip()
                if len(text) < 100: continue
                ar_ratio = sum(1 for c in text if "\u0600" <= c <= "\u06ff") / max(len(text), 1)
                if ar_ratio < 0.25: continue
                law_name = subj.split("/")[0].strip()
                law_name = re.sub(r"\(.*?\)", "", law_name).strip()
                if len(law_name) < 4: law_name = "وزارة العدل"
                ARTICLE_RE = re.compile(r"(?=المادة\s+(?:\(?\d+\)?|\w+))")
                parts = ARTICLE_RE.split(text)
                if len(parts) <= 1: parts = [text]
                for part in parts:
                    part = part.strip()
                    if len(part) < 50: continue
                    article_num = part.split("\n")[0].strip()[:100] if part.startswith("المادة") else "نص قانوني"
                    hf_items.append({"text": part[:3000], "law_name": law_name, "article_number": article_num, "is_exec_reg": 1 if "لائحة" in law_name else 0, "_src": "train"})
                    added += 1
            except Exception: pass
        print(f"  ✅ train.json: {added} articles")
    except FileNotFoundError: print(f"  ℹ️  train.json not found — continuing without it")

    for m in MANUAL_DOCS: hf_items.append({"text": m["text"], "law_name": m["law_name"], "article_number": m["article_number"], "is_exec_reg": 0, "_src": "manual"})
    
    seen_keys = set()
    unique_items = []
    for item in hf_items:
        key = re.sub(r"\s+", "", item["text"])[:80]
        if key and key not in seen_keys and len(item["text"]) > 30:
            seen_keys.add(key)
            unique_items.append(item)

    print("🔄 Building embeddings...")
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    docs_lc = [Document(page_content=item["text"], metadata={"law_name": item["law_name"], "article_number": item["article_number"], "is_exec_reg": item["is_exec_reg"], "_src": item["_src"]}) for item in unique_items]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, separators=["\nالمادة ", "\n\n", "\n", ".", " "])
    short = [d for d in docs_lc if len(d.page_content) <= 1500]
    long  = [d for d in docs_lc if len(d.page_content) > 1500]
    chunks = short + splitter.split_documents(long)
    
    chroma_client = chromadb.Client()
    vectorstore   = Chroma.from_documents(documents=chunks, embedding=embeddings, client=chroma_client, collection_name="saudi_legal_v10")
    raw        = vectorstore.get()
    bm25_texts = raw["documents"]
    bm25_metas = raw["metadatas"]
    bm25_index = BM25Okapi([tokenize(t) for t in bm25_texts])
    print("✅ Saudi Legal AI v10.1 — Ready!")

# ══════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════
app = FastAPI(title="Saudi Legal AI", version="10.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _rate_limit(ip: str, max_rpm: int = 15) -> bool:
    now = time.time(); user_ips[ip] = [t for t in user_ips[ip] if now - t < 60]
    if len(user_ips[ip]) >= max_rpm: return False
    user_ips[ip].append(now); return True

class QuestionRequest(BaseModel): question: str; include_sources: bool = True
class AnswerResponse(BaseModel): answer: str; model: str; coverage: int; sources: list = []; duration_ms: int = 0; disclaimer: str = "⚠️ للاستئناس فقط وليست استشارة قانونية معتمدة."

# ══════════════════════════════════════════════════════════
# Frontend HTML (Mobile fixes + Deduplicated sources)
# ══════════════════════════════════════════════════════════
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>المساعد القانوني السعودي</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@300;400;500;700;900&display=swap" rel="stylesheet">
<style>
  :root{--ink:#12100E;--parch:#F5F0E8;--gold:#B8860B;--gold-lt:#D4A843;--green:#1B4332;--cream:#FAF6EE;--border:#D4C9A8;--muted:#7A7060;--shadow:rgba(18,16,14,.12)}
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Noto Kufi Arabic',sans-serif;background:var(--parch);color:var(--ink);min-height:100vh;display:flex;flex-direction:column}
  header{background:var(--green);padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:64px;box-shadow:0 2px 16px var(--shadow);position:sticky;top:0;z-index:100}
  .hb{display:flex;align-items:center;gap:.85rem}
  #mb{background:transparent;border:none;color:var(--gold);font-size:1.5rem;cursor:pointer;display:none;margin-left:10px}
  .he{width:38px;height:38px;border:2px solid var(--gold);border-radius:50%;display:grid;place-items:center;font-size:1.1rem;color:var(--gold);flex-shrink:0}
  .ht{font-size:1rem;font-weight:700;color:var(--parch)}
  .hs{font-size:.68rem;color:rgba(245,240,232,.55);margin-top:1px}
  .badge{font-size:.65rem;background:rgba(184,134,11,.2);border:1px solid rgba(184,134,11,.4);color:var(--gold-lt);padding:3px 10px;border-radius:20px}
  .layout{display:flex;flex:1;overflow:hidden;position:relative;}
  .sidebar{width:260px;background:var(--cream);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow-y:auto;flex-shrink:0}
  .ss{padding:1.2rem 1rem;border-bottom:1px solid var(--border)}
  .sl{font-size:.65rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.75rem}
  .chip{display:block;width:100%;text-align:right;background:transparent;border:1px solid var(--border);border-radius:8px;padding:.55rem .8rem;font-family:'Noto Kufi Arabic',sans-serif;font-size:.78rem;color:var(--ink);cursor:pointer;margin-bottom:.45rem;transition:all .18s;line-height:1.4}
  .chip:hover{background:var(--green);color:var(--parch);border-color:var(--green)}
  .stat-row{display:flex;justify-content:space-between;font-size:.72rem;color:var(--muted);padding:.25rem 0;border-bottom:1px solid rgba(212,201,168,.4)}
  .sv{font-weight:700;color:var(--gold)}
  .chat-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden;max-width:780px;margin:0 auto;width:100%;padding:0 1rem}
  #chat{flex:1;overflow-y:auto;padding:1.5rem 0;display:flex;flex-direction:column;gap:1.25rem;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
  .msg{display:flex;gap:.75rem;animation:rise .3s ease}
  .msg.user{flex-direction:row-reverse}
  @keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
  .av{width:36px;height:36px;border-radius:50%;display:grid;place-items:center;font-size:.95rem;flex-shrink:0;margin-top:2px}
  .msg.bot .av{background:var(--green);color:var(--gold);border:1.5px solid var(--gold)}
  .msg.user .av{background:var(--gold);color:var(--green)}
  .bubble{max-width:84%;padding:.85rem 1.1rem;border-radius:16px;font-size:.88rem;line-height:1.8}
  .msg.bot .bubble{background:white;border:1px solid var(--border);border-top-right-radius:4px;box-shadow:0 2px 8px var(--shadow)}
  .msg.user .bubble{background:var(--green);color:var(--parch);border-top-left-radius:4px;box-shadow:0 2px 8px rgba(27,67,50,.25)}
  .ref{color:var(--gold);font-weight:700;font-size:.8rem;margin-bottom:.5rem}
  .ans{margin:.3rem 0}
  .meta{font-size:.65rem;color:var(--muted);margin-top:.6rem;border-top:1px solid var(--border);padding-top:.5rem}
  .srcs{margin-top:.6rem;display:flex;flex-wrap:wrap;gap:.3rem}
  .stag{font-size:.62rem;background:rgba(27,67,50,.06);border:1px solid var(--border);color:var(--green);padding:2px 8px;border-radius:20px}
  .dots span{display:inline-block;width:7px;height:7px;background:var(--gold);border-radius:50%;margin:0 2px;animation:bounce 1.1s infinite}
  .dots span:nth-child(2){animation-delay:.18s}
  .dots span:nth-child(3){animation-delay:.36s}
  @keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-7px)}}
  .input-wrap{border-top:1px solid var(--border);padding:.9rem 0;background:var(--parch)}
  .input-row{display:flex;gap:.65rem;align-items:flex-end}
  #q{flex:1;background:white;border:1.5px solid var(--border);border-radius:12px;padding:.75rem 1rem;font-family:'Noto Kufi Arabic',sans-serif;font-size:.9rem;color:var(--ink);resize:none;outline:none;min-height:48px;max-height:130px;line-height:1.6;transition:border-color .2s;box-shadow:inset 0 1px 4px var(--shadow)}
  #q:focus{border-color:var(--gold)}
  #q::placeholder{color:var(--border)}
  #sb{width:48px;height:48px;background:var(--green);border:none;border-radius:12px;cursor:pointer;font-size:1.15rem;color:var(--gold);flex-shrink:0;transition:all .2s;box-shadow:0 2px 8px rgba(27,67,50,.3);display:grid;place-items:center}
  #sb:hover:not(:disabled){background:#255c3f;transform:translateY(-1px)}
  #sb:disabled{opacity:.45;cursor:not-allowed;transform:none}
  .disc{text-align:center;font-size:.63rem;color:var(--muted);padding:.35rem 0}
  .welcome{background:white;border:1px solid var(--border);border-radius:16px;padding:1.5rem;text-align:center;box-shadow:0 2px 12px var(--shadow)}
  .wi{font-size:2.5rem;margin-bottom:.75rem}
  .welcome h2{font-size:1.1rem;font-weight:700;color:var(--green);margin-bottom:.4rem}
  .welcome p{font-size:.82rem;color:var(--muted);line-height:1.65}
  
  /* Mobile Styles */
  @media(max-width:700px){
      #mb{display:block}
      .sidebar{position:absolute;right:-260px;top:0;height:100%;z-index:1000;transition:right .3s ease;box-shadow:-2px 0 10px rgba(0,0,0,.1);display:flex!important}
      .sidebar.open{right:0}
      .chat-wrap{padding:0 .75rem}
      .bubble{max-width:92%}
  }
</style>
</head>
<body>
<header>
  <div class="hb">
    <button id="mb" onclick="document.querySelector('.sidebar').classList.toggle('open')">☰</button>
    <div class="he">⚖</div>
    <div><div class="ht">المساعد القانوني السعودي</div><div class="hs">أنظمة وزارة العدل · نظام العمل · الإجراءات الجزائية</div></div>
  </div>
  <span class="badge" id="badge">جارٍ التحميل…</span>
</header>
<div class="layout">
  <aside class="sidebar">
    <div class="ss">
      <div class="sl">أسئلة شائعة</div>
      <button class="chip" onclick="send('ما هي شروط مزاولة مهنة المحاماة؟')">شروط المحاماة</button>
      <button class="chip" onclick="send('كم ساعة العمل في اليوم نظاماً؟')">ساعات العمل</button>
      <button class="chip" onclick="send('فصلوني بدون سبب هل يحق لي تعويض؟')">تعويض الفصل</button>
      <button class="chip" onclick="send('كيف تحسب مكافأة نهاية الخدمة؟')">نهاية الخدمة</button>
      <button class="chip" onclick="send('ما هي حقوق الموقوف أثناء التحقيق؟')">حقوق الموقوف</button>
      <button class="chip" onclick="send('ما هي عقوبات غسل الأموال؟')">غسل الأموال</button>
      <button class="chip" onclick="send('ما هي أحكام الطلاق؟')">أحكام الطلاق</button>
      <button class="chip" onclick="send('ما هي إجراءات الإفلاس؟')">إجراءات الإفلاس</button>
      <button class="chip" onclick="send('شركتي ما بتدفش العلاوة السنوية')">العلاوة السنوية</button>
    </div>
    <div class="ss">
      <div class="sl">الإحصائيات</div>
      <div class="stat-row"><span>إجمالي الأسئلة</span><span class="sv" id="st">0</span></div>
      <div class="stat-row"><span>المجابة</span><span class="sv" id="ss2">0</span></div>
      <div class="stat-row"><span>خارج النطاق</span><span class="sv" id="sb2">0</span></div>
    </div>
    <div class="ss">
      <div class="sl">تنبيه قانوني</div>
      <p style="font-size:.7rem;color:var(--muted);line-height:1.65">المعلومات للاستئناس فقط وليست استشارة قانونية معتمدة. يُنصح بمراجعة محامٍ مختص.</p>
    </div>
  </aside>
  <div class="chat-wrap">
    <div id="chat">
      <div class="msg bot">
        <div class="av">⚖</div>
        <div class="bubble">
          <div class="welcome">
            <div class="wi">🏛️</div>
            <h2>مرحباً بك في المساعد القانوني السعودي</h2>
            <p>أجيب على أسئلتك في أنظمة وزارة العدل، نظام العمل، الإجراءات الجزائية، وغيرها.<br>اكتب سؤالك أو اختر من الأسئلة الشائعة.</p>
          </div>
        </div>
      </div>
    </div>
    <div class="input-wrap">
      <div class="input-row">
        <textarea id="q" placeholder="اكتب سؤالك القانوني هنا…" rows="1"
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();go();}"></textarea>
        <button id="sb" onclick="go()" title="إرسال">➤</button>
      </div>
      <div class="disc">⚠️ للاستئناس فقط · ليست استشارة قانونية معتمدة · وزارة العدل السعودية</div>
    </div>
  </div>
</div>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('q'),btn=document.getElementById('sb'),badge=document.getElementById('badge');
let tot=0,suc=0,blk=0;
fetch('/health').then(r=>r.json()).then(d=>{
  badge.textContent=`${(d.vectors||0).toLocaleString()} مادة · مباشر`;
  badge.style.cssText='background:rgba(27,67,50,.3);border-color:rgba(27,67,50,.6);color:#6fcf97';
}).catch(()=>{badge.textContent='متصل'});
function msg(role,html){
  const w=document.createElement('div');w.className=`msg ${role}`;
  w.innerHTML=`<div class="av">${role==='user'?'👤':'⚖'}</div><div class="bubble">${html}</div>`;
  chat.appendChild(w);chat.scrollTop=chat.scrollHeight;return w;
}
function loader(){return msg('bot','<div class="dots"><span></span><span></span><span></span></div>')}
function fmt(d){
  let h='';
  (d.answer||'').split('\n').forEach(l=>{
    l=l.trim();if(!l)return;
    if(l.startsWith('📋'))h+=`<div class="ref">${l}</div>`;
    else if(l.startsWith('✅'))h+=`<div class="ans"><strong>${l}</strong></div>`;
    else h+=`<div class="ans">${l}</div>`;
  });
  const s=(d.sources||[]);
  if(s.length){h+='<div class="srcs">';s.forEach(x=>{h+=`<span class="stag">📚 ${x.label || x.law}</span>`});h+='</div>'}
  if(d.model&&!['out_of_scope','low_coverage','unavailable'].includes(d.model))
    h+=`<div class="meta">🤖 ${d.model} · تغطية ${d.coverage}%</div>`;
  return h;
}
async function send(q){
  if(window.innerWidth <= 700) document.querySelector('.sidebar').classList.remove('open');
  if(!q?.trim()||btn.disabled)return;
  msg('user',q.replace(/</g,'&lt;'));inp.value='';inp.style.height='auto';btn.disabled=true;
  const load=loader();
  try{
    const res=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,include_sources:true})});
    const d=await res.json();load.remove();
    tot++;if(['out_of_scope','low_coverage'].includes(d.model))blk++;else suc++;
    document.getElementById('st').textContent=tot;
    document.getElementById('ss2').textContent=suc;
    document.getElementById('sb2').textContent=blk;
    msg('bot',res.ok?fmt(d):`<div style="color:#c0392b">⚠️ ${d.detail||'خطأ'}</div>`);
  }catch(e){load.remove();msg('bot','<div style="color:#c0392b">❌ تعذر الاتصال</div>')}
  finally{btn.disabled=false;inp.focus()}
}
function go(){send(inp.value.trim())}
inp.addEventListener('input',()=>{inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,130)+'px'});
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
def root():
    return FRONTEND_HTML

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "version": "10.1.0",
        "vectors": vectorstore._collection.count() if vectorstore else 0,
        "laws":    len(set(d.get("law_name", "") for d in bm25_metas)),
        "models":  [m["name"] for m in ACTIVE_MODELS] + (["Gemini"] if GEMINI_AVAILABLE else []),
    }

@app.get("/stats")
def stats():
    return {
        "total":      api_stats["total"],
        "success":    api_stats["success"],
        "blocked":    api_stats["blocked"],
        "errors":     api_stats["errors"],
        "unique_ips": len(user_ips),
    }

@app.get("/laws")
def laws():
    all_laws = sorted(set(m.get("law_name", "") for m in bm25_metas if m.get("law_name")))
    return {"laws": all_laws, "total": len(all_laws)}

@app.post("/ask", response_model=AnswerResponse)
async def ask_endpoint(req: QuestionRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "السؤال فاضي!")
    if len(req.question) > 1500:
        raise HTTPException(400, "السؤال طويل جداً (max 1500 حرف)")
    ip = request.headers.get("X-Forwarded-For", "unknown").split(",")[0].strip()
    if not _rate_limit(ip):
        raise HTTPException(429, "تجاوزت الحد — حاول بعد دقيقة")
    t0 = time.time()
    try:
        result = ask_legal(req.question.strip())
    except Exception as e:
        api_stats["errors"] += 1
        raise HTTPException(500, f"خطأ داخلي: {str(e)[:80]}")
    ms     = int((time.time() - t0) * 1000)
    status = "blocked" if result["model"] in {"out_of_scope", "low_coverage"} else "success"
    api_stats["total"]  += 1
    api_stats[status]   += 1
    request_log.appendleft({
        "time": time.strftime("%H:%M:%S"), "ip": ip,
        "q": req.question[:60], "model": result["model"],
        "ms": ms, "status": status,
    })
    return AnswerResponse(
        answer      = result["answer"],
        model       = result.get("model", ""),
        coverage    = result.get("coverage", 0),
        sources     = result.get("sources", []) if req.include_sources else [],
        duration_ms = ms,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)