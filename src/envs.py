import os

from huggingface_hub import HfApi, login

TOKEN = os.environ.get("TOKEN")  # A read/write token for your org

# Leaderboard ecosystem
OWNER = "la-leaderboard"
REPO_ID = f"{OWNER}/backend"
QUEUE_REPO = f"{OWNER}/requests-wip"  # TODO: Remove wip
RESULTS_REPO = f"{OWNER}/results-wip"  # TODO: Remove wip
LOGS_REPO = f"{OWNER}/logs"

# Evaluation variables
DEVICE = "cuda:0"
LIMIT = None  # TODO: Should be None for actual evaluations
NUM_FEWSHOT = 5  # TODO: Remove to use each task's default number of few-shots
LEADERBOARD_GROUP = None  # TODO: Update leaderboard group name
PARALLELIZE = True

# Cache setup
CACHE_PATH = os.getenv("HF_HOME", ".")  # /data/.huggingface
# Local caches
EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
EVAL_REQUESTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-queue-bk")
EVAL_RESULTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-results-bk")

# Backend variables
REFRESH_RATE = 10 * 60  # 10 min
NUM_LINES_VISUALIZE = 300

# Login to HuggingFace
login(token=TOKEN)
API = HfApi(token=TOKEN)

# Get hardware info
runtime = API.get_space_runtime(repo_id=REPO_ID)
HARDWARE = f"HuggingFace {runtime.hardware}"

# Evaluation tasks
TASKS_ES_NEW = [
    "aquas",
    "noticia",
    "ragquas",
    "teleia",
    "spalawex",
    "clintreates",
    "clindiagnoses",
    "humorqa",
    "crows_pairs_spanish",
    "offendes",
    "fake_news_es",
]

TASKS_ES_IBERO = [
    "belebele_spa_Latn",  # ERROR
    "escola",
    "mgsm_direct_es",
    "paws_es",
    "wnli_es",  # ERROR
    "xlsum_es",
    "xnli_es",
    "xquad_es",
    "xstorycloze_es",
]

TASKS_ES = TASKS_ES_NEW + TASKS_ES_IBERO

TASKS_EU_NEW = [
    "bec2016eu",
    "bertaqa_eu",
    "bhtc_v2",
    "epec_koref_bin",
    "eus_exams_eu",
    "eus_proficiency",
    "eus_reading",
    "eus_trivia",
    "qnlieu",
    "vaxx_stance",
    "wiceu",
]

TASKS_EU_IBERO = [
    "belebele_eus_Latn",  # ERROR
    "mgsm_direct_eu",
    "wnli_eu",  # ERROR
    "xcopa_eu",
    "xnli_eu",
    "xstorycloze_eu",
]

TASKS_EU = TASKS_EU_NEW + TASKS_EU_IBERO

TASKS_CA = [
    "arc_ca_aina",
    "belebele_cat_Latn",  # ERROR
    "cabreu",
    "catalanqa",
    "catcola",
    "copa_ca",
    "coqcat",
    "mgsm_direct_ca",
    "openbookqa_ca",
    "parafraseja",
    "paws_ca",
    "piqa_ca",
    "siqa_ca",
    "teca",
    "wnli_ca",  # ERROR
    "xnli_ca",
    "xquad_ca",
    "xstorycloze_ca",
]

TASKS_GL = [
    "belebele_glg_Latn",
    "galcola",
    "mgsm_direct_gl",
    "openbookqa_gl",
    "parafrases_gl",
    "paws_gl",
    "summarization_gl",
    "truthfulqa_gl",
]

TASKS_IBERO = TASKS_ES_IBERO + TASKS_EU_IBERO + TASKS_CA + TASKS_GL

TASKS_NEW = TASKS_ES_NEW + TASKS_EU_NEW

TASKS_ALL = TASKS_ES + TASKS_CA + TASKS_EU + TASKS_GL

TASKS_HARNESS = TASKS_ALL
