import os
import json
from datetime import datetime
import streamlit as st
import glob

# ---- Optional: OpenAI SDK (new style) ----
# pip install openai
from openai import OpenAI

APP_TITLE = "人物力トレーニング（面接・討論・模擬授業・願書・小論文）MVP"
SAVE_DIR = "submissions"

SYSTEM_BASE = """あなたは人物評価試験（面接・討論・模擬授業・願書・小論文）の訓練コーチです。
受講者の主体性（根拠・責任・接続・再現性）を最優先に見て、過度に模範解答化しない。
必要なら深掘り質問を出し、空虚語（熱意・貢献・成長 など）のみで終わる文章を具体化させる。
出力は日本語。敬体で丁寧に。"""

RUBRIC_COMMON = {
    "主体性（自分の判断が出ているか）": 5,
    "根拠（事実・体験・具体）": 5,
    "接続（現場課題／他者／役割への接続）": 5,
    "再現性（同様状況でも使える形）": 5,
    "伝達（簡潔さ・論理・読みやすさ）": 5
}

MODULES = {
    "面接": {
        "prompt": """【面接訓練】
あなたは採用面接官です。受講者に質問→回答→深掘り質問を3往復行い、
最後にルーブリック採点と改善指示を出してください。
最初の質問は「志望動機を教えてください」です。
受講者の回答が空虚なら、必ず具体に落とす追加質問をしてください。"""
    },
    "討論": {
        "prompt": """【討論訓練】
あなたは討論ファシリテーターです。
テーマに対し、(1)論点を3つ提示 (2)議論順序を提案 (3)受講者の立場表明を促す質問
(4)相手反論を想定して問い返し (5)合意形成案の作り方 を順に行ってください。
最後にルーブリック採点と改善指示を出してください。"""
    },
    "模擬授業": {
        "prompt": """【模擬授業訓練】
あなたは模擬授業評価者です。
受講者が提示した「授業案」を、導入/目標/展開/支援/まとめ/評価の観点で点検し、
改善案（板書・発問・つまずき支援）を具体で提案してください。
最後にルーブリック採点と改善指示を出してください。"""
    },
    "願書": {
        "prompt": """【願書（志望理由・自己PR）添削訓練】
受講者の文章を、(1)主体 (2)現場課題との接続 (3)根拠の具体 (4)空虚語の削減
(5)読みやすさ の観点で添削してください。
「改善指示→改善例（200〜300字）→受講者への質問3つ」を必ず出してください。"""
    },
    "小論文": {
        "prompt": """【小論文添削訓練】
受講者の小論文を、設問解釈/論点/根拠/提案の実行条件/反対意見・副作用への配慮
の観点で添削してください。
「骨格（序本結）→改善指示→改善例の段落見本→次回課題」を出してください。"""
    }
}


def ensure_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def save_submission(module: str, user_text: str, ai_text: str, meta: dict):
    """SAVE_DIR に必ず保存する（保存先の統一）"""
    ensure_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{module}.json"
    path = os.path.join(SAVE_DIR, filename)

    data = {
        "timestamp": timestamp,
        "module": module,
        "user_text": user_text,
        "ai_text": ai_text,
        "meta": meta
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return path


def list_submissions(limit=50):
    """submissionsフォルダ内のjsonを新しい順で返す（見る場所の統一）"""
    ensure_dir()
    files = glob.glob(os.path.join(SAVE_DIR, "*.json"))
    files.sort(reverse=True)
    return files[:limit]


def load_submission(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_llm(client: OpenAI, module: str, theme: str, user_text: str):
    module_prompt = MODULES[module]["prompt"]
    user_payload = f"""【テーマ／条件】
{theme}

【受講者の入力】
{user_text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 低コストで運用向け。必要なら変更
        messages=[
            {"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": module_prompt + "\n\n" + user_payload}
        ],
        temperature=0.4
    )
    return resp.choices[0].message.content


def rubric_template():
    return "\n".join([f"- {k}: /{v}" for k, v in RUBRIC_COMMON.items()])


# ---- UI ----
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    st.warning("環境変数 OPENAI_API_KEY が未設定です。設定後に再読み込みしてください。")

col1, col2 = st.columns([1, 2])

with col1:
    module = st.selectbox("訓練モジュール", list(MODULES.keys()))
    theme = st.text_area("テーマ／条件（任意）", placeholder="例：教員採用試験（小学校）／志望動機／討論テーマなど")
    st.markdown("### 共通採点軸（ルーブリック）")
    st.text(rubric_template())

    st.markdown("---")
    st.markdown("## 保存履歴")

    files = list_submissions(limit=50)

    if not files:
        st.caption("まだ保存履歴がありません。AI実行後に表示されます。")
    else:
        labels = [os.path.basename(p) for p in files]
        selected = st.selectbox("履歴を選択", labels, index=0)

        selected_path = files[labels.index(selected)]
        data = load_submission(selected_path)

        st.markdown("### プレビュー")
        st.caption(f"module: {data.get('module')} / timestamp: {data.get('timestamp')}")
        st.text_area("受講者入力", value=data.get("user_text", ""), height=120)
        st.text_area("AI出力", value=data.get("ai_text", ""), height=180)

        st.download_button(
            label="この履歴をJSONでダウンロード",
            data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=os.path.basename(selected_path),
            mime="application/json"
        )

with col2:
    st.markdown("### 受講者入力")
    user_text = st.text_area(
        "ここに回答・原稿・授業案などを貼り付けてください",
        height=250,
        placeholder="面接回答／討論の立場表明／模擬授業案／志望理由／小論文 など"
    )

    run = st.button("AI訓練を実行")
    if run:
        if not api_key:
            st.error("OPENAI_API_KEY がありません。環境変数を設定してください。")
        elif not user_text.strip():
            st.error("受講者入力が空です。")
        else:
            client = OpenAI(api_key=api_key)
            with st.spinner("生成中..."):
                ai_text = call_llm(client, module, theme, user_text)

            st.markdown("### AIフィードバック（そのまま返却可）")
            st.write(ai_text)

            # 保存
            path = save_submission(
                module=module,
                user_text=user_text,
                ai_text=ai_text,
                meta={"rubric_common": RUBRIC_COMMON}
            )
            st.success(f"保存しました: {path}")

            # ★保存後に左カラムの履歴も即更新させる
            st.rerun()

st.markdown("---")
st.caption("運用Tips: 受講者に「再提出」を必須化し、同一課題を2回以上回すと人物力（再現性）が急速に上がります。")
