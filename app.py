"""이음ON: public aggregate analysis and policy-response demo."""
from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from analysis_data import (
    FEATURE_LABELS,
    TYPE_GUIDE,
    generate_with_qwen,
    load_analytics,
    public_llm_context,
)
from dashboard_data import (
    answer_question,
    dashboard_summary,
    evidence_rows,
    load_agent_payload,
    load_evaluation,
    public_case_id,
)


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "demo/final_policy_agent_demo.json"
EVALUATION = ROOT / "demo/final_policy_agent_demo_eval.json"

st.set_page_config(page_title="이음ON | 사회적 고립 예방 분석 Agent", page_icon="◉", layout="wide")
st.markdown("""
<style>
.stApp{background:#fff9e9;color:#111}[data-testid="stHeader"]{background:rgba(255,249,233,.94)}
.block-container{max-width:1480px;padding-top:1.25rem;padding-bottom:2.5rem}
[data-testid="stSidebar"]{border-right:2px solid #ffb23e;background:#fffdf7}
.hero{padding:1.8rem 2rem;border:3px solid #ff9f43;border-radius:22px;background:linear-gradient(115deg,#ffd849,#ffb347 55%,#ff884d);color:#151515;box-shadow:7px 7px 0 #f36f4a;margin-bottom:1rem}
.hero h1{font-size:2.15rem;margin:.18rem 0 .45rem;letter-spacing:-.045em}.hero p{color:#29200f;margin:0;line-height:1.65;font-weight:600}
.eyebrow{color:#7b3d00;font-weight:900;font-size:.76rem;letter-spacing:.12em}
.pill{display:inline-block;border:1px solid #ed9b2e;border-radius:999px;padding:.32rem .7rem;font-size:.76rem;font-weight:800;margin:.15rem .2rem .15rem 0}
.pill-blue{color:#1a4e88;background:#dcecff}.pill-green{color:#245c3e;background:#dff3df}.pill-amber{color:#6d3b00;background:#ffe6a9}.pill-gray{color:#343434;background:#f1eadc}
.flow{background:white;border:2px solid #f4a33c;border-radius:17px;padding:1rem 1.05rem;min-height:132px;box-shadow:4px 4px 0 #ffd267}
.flow-num{font-size:.72rem;font-weight:900;color:#e86635;letter-spacing:.08em}.flow-title{font-size:1.02rem;font-weight:850;margin:.28rem 0}.flow-text{font-size:.86rem;color:#4d4538;line-height:1.5}
.callout{border:2px solid #ffad3d;border-left:8px solid #ff754c;background:white;border-radius:12px;padding:.85rem 1rem;margin:.5rem 0}
div[data-testid="stMetric"]{background:white;border:2px solid #f0aa45;border-radius:16px;padding:.78rem 1rem;box-shadow:4px 4px 0 #ffe08a}
div[data-testid="stExpander"]{background:white;border:1px solid #f1b657;border-radius:14px}
button[kind="secondary"]{border:1.5px solid #f0a03b!important;background:#fff!important;color:#171717!important}
button[kind="secondary"]:hover{border-color:#f36f4a!important;background:#fff3cf!important}
[data-baseweb="tab-list"]{gap:.25rem;border-bottom:2px solid #f7c35e}
[data-baseweb="tab"]{font-weight:750;color:#3a3124}
[aria-selected="true"]{color:#d9572b!important;border-bottom-color:#f36f4a!important}
.small{font-size:.82rem;color:#645847;line-height:1.55}.footer{font-size:.8rem;color:#695b46;line-height:1.6}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all():
    return load_agent_payload(RESULT), load_evaluation(EVALUATION), load_analytics()


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


payload, evaluation, analytics = load_all()
summary = dashboard_summary(payload, evaluation)
model = analytics["summary"]
gmm = analytics["gmm_original"]
scaled = analytics["gmm_scaled"]
persistence = analytics["persistence"]
card = analytics["card"]

st.markdown("""
<div class="hero">
    <div class="eyebrow">BIGCONTEST 2026 · 생활권 활동변화 기반 복지정책 검토 AGENT</div>
  <h1>보이지 않던 생활권의 변화를, 놓치지 않는 연결로</h1>
  <p>30,466개 생활권×월 활동변화에서 5개 유형을 발견하고, 지속성과 지역 맥락을 더해<br>
  담당자가 확인할 정책 개입 후보를 제시하는 Human-in-the-loop 분석 Agent</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 이음ON")
    st.caption("활동변화 탐지 → 유형 해석 → 정책 대응")
    st.markdown('<span class="pill pill-green">PUBLIC AGGREGATE DEMO</span>', unsafe_allow_html=True)
    st.info("대회 원자료·격자 식별자·좌표·개인정보는 이 서버에 포함하지 않습니다.")
    st.markdown("#### 분석 범위")
    st.write("2025년 8월–12월")
    st.write("강원 춘천시 · 서울 강남구")
    st.write("250m 생활권 × 월")
    st.markdown("#### 해석 계약")
    st.caption("활동변화 유형 ≠ 개인 고립 진단")
    st.caption("군집번호 ≠ 위험등급")
    st.caption("카드 맥락 = 도시×월 공유값")
    st.caption("최종 개입 = 담당자 승인 필요")

tabs = st.tabs([
    "프로젝트 개요", "GMM 활동유형", "카드·지역 맥락",
    "지속 관찰", "대표 사례·정책", "AI 분석 Agent",
])

with tabs[0]:
    st.markdown("### 데이터에서 대응까지")
    metrics = st.columns(5)
    metrics[0].metric("분석 관측", f"{model['analysis_rows']:,}행", border=True)
    metrics[1].metric("고유 생활권", f"{model['unique_grids']:,}개", border=True)
    metrics[2].metric("발견 유형", f"{model['gmm_types']}개", border=True)
    metrics[3].metric("지속 관찰 후보", f"{model['persistent_attention_candidates']:,}개", border=True)
    metrics[4].metric("외부데이터 QA", "FAIL 0", border=True)
    st.caption("지속 관찰 후보는 ‘높음·최우선’ 정책 검토 우선순위가 2개월 이상 연속된 생활권이며 개인 위험확률이 아닙니다.")

    cols = st.columns(4)
    flow = [
        ("01 · DETECT", "활동변화 탐지", "통신의 유동·주간·주말·시간대 다양성 변화를 도시·월 효과와 함께 보정"),
        ("02 · DISCOVER", "GMM 유형 발견", "정답 라벨 없이 30,466개 생활권×월을 5개 해석 가능한 활동유형으로 구분"),
        ("03 · CONTEXT", "맥락·지속성 확인", "카드 소비국면, 구조취약성, 시설 접근성, 연속 관측을 원인 단정 없이 결합"),
        ("04 · ACT", "Agent 정책 대응", "공식 정책 DB와 RAG 근거로 대응 후보를 만들고 불확실 사례는 사람 검토로 보류"),
    ]
    for col, (num, title, text) in zip(cols, flow):
        col.markdown(f'<div class="flow"><div class="flow-num">{num}</div><div class="flow-title">{title}</div><div class="flow-text">{text}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### 두 층을 억지로 섞지 않는 분석")
        layer = pd.DataFrame([
            {"분석층":"인구집단·카드 맥락","단위":"도시×주 → 도시×월","역할":"소비활동 변화와 외부충격 확인"},
            {"분석층":"생활권 활동변화","단위":"250m 격자×월","역할":"어느 생활권의 활동이 위축되는지 탐지"},
            {"분석층":"정책 대응 Agent","단위":"검증된 집계 결과","역할":"유형·우선순위별 공식 복지자원 연결"},
        ])
        st.dataframe(layer, hide_index=True, width="stretch")
    with right:
        st.markdown("#### 핵심 발견")
        st.markdown("""
        <div class="callout"><b>5개 유형</b><br><span class="small">안정형 2개, 위축형 2개, 회복·확장형 1개가 관측되었습니다.</span></div>
        <div class="callout"><b>429개 지속 관찰 후보</b><br><span class="small">춘천 370개, 강남 59개 생활권에서 높은 검토 우선순위가 2개월 이상 이어졌습니다.</span></div>
        <div class="callout"><b>불확실성은 자동 개입하지 않음</b><br><span class="small">카드 품질경고 6개 도시×월과 혼합 사례를 사람 검토로 보류합니다.</span></div>
        """, unsafe_allow_html=True)

with tabs[1]:
    st.markdown("### GMM으로 발견한 5개 활동변화 유형")
    st.caption("군집 수는 BIC·안정성·최소 군집 크기·도시/시기 편중과 해석 가능성을 함께 검토해 K=5로 확정했습니다.")
    left, right = st.columns([1.05, 1])
    with left:
        distribution = alt.Chart(gmm).mark_bar(cornerRadiusEnd=7).encode(
            x=alt.X("COUNT:Q", title="생활권×월 관측 수"),
            y=alt.Y("ACTIVITY_TYPE:N", title=None, sort="-x"),
        color=alt.Color("ACTIVITY_TYPE:N", legend=None, scale=alt.Scale(range=["#ffd447", "#ffb347", "#ff875e", "#ef5b5b", "#76b7e8"])),
            tooltip=["ACTIVITY_TYPE:N", alt.Tooltip("COUNT:Q", format=","), alt.Tooltip("SHARE:Q", format=".1%")],
        ).properties(height=310)
        st.altair_chart(distribution, width="stretch")
    with right:
        selected_type = st.selectbox("유형 상세 탐색", gmm["ACTIVITY_TYPE"].tolist(), index=2)
        row = gmm.loc[gmm["ACTIVITY_TYPE"].eq(selected_type)].iloc[0]
        st.markdown(f"#### {selected_type}")
        st.write(TYPE_GUIDE[selected_type])
        m = st.columns(3)
        m[0].metric("관측 수", f"{int(row['COUNT']):,}", border=True)
        m[1].metric("전체 비중", f"{row['SHARE']:.1%}", border=True)
        m[2].metric("평균 연속감소", f"{row['MEAN_RELATIVE_DECLINE_STREAK']:.2f}", border=True)
        st.info("양수·음수 방향은 변수별 감소 정의를 따르며, 단일 값이 원인이나 개인 위험을 의미하지 않습니다.")

    st.markdown("#### 도시·월 효과를 보정한 유형별 프로파일")
    long = scaled.melt(id_vars=["CLUSTER_FINAL", "ACTIVITY_TYPE"], var_name="FEATURE", value_name="Z")
    long["지표"] = long["FEATURE"].map(FEATURE_LABELS)
    heat = alt.Chart(long).mark_rect(cornerRadius=2).encode(
        x=alt.X("지표:N", title=None, sort=list(FEATURE_LABELS.values()), axis=alt.Axis(labelAngle=-25)),
        y=alt.Y("ACTIVITY_TYPE:N", title=None, sort=gmm["ACTIVITY_TYPE"].tolist()),
        color=alt.Color("Z:Q", title="보정 평균", scale=alt.Scale(scheme="redblue", domainMid=0)),
        tooltip=["ACTIVITY_TYPE:N", "지표:N", alt.Tooltip("Z:Q", format=".2f")],
    ).properties(height=270)
    st.altair_chart(heat, width="stretch")
    st.caption("색은 표준화된 상대 차이이며 위험확률이나 인과효과가 아닙니다. Cluster 0~4는 라벨일 뿐 서열이 아닙니다.")

with tabs[2]:
    st.markdown("### 카드 소비국면은 원인이 아니라 보조 맥락")
    st.caption("카드 원천 단위는 지역×주이며, 통신 생활권에 복제하지 않고 도시×월 공통 맥락으로만 연결했습니다.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("주간 특징행", f"{model['card_feature_rows']}행", border=True)
    c2.metric("모델 포함", f"{model['card_model_rows']}행", border=True)
    c3.metric("추석 영향 제외", f"{model['card_excluded_holiday_rows']}행", border=True)
    c4.metric("검토·차단 맥락", f"{model['card_quality_blocked_contexts']}/10", border=True)
    display_card = card.copy()
    display_card["월"] = display_card["STD_YM"].str[:4] + "-" + display_card["STD_YM"].str[4:]
    display_card["품질상태"] = display_card["CARD_CONTEXT_REVIEW_REQUIRED"].map({0:"정상", 1:"검토 필요"})
    timeline = alt.Chart(display_card).mark_circle(size=340, stroke="white", strokeWidth=2).encode(
        x=alt.X("월:N", title=None), y=alt.Y("CITY_PROXY:N", title=None),
        color=alt.Color("CARD_CONSUMPTION_STATE:N", title="소비국면", scale=alt.Scale(range=["#f06d4f", "#6caee0"])),
        shape=alt.Shape("품질상태:N", title="품질상태"),
        tooltip=["CITY_PROXY:N", "월:N", "CARD_CONSUMPTION_STATE:N", "품질상태:N", alt.Tooltip("CARD_MEMBERSHIP_PROB:Q", format=".3f")],
    ).properties(height=230)
    st.altair_chart(timeline, width="stretch")
    warnings = display_card.loc[display_card["CARD_CONTEXT_REVIEW_REQUIRED"].eq(1), ["월", "CITY_PROXY", "CARD_CONSUMPTION_STATE", "CARD_CONTEXT_REVIEW_REASON"]]
    with st.expander("품질경고 6개 도시×월 보기"):
        st.dataframe(warnings, hide_index=True, width="stretch")
    st.warning("2025년 추석 영향 주는 학습과 기준선에서 제외했습니다. 카드 국면은 생활권의 위험도를 올리거나 내리는 점수로 사용하지 않습니다.")

with tabs[3]:
    st.markdown("### 한 달의 급락보다, 반복되는 신호를 본다")
    st.caption("현재 승인된 결과는 1~5 위험등급이 아니라 4단계 정책 검토 우선순위와 지속성입니다.")
    total = int(persistence["PERSISTENT_ATTENTION_GRIDS"].sum())
    a, b, c = st.columns(3)
    a.metric("전체 지속 관찰 후보", f"{total:,}개", border=True)
    b.metric("춘천", "370개 · 전체의 5.61%", border=True)
    c.metric("강남", "59개 · 전체의 8.58%", border=True)
    chart_data = persistence.melt(
        id_vars=["CITY_PROXY"],
        value_vars=["UNIQUE_GRIDS", "ELIGIBLE_GRIDS", "PERSISTENT_ATTENTION_GRIDS"],
        var_name="구분", value_name="생활권 수",
    )
    names = {"UNIQUE_GRIDS":"전체 생활권", "ELIGIBLE_GRIDS":"2개월 이상 관측 가능", "PERSISTENT_ATTENTION_GRIDS":"지속 관찰 후보"}
    chart_data["구분"] = chart_data["구분"].map(names)
    bars = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=5).encode(
        x=alt.X("생활권 수:Q", title="생활권 수", scale=alt.Scale(type="sqrt")),
        y=alt.Y("CITY_PROXY:N", title=None),
        color=alt.Color("구분:N", scale=alt.Scale(range=["#ffe7a3", "#ffb347", "#f36f4a"])),
        row=alt.Row("구분:N", header=alt.Header(labelOrient="top", labelFontWeight="bold")),
        tooltip=["CITY_PROXY:N", "구분:N", alt.Tooltip("생활권 수:Q", format=",")],
    ).properties(height=78)
    st.altair_chart(bars, width="stretch")
    st.markdown("""
    <div class="callout"><b>해석</b><br><span class="small">춘천은 절대 후보 수가 많고, 강남은 전체 생활권 대비 후보 비율이 더 높습니다. 도시의 격자 수와 관측가능 범위가 다르므로 후보 수만으로 도시 위험도를 비교할 수 없습니다.</span></div>
    """, unsafe_allow_html=True)
    priority = pd.DataFrame([
        {"단계":"정기", "의미":"현재 활동 유지", "조치":"예방형 정보·정기 갱신"},
        {"단계":"관찰", "의미":"변화 신호 관찰", "조치":"다음 월 회복 여부 확인"},
        {"단계":"높음", "의미":"복수 신호 또는 지속성", "조치":"정책 후보·현장 맥락 검토"},
        {"단계":"최우선", "의미":"강한 복합 신호", "조치":"담당자 우선 확인, 자동 개입 금지"},
    ])
    st.dataframe(priority, hide_index=True, width="stretch")

with tabs[4]:
    st.markdown("### 유형과 우선순위에 따라 달라지는 대응")
    reports = payload["representative_reports"]
    labels = [f"{public_case_id(r, i)} · {r['region_name']} · {r['activity_type']}" for i, r in enumerate(reports)]
    selected = st.selectbox("비식별 대표 사례", labels)
    report = reports[labels.index(selected)]
    held = report.get("decision_status") == "hold_for_human_review"
    st.markdown(f"#### {report['region_name']} · {report['activity_type']}")
    st.markdown('<span class="pill pill-amber">담당자 검토 보류</span>' if held else '<span class="pill pill-green">승인 검토 가능</span>', unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("관측월", report.get("period", "—"), border=True)
    m[1].metric("정책 우선순위", report.get("priority_level", "—"), border=True)
    m[2].metric("유형 소속확률", f"{float(report.get('membership_probability',0)):.1%}", border=True)
    m[3].metric("추천 후보", f"{len(report.get('recommended_services',[]))}개", border=True)
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("##### 분석 근거")
        for item in report.get("interpretations", []): st.write(f"- {item}")
        card_context = report.get("card_context", {})
        st.info(card_context.get("CARD_CONSUMPTION_STATE", "카드 맥락 없음"))
        for warning in report.get("data_quality_warnings", []): st.warning(warning)
    with right:
        st.markdown("##### 공식 정책 후보")
        for service in report.get("recommended_services", []):
            with st.expander(f"{service['service_name']} · {service['service_id']}", expanded=True):
                st.write(service.get("support_purpose", "담당자 검토 필요"))
                for check in service.get("conditions_to_check", []): st.write(f"- {check}")
                if service.get("official_source_url"): st.link_button("공식 출처 확인", service["official_source_url"], width="stretch")
    st.markdown("##### 검색된 정책 근거")
    evidence = pd.DataFrame(evidence_rows(report))
    st.dataframe(evidence, hide_index=True, width="stretch", column_config={"공식 URL": st.column_config.LinkColumn("공식 URL")})

with tabs[5]:
    st.markdown("### 공개 집계 결과에 자유롭게 질문하세요")
    qwen_model = secret("QWEN_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    st.caption("API 키 없이 배포 서버에서 실행되는 Qwen이 GMM·카드·지속관찰·공식 정책 집계 결과를 종합합니다. 최초 질문은 모델 초기화로 다소 지연될 수 있습니다.")
    examples = [
        "5개 GMM 유형의 차이를 핵심 지표 중심으로 설명해줘",
        "춘천 370개와 강남 59개를 어떻게 해석해야 해?",
        "복합 시간활동 위축형에 어떤 개입이 적절하고 왜 그래?",
        "이 분석으로 개인의 고립을 판단할 수 없는 이유를 설명해줘",
    ]
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        if cols[idx % 2].button(example, key=f"prompt_{idx}", width="stretch"):
            st.session_state["pending_question"] = example
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    question = st.chat_input("예: 위축형과 회복형을 비교하고 정책 대응 차이를 설명해줘") or st.session_state.pop("pending_question", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"): st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Qwen이 공개 집계 결과를 종합하고 있습니다..."):
                    answer = generate_with_qwen(
                        question,
                        public_llm_context(payload, analytics),
                        model_id=qwen_model,
                        history=st.session_state.messages[:-1],
                    )
            except RuntimeError as exc:
                st.error(str(exc))
                answer = answer_question(question, payload)["answer"]
                st.caption("모델 실행이 불가능해 검증된 로컬 응답을 표시했습니다.")
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.expander("Agent가 답할 때 지키는 범위"):
        st.write("- 관측 사실·해석·한계를 구분합니다.")
        st.write("- 제공되지 않은 수치나 복지서비스를 만들지 않습니다.")
        st.write("- 1~5 위험등급이 승인된 것처럼 답하지 않습니다.")
        st.write("- 개인 고립·자격을 확정하거나 자동 개입하지 않습니다.")

st.divider()
st.markdown("""
<p class="footer"><b>생성형 AI 활용 고지</b> · 로컬 Qwen 모델은 공개·집계된 분석 결과를 설명하는 데만 사용됩니다.
계산된 GMM 유형, 정책 검토 우선순위와 서비스 ID를 변경하지 않으며, 모든 정책 제안은 담당자 확인을 전제로 합니다.<br>
이음ON 공개 데모에는 대회 제공 원자료, 격자 식별자·좌표와 개인 식별정보가 포함되지 않습니다.</p>
""", unsafe_allow_html=True)
