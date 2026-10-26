"""Interactive submission dashboard for the BigContest policy Agent."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dashboard_data import (  # noqa: E402
    answer_question, case_label, dashboard_summary, evidence_rows,
    filter_reports, load_agent_payload, load_evaluation, public_case_id,
)

DEFAULT_RESULT = ROOT / "outputs/local_agent/final_policy_agent_3cases.json"
DEFAULT_EVAL = ROOT / "outputs/local_agent/final_policy_agent_3cases_eval.json"
DEMO_RESULT = ROOT / "demo/final_policy_agent_demo.json"
DEMO_EVAL = ROOT / "demo/final_policy_agent_demo_eval.json"

st.set_page_config(page_title="이음ON | 정책 검토 Agent", page_icon="◉", layout="wide")
st.markdown("""
<style>
.stApp{background:#f6f8fb}[data-testid="stHeader"]{background:rgba(246,248,251,.9)}
[data-testid="stSidebar"]{border-right:1px solid #e4e9f0}.block-container{max-width:1440px;padding-top:1.4rem}
.hero{padding:1.55rem 1.7rem;border-radius:20px;background:linear-gradient(125deg,#0b2447,#19376d 58%,#2f6b72);color:#fff;box-shadow:0 14px 35px rgba(11,36,71,.16);margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem;letter-spacing:-.035em}.hero p{margin:.45rem 0 0;color:#dce9f4}
.eyebrow{color:#7fd1c8;font-size:.76rem;font-weight:700;letter-spacing:.12em}
.badge{display:inline-block;padding:.32rem .68rem;border-radius:999px;font-size:.78rem;font-weight:700}
.ok{color:#136f63;background:#dff7f1}.hold{color:#985a00;background:#fff0d2}.neutral{color:#385170;background:#e8eef6}
div[data-testid="stMetric"]{background:white;border:1px solid #e4e9f0;border-radius:16px;padding:.8rem 1rem;box-shadow:0 5px 16px rgba(20,45,75,.045)}
div[data-testid="stExpander"]{background:white;border-radius:14px}.footer{color:#5c6b7a;font-size:.82rem;line-height:1.55}
</style>""", unsafe_allow_html=True)


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


with st.sidebar:
    st.markdown("### 이음ON")
    st.caption("생활권 활동변화 기반 정책 검토 Agent")
    st.divider()
    result_upload = st.file_uploader("Agent 결과 JSON", type=["json"])
    eval_upload = st.file_uploader("평가 JSON (선택)", type=["json"])
    st.caption("업로드 파일은 현재 세션에서만 사용됩니다. 미업로드 시 공개 데모 데이터를 사용합니다.")

try:
    configured_result = Path(os.getenv("POLICY_AGENT_RESULT", str(DEFAULT_RESULT)))
    result_source = result_upload or (configured_result if configured_result.exists() else DEMO_RESULT)
    demo_mode = not result_upload and Path(result_source) == DEMO_RESULT
    payload = load_agent_payload(result_source)
    eval_source = eval_upload or (DEMO_EVAL if demo_mode else (DEFAULT_EVAL if DEFAULT_EVAL.exists() else None))
    evaluation = load_evaluation(eval_source)
    load_error = None
except (OSError, ValueError, json.JSONDecodeError) as exc:
    payload, evaluation, load_error = None, None, str(exc)

st.markdown("""
<div class="hero"><div class="eyebrow">BIGCONTEST 2026 · HUMAN-IN-THE-LOOP POLICY COPILOT</div>
<h1>보이지 않던 지역의 변화를, 놓치지 않는 복지 연결로</h1>
<p>통신 활동변화 · 카드 소비맥락 · 서비스 접근성 · 공식 정책 근거를 한 화면에서 검토합니다.</p></div>
""", unsafe_allow_html=True)

if payload is None:
    st.warning("최종 Agent 결과를 불러오지 못했습니다.")
    st.code("python src\\agent\\run_final_policy_agent.py `\n  --input outputs\\tables\\policy_agent_input_final_250m.csv `\n  --policy-as-of 2026-10-25 `\n  --expected-sha256 <승인된 SHA-256> `\n  --output outputs\\local_agent\\final_policy_agent_3cases.json", language="powershell")
    st.info("왼쪽에서 결과 JSON을 업로드하거나 POLICY_AGENT_RESULT 환경변수로 경로를 지정하세요.")
    st.caption(f"확인 정보: {load_error}")
    st.stop()

reports = payload["representative_reports"]
summary = dashboard_summary(payload, evaluation)
with st.sidebar:
    st.markdown(
        '<span class="badge neutral">공개 데모 · 비식별 집계 예시</span>' if demo_mode
        else '<span class="badge ok">검증 실행 결과</span>',
        unsafe_allow_html=True,
    )
    st.caption("원자료·개인정보·격자 식별자·좌표는 공개하지 않습니다." if demo_mode else "연결된 로컬/업로드 결과를 표시합니다.")
    st.divider(); st.markdown("**탐색 필터**")
    regions = sorted({str(row["region_name"]) for row in reports})
    priorities = sorted({str(row["priority_level"]) for row in reports}, key=lambda v:{"정기":0,"관찰":1,"높음":2,"최우선":3}.get(v,9))
    selected_regions = set(st.multiselect("지역", regions, default=regions))
    selected_priorities = set(st.multiselect("정책 우선순위", priorities, default=priorities))
    filtered = filter_reports(reports, regions=selected_regions, priorities=selected_priorities)
    st.divider(); st.markdown("**실행 상태**")
    st.markdown('<span class="badge ok">검증 결과 통과</span>' if evaluation and evaluation.get("passed") else '<span class="badge neutral">평가 파일 미연결</span>', unsafe_allow_html=True)
    st.caption(f"생성 모드 · {payload.get('pipeline',{}).get('mode','offline_submission_default')}")
    st.caption(f"정책 기준일 · {payload.get('policy_as_of','—')}")

overview, cases, evidence, ask = st.tabs(["전체 현황", "대표 사례", "정책 근거", "분석 결과에 질문"])

with overview:
    if demo_mode:
        st.info("공개 데모는 실제 파이프라인의 출력 계약을 재현한 비식별 대표 사례입니다. 30,466행 원자료와 격자 식별자는 서버에 포함하지 않습니다.")
    cols = st.columns(5)
    cols[0].metric("분석 행", f"{summary['record_count']:,}", border=True)
    cols[1].metric("대표 사례", f"{summary['report_count']}건", border=True)
    cols[2].metric("추천 정책", f"{summary['recommendation_count']}건", border=True)
    cols[3].metric("검토 보류", f"{summary['held_count']}건", border=True)
    cols[4].metric("공식 출처 포함률", pct(summary["official_source_coverage"]), border=True)
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### 대표 사례 구성")
        rows = [{"지역":r["region_name"],"활동유형":r["activity_type"],"사례":1} for r in filtered]
        if rows:
            chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=6).encode(
                x=alt.X("sum(사례):Q", title="사례 수", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("활동유형:N", title=None, sort="-x"),
                color=alt.Color("지역:N", scale=alt.Scale(range=["#2f6b72","#5f7db5"])),
                tooltip=["지역:N","활동유형:N","sum(사례):Q"]).properties(height=285)
            st.altair_chart(chart, width="stretch")
        else: st.info("현재 필터에 맞는 사례가 없습니다.")
    with right:
        st.markdown("#### 제출 검증 게이트")
        checks = [
            ("타 지역 정책 0건", evaluation and not evaluation.get("wrong_region_recommendation_count")),
            ("비활성 정책 0건", evaluation and not evaluation.get("inactive_policy_recommendation_count")),
            ("DB 밖 서비스 0건", evaluation and not evaluation.get("unknown_policy_service_count")),
            ("정책별 공식 근거 100%", evaluation and evaluation.get("policy_evidence_coverage") == 1),
            ("정책별 인용 100%", evaluation and evaluation.get("policy_citation_coverage") == 1),
            ("사람 승인 요구 100%", evaluation and evaluation.get("human_approval_coverage") == 1),
        ]
        for label, passed in checks: st.markdown(f"{'✅' if passed else '◻️'} **{label}**")
        st.caption("평가 JSON 연결 시 체크가 확정됩니다.")
    st.markdown("#### 해석 원칙")
    notes = st.columns(3)
    notes[0].info("GMM 유형은 생활권 활동변화 유형이며 개인의 고립 진단이 아닙니다.")
    notes[1].info("카드·구조지표는 지역 공유 맥락이며 격자 고유 위험값이 아닙니다.")
    notes[2].info("정책 추천은 검토 후보이며 자격과 운영 여부를 자동 확정하지 않습니다.")

with cases:
    if not filtered: st.info("왼쪽 필터를 조정하세요.")
    else:
        labels = [case_label(row, reports.index(row)) for row in filtered]
        chosen = st.selectbox("대표 사례 선택", labels)
        report = filtered[labels.index(chosen)]
        safe_id = public_case_id(report, reports.index(report))
        title, badge = st.columns([4,1])
        title.markdown(f"### {safe_id} · {report['region_name']}")
        title.caption(str(report.get("case_role") or "승인된 대표 정책 검토 사례"))
        held = report.get("decision_status") == "hold_for_human_review"
        badge.markdown('<span class="badge hold">담당자 검토 보류</span>' if held else '<span class="badge ok">승인 검토 가능</span>', unsafe_allow_html=True)
        metrics = st.columns(4)
        metrics[0].metric("관측기간", report.get("period","—"), border=True)
        metrics[1].metric("활동유형", report.get("activity_type","—"), border=True)
        metrics[2].metric("정책 우선순위", report.get("priority_level","—"), border=True)
        metrics[3].metric("최대 소속확률", pct(report.get("membership_probability")), border=True)
        context_col, policy_col = st.columns([1,1.1])
        with context_col:
            st.markdown("#### 분석·외부 맥락")
            with st.expander("활동변화 해석", expanded=True):
                for item in report.get("interpretations",[]): st.write(f"- {item}")
            with st.expander("카드 소비맥락", expanded=True):
                card = report.get("card_context") or {}
                st.write(card.get("CARD_CONSUMPTION_STATE","연결된 소비국면이 없습니다."))
                if card.get("CARD_CONTEXT_REVIEW_REASON"): st.caption(card["CARD_CONTEXT_REVIEW_REASON"])
                st.caption("동일 도시·월에 공유되는 보조 맥락")
            with st.expander("구조취약성·서비스 접근성"):
                structural = report.get("structural_vulnerability",{})
                access = report.get("service_access_vulnerability",{})
                st.caption(structural.get("interpretation","지역 공유 설명값"))
                if structural.get("values"): st.json(structural["values"], expanded=False)
                st.caption(access.get("interpretation","직선거리 보조지표"))
                if access.get("values"): st.json(access["values"], expanded=False)
        with policy_col:
            st.markdown("#### 추천 정책")
            services = report.get("recommended_services",[])
            if not services: st.warning("규칙 기반 필터를 통과한 추천 정책이 없습니다.")
            for service in services:
                name = service.get("service_name", service.get("service_id","정책"))
                with st.expander(f"{name} · {service.get('service_id','')}", expanded=True):
                    st.write(service.get("support_purpose") or service.get("reason") or "담당자 검토 필요")
                    for check in service.get("conditions_to_check",[]): st.write(f"- {check}")
                    if service.get("official_source_url"): st.link_button("공식 안내 열기", service["official_source_url"], width="stretch")
            for warning in report.get("data_quality_warnings",[]): st.warning(warning)
        explanation = report.get("generated_explanation",{})
        st.markdown("#### 담당자용 설명")
        st.write(explanation.get("summary") or "규칙 기반 결과와 공식 근거를 확인해 주세요.")
        st.caption(f"생성 모드 · {explanation.get('generation_mode','rule_fallback')}")

with evidence:
    st.markdown("### 추천 정책별 공식 근거")
    st.caption("규칙 선택 정책과 연결되고 검수 게이트를 통과한 근거만 표시합니다.")
    rows = [{"사례":public_case_id(report,reports.index(report)), **row}
            for report in filtered for row in evidence_rows(report)]
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame, width="stretch", hide_index=True,
                     column_config={"공식 URL":st.column_config.LinkColumn("공식 URL")})
        st.download_button("근거 목록 CSV 다운로드", frame.to_csv(index=False).encode("utf-8-sig"),
                           file_name="policy_evidence_export.csv", mime="text/csv")
    else: st.info("현재 필터에 표시할 검색 근거가 없습니다.")
    rag = payload.get("rag",{}); stats = st.columns(4)
    stats[0].metric("로드 문서",rag.get("document_count","—"),border=True)
    stats[1].metric("검색 가능 문서",rag.get("retrievable_document_count","—"),border=True)
    stats[2].metric("승인된 추가 문서",rag.get("retrievable_additional_document_count","—"),border=True)
    stats[3].metric("서비스 연결",rag.get("linked_service_count","—"),border=True)

with ask:
    st.markdown("### 분석 결과에 질문하기")
    st.caption("현재는 검증된 결과를 계산하는 규칙 질의응답이며, 로컬 LLM은 같은 응답 계약에 연결됩니다.")
    examples = ["춘천 대표 사례는 몇 건이야?","강남과 춘천 사례를 비교해줘","추천 정책을 정리해줘","검토 보류 사례는 몇 건이야?"]
    buttons = st.columns(len(examples))
    for index, example in enumerate(examples):
        if buttons[index].button(example,key=f"ex{index}",width="stretch"): st.session_state.pending_question=example
    if "chat_history" not in st.session_state: st.session_state.chat_history=[]
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("meta"):
                with st.expander("답변 근거와 제한"): st.json(message["meta"],expanded=False)
    typed = st.chat_input("예: 춘천의 추천 정책과 담당자 확인사항을 알려줘")
    question = typed or st.session_state.pop("pending_question",None)
    if question:
        response = answer_question(question,payload)
        st.session_state.chat_history += [
            {"role":"user","content":question},
            {"role":"assistant","content":response["answer"],"meta":{key:response.get(key) for key in ("generation_mode","tools_used","filters","evidence","cautions")}},
        ]
        st.rerun()

st.divider()
st.markdown('<p class="footer">이음ON은 생활권 단위 정책 검토를 지원합니다. 개인을 진단하거나 복지 대상자를 자동 선정하지 않으며, 모든 추천은 담당자의 확인과 승인을 거쳐야 합니다.</p>', unsafe_allow_html=True)
