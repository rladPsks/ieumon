"""Public aggregate analytics and local Qwen generation for the demo."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYTICS = ROOT / "analytics"


TYPE_GUIDE = {
    "고활동 안정형": "활동량이 높고 변화가 작아 예방형 정보 제공이 적합한 유형",
    "일반 활동안정형": "전반적으로 안정적이며 저강도 관찰과 선택형 지원이 적합한 유형",
    "복합 시간활동 위축형": "유동·활동시간·다양성 신호가 함께 위축되어 우선 확인이 필요한 유형",
    "활동 회복·확장형": "최근 활동 범위와 시간이 회복·확장되는 흐름을 보이는 유형",
    "점진적 활동시간 축소형": "활동시간 축소가 점진적으로 누적되어 조기 관찰이 필요한 유형",
}

ACTION_PLAYBOOK = {
    "정기": "예방 정보를 제공하고 월 단위 갱신만 유지한다.",
    "관찰": "다음 달 회복 여부를 확인하고 외부충격·품질경고를 먼저 점검한다.",
    "높음": "지속성과 시설 접근성을 함께 확인한 뒤 담당자 검토 대상을 좁힌다.",
    "최우선": "현장 확인 순서를 앞당기되 개인 판정이나 자동 개입은 하지 않는다.",
}

FEATURE_LABELS = {
    "ADJ_RELATIVE_FLOW_DECLINE": "상대 유동 감소",
    "ADJ_DAYTIME_DECLINE": "주간활동 감소",
    "ADJ_WEEKEND_DECLINE": "주말활동 감소",
    "ADJ_TIME_ENTROPY_DECLINE": "시간대 다양성 감소",
    "ADJ_EFFECTIVE_ACTIVE_HOURS_DECLINE": "유효 활동시간 감소",
}


def load_analytics() -> dict[str, Any]:
    return {
        "summary": json.loads((ANALYTICS / "model_summary.json").read_text(encoding="utf-8")),
        "gmm_original": pd.read_csv(ANALYTICS / "gmm_profile_original.csv"),
        "gmm_scaled": pd.read_csv(ANALYTICS / "gmm_profile_scaled.csv"),
        "persistence": pd.read_csv(ANALYTICS / "persistence_city_summary.csv"),
        "card": pd.read_csv(ANALYTICS / "card_region_month_context.csv", dtype={"STD_YM": str}).fillna(""),
    }


def public_llm_context(payload: Mapping[str, Any], analytics: Mapping[str, Any]) -> dict[str, Any]:
    """Return only aggregate, presentation-safe information for external generation."""
    profiles = analytics["gmm_original"].round(5).to_dict(orient="records")
    scaled = analytics["gmm_scaled"].round(4).to_dict(orient="records")
    persistence = analytics["persistence"].round(4).to_dict(orient="records")
    card = analytics["card"].to_dict(orient="records")
    reports = []
    for report in payload.get("representative_reports", []):
        reports.append({
            "case_id": report.get("case_id"),
            "region_name": report.get("region_name"),
            "period": report.get("period"),
            "activity_type": report.get("activity_type"),
            "priority_level": report.get("priority_level"),
            "decision_status": report.get("decision_status"),
            "interpretations": report.get("interpretations", []),
            "recommended_services": [
                {key: service.get(key) for key in (
                    "service_id", "service_name", "support_purpose",
                    "conditions_to_check", "official_source_url"
                )}
                for service in report.get("recommended_services", [])
            ],
        })
    return {
        "model_summary": analytics["summary"],
        "gmm_cluster_profiles": profiles,
        "gmm_standardized_profiles": scaled,
        "type_interpretations": TYPE_GUIDE,
        "persistent_attention_by_city": persistence,
        "card_city_month_context": card,
        "representative_policy_cases": reports,
    }


def focused_agent_context(
    question: str,
    payload: Mapping[str, Any],
    analytics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact, question-specific evidence package for the small local model."""
    text = question.replace(" ", "")
    summary = analytics["summary"]
    persistence = analytics["persistence"].round(4).to_dict(orient="records")
    context: dict[str, Any] = {
        "question_type": "general_analysis",
        "project_scope": {
            "analysis_rows": summary["analysis_rows"],
            "unique_grids": summary["unique_grids"],
            "period": summary["period"],
            "analysis_unit": summary["analysis_unit"],
            "approved_output": "5개 GMM 활동유형 + 4단계 정책 검토 우선순위",
            "not_available": "승인된 개인 위험확률, 1~5 위험등급, 도시별 GMM 유형 비중",
        },
        "persistent_attention_by_city": persistence,
        "action_playbook": ACTION_PLAYBOOK,
        "interpretation_contract": [
            "후보 수는 도시 위험도나 개인 고립자 수가 아니다.",
            "도시 규모가 다르므로 절대 수와 비율을 함께 본다.",
            "대표 사례 구성은 도시 전체 특성으로 일반화하지 않는다.",
            "현장 확인과 정책 운영·자격 검토가 최종 단계다.",
        ],
    }
    city_comparison = "강남" in text and "춘천" in text
    if city_comparison:
        context["question_type"] = "city_comparison"
        context["required_comparison"] = {
            "춘천": "지속관찰 후보 370개, 전체 생활권 대비 5.61%, 관측 가능 생활권 대비 6.24%",
            "강남": "지속관찰 후보 59개, 전체 및 관측 가능 생활권 대비 8.58%",
            "correct_conclusion": (
                "춘천은 절대 후보 수가 많고 강남은 생활권 대비 비율이 높다. "
                "따라서 춘천은 넓은 후보군의 검토 순서화, 강남은 비율이 높은 소수 후보의 집중 확인이 필요하다."
            ),
            "forbidden_inference": "도시별 GMM 유형 비중 자료가 없으므로 어느 도시가 특정 유형이라고 말하지 않는다.",
        }

    if any(token in text for token in ("GMM", "유형", "위축", "회복", "활동")):
        context["gmm_profiles"] = analytics["gmm_original"].round(4).to_dict(orient="records")
        context["gmm_standardized_profiles"] = analytics["gmm_scaled"].round(3).to_dict(orient="records")
        context["type_interpretations"] = TYPE_GUIDE

    if any(token in text for token in ("카드", "소비", "품질", "외부충격")) or city_comparison:
        card = analytics["card"].copy()
        card_summary = []
        for city, rows in card.groupby("CITY_PROXY"):
            flagged = rows.loc[rows["CARD_CONTEXT_REVIEW_REQUIRED"].eq(1), "STD_YM"].tolist()
            card_summary.append({
                "city": city,
                "observed_months": len(rows),
                "review_required_months": flagged,
                "state_counts": rows["CARD_CONSUMPTION_STATE"].value_counts().to_dict(),
                "rule": "도시×월 공유 맥락이며 격자 고유 원인이나 위험점수가 아니다.",
            })
        context["card_city_summary"] = card_summary

    if any(token in text for token in ("정책", "서비스", "개입", "대응", "사례")):
        context["representative_policy_cases"] = public_llm_context(
            payload, analytics
        )["representative_policy_cases"]
    return context


def grounded_action_fallback(question: str, context: Mapping[str, Any]) -> str:
    """Return an actionable verified answer when local generation is unusable."""
    text = question.replace(" ", "")
    if "강남" in text and "춘천" in text:
        return """**핵심 결론**  
춘천은 지속관찰 후보의 절대 수가 많고, 강남은 전체 생활권 대비 후보 비율이 더 높습니다. 같은 기준으로 보되 현장 검토 전략은 달라야 합니다.

**확인된 수치**
- 춘천: 370개, 전체 생활권의 5.61%(관측 가능 생활권의 6.24%)
- 강남: 59개, 전체 생활권의 8.58%
- 두 지역 합계: 429개 지속관찰 후보

**그래서 무엇을 해야 하나**
- 춘천은 후보가 많으므로 지속기간·정책 우선순위·서비스 접근성을 이용해 현장 확인 순서를 먼저 좁혀야 합니다.
- 강남은 후보 수는 적지만 비율이 높으므로 해당 생활권의 반복 신호와 카드 품질경고를 집중 확인해야 합니다.
- 두 지역 모두 공식 정책의 운영 여부와 대상 조건을 확인한 뒤 담당자가 최종 연결 여부를 결정해야 합니다.

**해석상 주의**  
이는 도시 위험도나 개인 고립자 수 비교가 아닙니다. 현재 자료에는 도시별 GMM 유형 비중이 없어 특정 도시가 어느 활동유형에 더 취약하다고 단정할 수 없습니다."""
    return """**핵심 결론**  
현재 결과는 개인을 판정하는 위험점수가 아니라, 담당자가 먼저 확인할 생활권과 대응 방향을 좁히는 근거입니다.

**담당자 실행안**
- 활동유형과 4단계 정책 검토 우선순위를 함께 확인합니다.
- 카드 품질경고와 외부충격 가능성을 먼저 제외합니다.
- 지속성이 확인된 경우 서비스 접근성과 공식 정책 운영·자격 조건을 검토합니다.
- 최종 개입 여부는 현장 확인과 사람 승인을 거쳐 결정합니다.

**해석상 주의**  
GMM 군집번호는 위험등급이 아니며 생활권 결과를 개인의 사회적 고립으로 해석할 수 없습니다."""


def answer_is_usable(answer: str, question: str) -> bool:
    """Reject repetitive or ungrounded small-model answers."""
    compact = answer.replace(" ", "")
    if len(answer.strip()) < 80:
        return False
    repeated_phrases = ("활동시간축소", "활동시간확장", "동일이아닌")
    if any(compact.count(phrase) >= 3 for phrase in repeated_phrases):
        return False
    q = question.replace(" ", "")
    if "강남" in q and "춘천" in q:
        required = ("370", "59", "5.61", "8.58")
        if not all(value in answer for value in required):
            return False
    return True


SYSTEM_PROMPT = """당신은 지자체 복지 담당자의 다음 행동을 설계하는 '이음ON 정책 코파일럿'이다.
목표는 분석을 다시 읽어주는 것이 아니라, 제공된 증거만으로 무엇을 확인하고 어떤 순서로 대응할지 제안하는 것이다.

반드시 지킬 규칙:
1. 첫 문장에 질문에 대한 결론을 바로 쓴다.
2. 반드시 '핵심 결론 / 확인된 수치 / 그래서 무엇을 해야 하나 / 해석상 주의' 순서로 답한다.
3. 실행안은 담당자가 실제로 할 수 있는 동사(확인, 선별, 연결, 재평가)로 2~4개 제시한다.
4. 제공된 Evidence Package에 없는 도시별 유형, 수치, 원인, 정책을 만들지 않는다.
5. 도시별 GMM 유형 비중이 없으면 어느 도시가 특정 유형에 취약하다고 말하지 않는다.
6. 같은 문장이나 의미를 반복하지 않는다. 답변은 한국어 700자 이내로 쓴다.
7. 관측 사실, 해석, 실행 제안, 한계를 구분한다.
8. 생활권 활동변화를 개인의 사회적 고립이나 복지 자격으로 확정하지 않는다.
9. GMM 군집번호는 위험등급이 아니다.
10. 팀이 승인한 1~5 위험등급은 아직 없으며 현재 결과는 4단계 정책 검토 우선순위다.
11. 카드값은 도시×월 공유 맥락이며 250m 격자 고유값이 아니다.
12. 정책을 언급할 때는 사람의 자격·운영 확인이 필요함을 밝힌다.
"""


@lru_cache(maxsize=1)
def load_local_qwen(model_id: str):
    """Load a small local Qwen model once per Streamlit worker."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "로컬 Qwen 실행 패키지가 설치되지 않았습니다. requirements.txt를 다시 배포해 주세요."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model, torch


def generate_with_qwen(
    question: str,
    context: Mapping[str, Any],
    *,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    history: list[dict[str, str]] | None = None,
) -> str:
    recent = (history or [])[-6:]
    tokenizer, model, torch = load_local_qwen(model_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        {"role": row["role"], "content": row["content"]}
        for row in recent
        if row.get("role") in {"user", "assistant"}
    )
    messages.append({
        "role": "user",
        "content": (
            "[공개 집계 컨텍스트]\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            + "\n\n[질문]\n"
            + question
        ),
    })
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=12000)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=360,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not answer:
        raise RuntimeError("Qwen이 빈 응답을 반환했습니다.")
    return answer
