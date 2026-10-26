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


SYSTEM_PROMPT = """당신은 2026 빅콘테스트 사회적 고립 예방 프로젝트 '이음ON'의 분석 Agent다.
아래 공개 집계 컨텍스트만 사용하여 한국어로 답한다. 질문에 따라 수치 비교, GMM 유형 해석,
카드 보조맥락, 지속 관찰 후보, 정책 개입 방향을 자연스럽게 종합하라.

반드시 지킬 규칙:
1. 관측 사실, 해석, 한계를 구분한다.
2. 생활권 활동변화를 개인의 사회적 고립이나 복지 자격으로 확정하지 않는다.
3. GMM 군집번호는 위험등급이 아니다.
4. 팀이 승인한 1~5 위험등급은 아직 없으며 현재 결과는 4단계 정책 검토 우선순위다.
5. 카드값은 도시×월 공유 맥락이며 250m 격자 고유값이 아니다.
6. 제공되지 않은 수치·정책·인과관계를 만들지 않는다.
7. 정책을 언급할 때는 사람의 자격·운영 확인이 필요함을 밝힌다.
8. 답변 끝에는 '담당자 확인사항' 또는 '해석상 주의'를 짧게 덧붙인다.
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
            max_new_tokens=480,
            do_sample=True,
            temperature=0.35,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not answer:
        raise RuntimeError("Qwen이 빈 응답을 반환했습니다.")
    return answer
