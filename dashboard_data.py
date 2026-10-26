"""Testable data contract for the submission dashboard."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Mapping

PRIORITY_ORDER = {"정기": 0, "관찰": 1, "높음": 2, "최우선": 3}


def _read_json(source: Path | BinaryIO) -> Any:
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig")
        return json.loads(raw)
    return json.loads(Path(source).read_text(encoding="utf-8-sig"))


def load_agent_payload(source: Path | BinaryIO) -> dict[str, Any]:
    payload = _read_json(source)
    if not isinstance(payload, dict):
        raise ValueError("Agent result must be a JSON object")
    reports = payload.get("representative_reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("Agent result has no representative_reports")
    for index, report in enumerate(reports, 1):
        if not isinstance(report, dict):
            raise ValueError(f"report {index} is not an object")
        required = {"region_name", "period", "activity_type", "priority_level"}
        missing = sorted(field for field in required if not report.get(field))
        if missing:
            raise ValueError(f"report {index} missing fields: {missing}")
    return payload


def load_evaluation(source: Path | BinaryIO | None) -> dict[str, Any] | None:
    if source is None:
        return None
    value = _read_json(source)
    if not isinstance(value, dict):
        raise ValueError("evaluation must be a JSON object")
    return value


def public_case_id(report: Mapping[str, Any], index: int) -> str:
    """Return a presentation-safe ID; GRID_ID is never exposed."""
    return str(report.get("case_id") or f"CASE_{index + 1}")


def case_label(report: Mapping[str, Any], index: int) -> str:
    return " · ".join((public_case_id(report, index), str(report["region_name"]), str(report["activity_type"])))


def filter_reports(reports: list[dict[str, Any]], *, regions: set[str] | None = None,
                   priorities: set[str] | None = None) -> list[dict[str, Any]]:
    return [row for row in reports
            if (not regions or str(row.get("region_name")) in regions)
            and (not priorities or str(row.get("priority_level")) in priorities)]


def dashboard_summary(payload: Mapping[str, Any], evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    reports = list(payload.get("representative_reports", []))
    recommendations = [service for report in reports for service in report.get("recommended_services", [])]
    return {
        "record_count": int(payload.get("record_count") or 0),
        "report_count": len(reports),
        "recommendation_count": len(recommendations),
        "held_count": sum(row.get("decision_status") == "hold_for_human_review" for row in reports),
        "regions": Counter(str(row.get("region_name", "미상")) for row in reports),
        "passed": None if evaluation is None else bool(evaluation.get("passed")),
        "official_source_coverage": None if evaluation is None else evaluation.get("official_source_coverage"),
        "policy_evidence_coverage": None if evaluation is None else evaluation.get("policy_evidence_coverage"),
    }


def evidence_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{
        "정책 ID": ", ".join(map(str, item.get("service_ids", []))),
        "문서": str(item.get("title", "공식 근거")),
        "근거 구간": str(item.get("section_type", "evidence")),
        "공식 URL": str(item.get("official_url", "")),
        "인용 ID": f"{item.get('document_id', '')} / {item.get('chunk_id', '')}",
    } for item in report.get("retrieved_evidence", [])]


def _number(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def answer_question(question: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Answer only supported questions from verified Agent output."""
    text = re.sub(r"\s+", " ", question).strip()
    reports = list(payload.get("representative_reports", []))
    tools = ["representative_report_lookup"]
    cautions = ["생활권 단위 정책 검토 결과이며 개인의 고립·복지대상 여부를 판정하지 않습니다."]
    regions = [region for region in sorted({str(row.get("region_name", "")) for row in reports})
               if region and (region in text or region.replace("특별자치도 ", "") in text)]
    selected = filter_reports(reports, regions=set(regions) if regions else None)

    if not text:
        answer, evidence = "질문을 입력해 주세요.", []
    elif any(token in text for token in ("몇", "개수", "몇 건", "건수")):
        tools.append("count_reports")
        answer = f"현재 승인된 대표 보고서 기준으로 조건에 맞는 사례는 {len(selected)}건입니다."
        evidence = [f"대표 보고서 {len(reports)}건 중 필터 결과 {len(selected)}건"]
    elif any(token in text for token in ("비교", "차이")) or (
        "강남" in text and "춘천" in text
    ):
        tools.append("compare_regions")
        counts = Counter(str(row.get("region_name")) for row in (selected or reports))
        detail = ", ".join(f"{name} {count}건" for name, count in sorted(counts.items()))
        answer = f"대표 사례 구성은 {detail}입니다. 도시 전체의 위험도 차이가 아닌 승인 사례 구성 비교입니다."
        evidence = [detail]
    elif any(token in text for token in ("정책", "서비스", "추천")):
        tools.append("list_rule_selected_services")
        services = {str(s.get("service_id")): str(s.get("service_name", s.get("service_id")))
                    for row in selected for s in row.get("recommended_services", []) if s.get("service_id")}
        rendered = ", ".join(f"{sid} {name}" for sid, name in sorted(services.items()))
        answer = f"규칙 기반 검색이 선택한 정책은 {rendered or '없음'}입니다. 신청 전 대상·운영 여부를 담당자가 확인해야 합니다."
        evidence = [f"정책 DB 안의 service_id {len(services)}개"]
    elif any(token in text for token in ("보류", "불확실", "검토")):
        tools.append("inspect_review_status")
        held = sum(row.get("decision_status") == "hold_for_human_review" for row in selected)
        answer = f"선택 범위에서 담당자 검토 보류 사례는 {held}건입니다. 품질·불확실성 신호가 있으면 자동 개입하지 않습니다."
        evidence = [f"hold_for_human_review={held}"]
    elif any(token in text for token in ("접근", "시설", "거리")):
        tools.append("inspect_accessibility_context")
        distances = [number for row in selected
                     for key, value in row.get("service_access_vulnerability", {}).get("values", {}).items()
                     if "distance" in key and (number := _number(value)) is not None]
        answer = (f"확인 가능한 최근접 시설 직선거리는 {min(distances):,.0f}m~{max(distances):,.0f}m입니다."
                  if distances else "현재 대표 결과에서 수치형 시설 거리를 확인할 수 없어 추정하지 않습니다.")
        evidence = [f"직선거리 관측 {len(distances)}개"] if distances else []
        cautions.append("직선거리는 이동시간이나 실제 서비스 이용 가능성을 의미하지 않습니다.")
    elif any(token in text for token in ("왜", "이유", "근거")):
        tools.append("explain_case_evidence")
        top = max(selected or reports, key=lambda row: PRIORITY_ORDER.get(str(row.get("priority_level")), -1))
        answer = (f"{top.get('region_name')}의 {top.get('activity_type')} 사례는 정책 우선순위가 "
                  f"'{top.get('priority_level')}'입니다. 활동유형·우선순위·품질 신호를 함께 검토합니다.")
        evidence = [str(value) for value in top.get("interpretations", [])[:3]]
    else:
        answer = ("대표 사례의 개수·도시 비교·추천 정책·보류 상태·시설 접근성·선정 근거 질문을 지원합니다. "
                  "미래 예측이나 개인 위험 판정은 지원하지 않습니다.")
        evidence = []
    return {"answer": answer, "generation_mode": "rule_query", "tools_used": tools,
            "filters": {"regions": regions}, "evidence": evidence, "cautions": cautions}
