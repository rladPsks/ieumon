# 이음ON

생활권 활동변화 신호를 공식 복지정책과 연결하는 Human-in-the-loop 정책 검토 AI Agent의 공개 데모입니다.

## 공개 데모 원칙

- 비식별·집계 대표 사례만 제공합니다.
- 대회 제공 통신·카드 원본은 포함하지 않습니다.
- 격자 식별자, 좌표, 개인 식별값을 공개하지 않습니다.
- 정책 추천은 담당자 검토 후보이며 개인의 고립이나 복지 자격을 자동 판정하지 않습니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 배포

Streamlit Community Cloud에서 다음과 같이 지정합니다.

- Repository: `rladPsks/ieumon`
- Branch: `main`
- Main file: `app.py`
- Secrets: 없음

> 2026 빅콘테스트 AI·데이터활용 분야 「사회적 고립 예방 및 대응 모델 개발」 공개 프로토타입
