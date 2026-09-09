# survev CPC — SLM 기반 "사람 같은" 듀오 파트너: 브레인스토밍 & 액션플랜

작성일 2026-09-08 · v0.2 (6주 agile로 개정) · 대상: 준성 개인 연구 프로젝트 (CPC)

## 0. 한 줄 요약과 오늘의 결정

목표는 오픈소스 surviv.io(survev)에서 사람과 듀오를 뛰는 AI 파트너(CPC)를 만들되, 판단과 소통은 SLM이 맡고 조준·이동 같은 모터 레벨은 별도 컨트롤러가 맡는 계층형 구조로 가는 것이다. 이는 크래프톤 PUBG Ally의 System 1(Behavior Tree, per-frame 조작) / System 2(SLM, 의도 해석·협업·발화) 분리와 같은 구조다. 3090 4대는 2B~4B SLM을 서빙·SFT·(LoRA) RL까지 돌리기에 충분하고, 진짜 병목은 GPU가 아니라 (1) 환경·인터페이스 엔지니어링, (2) 사람다움을 만들 데이터, (3) 사람다움을 재는 평가 루프다.

오늘 정한 것 세 가지: 계층형 아키텍처(SLM = 전술/소통 두뇌), 사람 데이터 없이 합성 우선(스크립트 봇 + 교사 LLM + RL, 사람 데이터는 나중에 슬롯만 마련), 그리고 기간. 처음엔 3개월로 잡았다가 **6주 agile**로 줄였다(v0.2). 6주 안에 나름의 co-playable bot을 만들고, 그 시점에 다음 단계를 평가한다. 진행 방식은 FDE식이다: 매주 플레이 가능한 빌드를 내고, 준성 본인(+지인 1~2명)이 듀오로 뛰어보고, "뭐가 이상했나" 한 줄 로그가 다음 주 백로그가 된다. 8절에 6주 스프린트 계획이 있다.

런타임 배치도 정했다(v0.2). 팀메이트(CPC 본체)의 System 1은 TS로 게임 서버 프로세스 안에서 매 틱 돈다. 학습된 모터 정책이 생기면 PyTorch → ONNX → `onnxruntime-node`로 같은 프로세스에 들어간다. System 2(SLM)는 3090의 vLLM 사이드카이고 TS가 이벤트 때 HTTP로 부른다. 적(상대 정책)·교사 데이터 생성·학습·평가는 Python이 브리지 너머에서 맡는다. 브리지는 틱 조종이 아니라 결정 단위 지휘다(8-1절 계약 참고).

솔직한 전제 하나: "SLM만으로 거의 완벽한 사람처럼"은 SLM이 초당 수십 번의 조준·이동 입력을 직접 만들어서는 달성되지 않는다. 사람다움의 절반은 모터 레벨(반응시간, 조준 오차, 머뭇거림)에서 나오고, 나머지 절반은 전술·소통 레벨(어디로 갈지, 언제 밀지, 뭐라고 말할지)에서 나온다. SLM은 후자를 맡고, 전자는 명시적 인지 제약(reaction delay, aim noise, attention, memory decay)을 넣은 컨트롤러가 맡는다. 합성 우선이므로 모터 레벨 사람다움은 처음엔 "문헌 기반 파라미터"로 만들고, 나중에 사람 데이터가 생기면 BC 정책으로 교체할 수 있게 인터페이스를 고정해 둔다.

## 1. 지금 손에 있는 것 (두 repo 점검)

evolutionary-ai-battle은 이미 "Co-Player Evaluation Harness"로 방향을 틀어 놓았다. 쓸모 있는 자산은 다음과 같다. 공통 JSON 스키마(BattleSnapshot / TacticalObservation / BattleAction / MultiAgentStep / EpisodeTrajectory)가 docs/common-interface-v0.md에 정의되어 있고 TorchRL-friendly하지만 의존성이 없다. 메트릭 벡터 평가(combat / survival / cooperation / movement — teammateResponseRate, isolationRate, avgAllyDistance, stuckSteps 등)가 스칼라 보상으로 뭉개지지 않은 채 정의되어 있는데, 이건 CPC의 "도움이 되는가"를 재는 축으로 그대로 쓴다. Python 계층형 베이스라인(build_context → create_global_plan_if_needed → select_intent → create_local_plan → controller → build_action)은 사실상 System 1/2의 뼈대다. select_intent와 global plan 자리에 SLM을 꽂고, local plan / controller / build_action은 System 1로 남기면 된다. 이산 액션 공간(move 9 bins × aim 16 bins × fire)은 toy 환경용이며, survev에서는 연속 aim으로 바꿔야 한다. legacy에 PPO/TorchRL 코드가 있어 RL 스모크 테스트 경험이 있다.

survev 쪽은 cpc_dev(PR-S1)까지 왔다. 오프라인 Game 생성, addTestPlayer(NoOpSocket)로 듀오 2v2 스폰, game.update(0.1)로 스텝, BattleSnapshot-like JSON 추출까지 된다. 남은 PR-S2~S6(솔로 시나리오, CPC 액션 → InputMsg 어댑터, 패시브 메트릭, fire/damage/death 이벤트 탭, EpisodeTrajectory JSONL export)이 정확히 이번 계획의 0단계다. 엔진 사실 몇 가지를 적어 두면, gameTps 100 / netSyncTps 33, InputMsg는 moveLeft/Right/Up/Down, shootStart/shootHold, toMouseDir/toMouseLen, inputs: Input[](Reload, Interact, Revive, Loot, Equip*, UseBandage/HealthKit/Soda/Painkiller, EmoteMenu, TeamPingMenu 등), useItem으로 구성된다. 서버는 플레이어의 zoom(스코프)에 따라 컬링된 오브젝트만 클라이언트에 보낸다. 이 "서버가 그 플레이어에게 보내는 것"이 곧 사람이 아는 정보 집합이라는 점이 중요하다. 원작 surviv.io에는 텍스트 채팅이 없고 팀 핑과 이모트만 있다(소통 채널 설계에 영향).

없는 것: 액션 어댑터, 사람 정보 집합 기준의 관측 추출, 이벤트 탭, Python 브리지, 시드 결정성(README가 인정하듯 Math.random 잔존), 스킬 컨트롤러(내비게이션·루팅·교전), SLM 연동, 사람다움 평가 도구.

## 2. 크래프톤 PUBG Ally에서 가져올 것과 일부러 다르게 갈 것

가져올 것. 첫째, System 1/2 분리 — BT가 이동·조준·즉각 교전을 tick rate로 처리하고 SLM은 의도 해석·협업·발화를 맡아 반사 행동이 모델을 기다리지 않게 한다. 둘째, 텍스트 관측 도구 — "You are armed with M416, 24/30 bullets in the magazine. HP 78%, inside the safe zone" 같은 평문을 도구 호출로 가져오고, 프롬프트에 "tool results are the only ground truth"를 못 박는다. 셋째, 세계를 좁히기 — Sanhok 한 맵, AI Duo 한 모드, 고정 아이템 분류로 문제를 tractable하게 만들었다. 우리도 메인 맵 하나, 듀오 모드, 20개 안팎의 루트 분류로 시작한다. 넷째, 증류 파이프라인 — 대형 LLM 교사 → Minitron 8B 중간 교사 → Minitron 2B(Q4_K_M) 학생. 같은 아키텍처 계열 안에서 soft label KD가 직접 SFT보다 좋았다는 결론. 다섯째, 메모리 분리(매치 간 장기: 플레이어 프로필 / 매치 내 단기), 그리고 KV 캐시를 위해 시스템 프롬프트를 안정적으로 유지하고 실시간 정보만 갈아끼우는 프롬프트 구조. 여섯째, 평가 철학 — 자동 프로토콜/툴 사용 검증 + 라이브 A/B 플레이테스트 + 설문. "루프를 일찍 돌리고, 자주 돌릴 만큼 싸게 유지하라".

다르게 갈 것. Ally는 워크플로우(Action/Memory/Strategic/Proactive 에이전트 분리)에서 자율 에이전트(모델이 도구를 골라 3~5회 호출 후 결정)로 진화했지만, 2D 서바이브의 상태는 작다. 전술 루프는 다중 도구 호출 대신 400토큰 이하의 압축 상태 블록을 매 호출에 푸시하는 편이 지연이 짧고 학습 데이터 만들기도 쉽다. 도구 호출은 드문 조회(무기 스탯, 지형 지식)와 메모리 갱신에만 남긴다. 또 Ally는 음성이 핵심 UX지만 우리는 텍스트 채팅 오버레이(클라이언트 수정) + 팀 핑으로 시작하고, 음성은 stretch로 미룬다. 마지막으로 Ally의 System 1은 사람다움보다 유능함이 목표인 BT지만, 우리 System 1은 "사람의 텔(tell)을 내지 않는 것"이 1급 요구사항이다.

참고로 Orak 벤치마크(크래프톤 AI)는 8B 이하 오픈 SLM(Qwen2.5-3B/7B, LLaMA-3.2-3B, Minitron-4B/8B)이 zero-shot으로는 RPG·전략·시뮬레이션 장르에서 대부분 0점, 나머지 장르에서도 상용 모델에 크게 뒤졌고, 대형 LLM 궤적 ~1만 샘플로 SFT하면 메타 지식이 전이된다고 보고했다. NVIDIA Nemotron 3 Nano 4B(2026-03)는 동급에서 Orak 최고 성능을 주장한다. 즉 "SLM zero-shot으로 잘 놀길 기대하지 말고, 좁힌 세계에서 궤적으로 가르쳐라"가 일관된 교훈이다.

## 3. "사람다움"을 분해하기 — 어디서 사람다움이 나오고, 어디서 들키는가

사람다움은 세 층위로 나뉜다. 모터 층(100ms 단위): 반응시간, 조준 오차와 트래킹, 이동의 부드러움, 문 앞에서 멈칫, 루팅 전에 아이템을 "보는" 행동. 전술 층(1~수 초 단위): 교전/후퇴/루팅/힐 타이밍, 커버 선택, 팀원과의 거리 유지, 존 로테이션, 다운된 팀원 살리기. 사회·전략 층(10초~매치 단위): 페르소나의 일관성, 채팅과 핑의 타이밍과 말투, 파트너의 스타일 기억, 실수 후의 반응("아 미안").

AAMAS 2025 "The Many Challenges of Human-Like Agents in Virtual Game Environments"가 정리한 13개 난제 중 우리 프로젝트에 직접 걸리는 것은 이렇다. 사람은 다양하다(평균 사람을 만들면 어중간한 봇이 된다 → 페르소나 샘플링). 초인적 행동을 빼면 봇이 너무 약해진다(→ 스킬 티어를 파라미터로). 목적 없는 행동(둘러보기, 이모트, 잡동사니 줍기)이 사람의 신호다. 생물학적 제약(시야, 소리의 불확실성, 망각, 멀티태스킹 한계, 반응 지연, 당황). 실수를 하되 같은 실수를 반복하지 않는다. 사람과 봇을 구분하는 분류기를 QA 도구로 쓰라(그들은 F1 0.92짜리 분류기를 만들었다). 그리고 BotPrize 이래로 평가의 표준은 관전자 판정 튜링 테스트다.

이를 뒤집으면 "봇임을 들키는 텔 목록"이 되고, 이것이 System 1의 요구사항이자 자동 평가 지표가 된다.

| 텔(tell) | 사람 범위 (설계 목표) | 자동 검출 지표 |
|---|---|---|
| 반응시간 | 적 등장 후 첫 조준 변화 200~350ms(로그노멀), 루팅/힐 중이면 +100ms, 기습이면 +150ms | 적 최초 가시화 → 조준 벡터 변화까지 tick 수 히스토그램 |
| 조준 정확도 | 초기 오차 σ ∝ 거리, 250~400ms 시정수로 수렴, 트래킹 지연 80~150ms, 오버슈트 | 거리별 명중률, 조준각 오차 자기상관 |
| 이동 | 경로 최적성 0.7~0.95, 방향 전환에 관성, 문·모서리에서 감속, 가끔 오브젝트에 걸림 | 실제 경로 길이 / 최단 경로, 헤딩 변화율 분포 |
| 결정 주기 | 전술 판단 1~3초, 위기 시 짧아짐, 평온 시 길어짐 | 스킬 전환 간격 분포 |
| 관심/기억 | 화면 밖은 모름, 마지막 목격 위치는 8~15초 뒤 흐려짐 | 비가시 정보에 근거한 행동 비율(정보 집합 위반 = 0이어야 함) |
| 유휴 행동 | 안전할 때 둘러보기(조준 스윕), 이모트, 잡동사니 루팅 | 정지·비목적 행동 비율 |
| 소통 | 짧고 상황 지시적, 늦게 말하기도, 가끔 틀림 | 발화 길이·지연 분포, 관측 불일치율 |

정보 집합 동등성(information-set parity)을 원칙으로 못 박는다. 에이전트는 서버가 그 플레이어 소켓에 보냈을 정보(zoom 컬링된 오브젝트, 총성 이벤트, 킬피드, 팀 상태, 존)만 관측한다. 이는 공정성뿐 아니라 사람다움의 전제조건이고, survev 서버 구조상 거의 공짜로 얻을 수 있다.

## 4. 목표 아키텍처

```
 ┌──────────────── survev game server (Node/TS, 100 TPS) ────────────────┐     ┌── 3090 · Python ──┐
 │ Observation builder (client-visible set · events · memory)             │     │ System 2: SLM      │
 │      │ compact state block (≤400 tok)                                  │HTTP │  vLLM 사이드카      │
 │      ▼ on interrupt (enemy seen · hit · mate down · ping · skill done)  │────▶│  in: persona+rules │
 │ System 2 client ◀── {skill, params, commit_ms, say?, ping?} ───────────│◀────│      + state block │
 │      │                                                                 │     │  out: skill JSON   │
 │      ▼                                                                 │     └────────────────────┘
 │ System 1 (TS, every tick): skills → controllers                        │
 │   move_to · follow · loot · heal · engage · retreat · revive · …        │     ┌── Python · bridge ─┐
 │   + humanization (reaction delay · aim noise · path jitter · memory)   │ WS  │ 상대 정책 · 헤드리스 │
 │   (later: ONNX motor policy via onnxruntime-node, same process)        │◀───▶│ 롤아웃 · 교사 라벨링 │
 │ Action adapter → InputMsg → player.handleInput()                       │step │ · SFT/RFT · 하네스  │
 │ Event taps + JSONL logger (harness 공통 스키마)                          │(ticks)│ 메트릭/평가        │
 └────────────────────────────────────────────────────────────────────────┘     └────────────────────┘
```

v0.2 배치 원칙: 팀메이트 본체(System 1)는 게임 서버 안 TS, SLM은 사이드카, Python은 브리지 너머에서 적·데이터·학습·평가를 맡는다. 라이브(사람과 듀오)와 헤드리스(학습·평가) 모두 같은 TS 코드가 돈다.

System 2 입출력 예시. 상태 블록은 매 호출마다 갈아끼우고, 페르소나·규칙·스킬 목록은 고정 prefix로 두어 KV 캐시를 재사용한다.

```
[t=143s | zone: shrinking → center(312,208) r=95 in 22s, you are INSIDE]
[you: 62hp adr25 | M416 21/30, MP5 30/30 | 2 bandage 1 medkit 1 soda | scope 2x]
[teammate 'Kim': 34hp, 18m NW, up, said 3s ago: "밀자"]
[enemies: E1 41m N behind rock (last seen 1.2s ago, weapon ?) | shots heard: NE far x3 (4s ago)]
[cover: rock 6m N, bush 9m E, house door 14m W]
[loot: 7.62 ammo 5m, 4x scope 11m (in house)]
[recent: -18hp from N (2.1s ago); teammate pinged (330,190) 5s ago]
[memory: E1 is probably the one who shot Kim at t=131]
```

```json
{"skill":"hold","params":{"cover":"rock_N","face":"N"},"commit_ms":1800,
 "say":"바위 뒤 하나 있어, 잠깐 기다려","ping":null}
```

스킬 어휘(초안 11개): move_to(x,y,style), follow(teammate, dist), rotate_zone, loot(target|class), heal(item), engage(target, style: push|hold_angle|trade), peek(dir), retreat(cover|away_from), hold(cover, face), revive(teammate), idle_look. 각 스킬은 파라미터, 완료 조건, 중단 조건을 가지며 System 1이 tick마다 실행한다.

결정 주기는 "commit window + interrupt"로 한다. SLM은 commit_ms 동안 결정을 유지하고, 그 사이 System 1이 자율 실행한다. 다음 이벤트가 오면 commit이 깨지고 SLM을 다시 부른다: 적 최초 가시화, 피격, 팀원 다운/핑/채팅, 존 경고, 루트 발견, 스킬 완료·실패. 이렇게 하면 평균 호출 빈도는 1~2Hz 이하로 떨어지고, 결정 간격 분포가 자연스럽게 "위기에 짧고 평온에 길어져" 사람의 주의 패턴을 닮는다. SLM은 절대 회피·조준의 critical path에 있지 않다.

System 1 휴머나이제이션 레이어는 스킬 티어(bronze/silver/gold)와 페르소나(aggressive / cautious / looter / chatty / quiet)에 따라 파라미터를 샘플링한다. 교전 컨트롤러는 "이상적 조준 벡터 + 상태 의존 노이즈 + 지연 필터"로 만들고, 발사는 3~6발 버스트와 재장전 습관을 갖는다. 내비게이션은 맵 장애물에서 만든 격자 위 A* + 경로 스무딩 + 웨이포인트 지터, 문·모서리 감속, 안전할 때 조준 스윕. 이 파라미터들은 나중에 사람 로그가 생기면 그대로 피팅 대상이 된다.

소통 채널은 두 가지를 같이 간다. 클라이언트에 간단한 텍스트 채팅 오버레이를 추가해 파트너와 SLM이 대화하고(Ally의 음성 역할), 팀 핑/이모트를 System 2 출력에 포함한다. 핑·이모트만으로 소통하는 제약 버전은 별도 실험거리로 남긴다.

## 5. 합성 우선 데이터 전략 — 사람 데이터 없이 사람다움을 어디서 얻는가

사람 데이터가 없을 때 사람다움의 원천은 세 가지다. (a) 명시적 인지 제약: 3절의 파라미터를 문헌값(시각 반응시간 중앙값 ~220ms, Fitts 법칙류 조준 수렴)으로 초기화한다. (b) 대형 LLM의 사람 사전지식: LLM은 배틀로얄을 사람이 어떻게 플레이하고 어떻게 말하는지에 대한 방대한 텍스트(가이드, 커뮤니티, 해설)를 봤다. 페르소나와 인지 제약을 프롬프트에 명시한 교사 LLM을 "사람 시뮬레이터"로 써서 전술 결정과 발화를 생성하고 SLM에 증류한다. (c) 결과 필터링과 RL: 하네스의 메트릭 벡터로 "도움이 되는" 결정을 고르되, 텔 지표를 제약으로 걸어 초인적 방향으로 새지 않게 한다.

파이프라인은 4단계다.

1단계, 상태 분포 만들기. Ally v0(zero-shot SLM 또는 스크립트 정책) + 스크립트 상대/파트너로 헤드리스 롤아웃을 수천 매치 돌려 (state block, 이벤트, 결과)를 모은다. 오프라인 Game은 실시간보다 빨리 돌릴 수 있으므로 CPU 코어 수만큼 병렬로 돌린다.

2단계, 교사 라벨링. 교사(로컬 Qwen3.6-27B int4 한 장, 또는 API 모델)에게 상태 블록 + 페르소나 카드 + 인지 제약을 주고 {짧은 근거(≤40토큰), 스킬 JSON, 발화}를 생성한다. 같은 상태에 페르소나 3~5개를 샘플링해 다양성을 강제한다. Orak 방식대로 결과가 좋은 궤적을 우선 채택하되, "사람다운 실수"는 티어별 비율로 일부러 남긴다(예: bronze는 리로드 잊음 5%, 존 늦게 로테 8%). 발화는 관측 불일치 검사(말한 정보가 상태 블록에 있는가)로 거른다. 1~3만 샘플 목표.

3단계, SFT. 학생 SLM(2B~4B)을 "근거 포함으로 학습, 추론 시엔 근거 생략" 두 가지 포맷으로 함께 학습시키고, 추론 시 근거 없이 스킬 JSON만 내게 해서 지연을 줄인다. 근거 생략에 따른 결정 품질 하락을 측정하는 것이 자체로 재미있는 실험이다. 크기가 허락하면 9B 중간 교사 → 4B/2B soft-label KD(같은 계열: Qwen3.5 9B → 4B/2B, 또는 Nemotron Nano 9B → 4B)로 PUBG Ally의 결론을 재현해 본다.

4단계, 자기 개선. 먼저 RFT/expert iteration(같은 상태에서 N개 결정 샘플 → 메트릭+텔 제약으로 상위 채택 → 재-SFT)을 돌린다. 구현이 단순하고 안정적이다. 시간이 남으면 GRPO(verl 또는 Unsloth)로 멀티턴 에피소드 보상을 직접 최적화한다. 보상 = teammateResponseRate, 비고립률, 생존, 데미지의 가중합 + 텔 위반 페널티 + 발화 근거성(LLM-judge). "사람다움을 학습된 보상으로" — 사람 궤적이 생기면 사람/봇 판별기를 보상으로 쓰는 GAIL식 확장이 자연스럽게 붙는다.

사람 데이터 슬롯. 지금부터 모든 매치(개발 서버 포함)를 InputMsg + 상태 스냅샷 JSONL로 남기는 로거를 붙여 두면, 본인 플레이 몇 시간만으로도 (i) Stage A 문서의 "user-controlled baseline", (ii) 텔 지표의 사람 기준선, (iii) 판별기 v0의 양성 샘플이 된다. 데이터 정책을 바꾸는 게 아니라, 비용이 거의 0인 평가용 기준선으로 쓰자는 제안이다.

## 6. 컴퓨팅 예산 — 3090 4대로 무엇이 되는가

3090은 24GB, ~936GB/s, BF16 GEMM 실측 71~77 TFLOP/s(nano-vLLM 측정)다. 디코드는 대역폭 바운드라서 단일 스트림 tok/s ≈ 유효 대역폭 / 토큰당 바이트로 어림할 수 있다. 실측 예로 Qwen3-0.6B BF16이 단일 스트림 334 tok/s, 배치 128에서 4,794 tok/s, 1k 토큰 프리필 ~58k tok/s였다.

| 모델 | 가중치 | 단일 스트림 디코드(추정) | 40토큰 결정 지연 | 배치 64 집계(추정) |
|---|---|---|---|---|
| 2B BF16 (Qwen3.5-2B, Minitron-2B) | ~4GB | 120~170 tok/s | ~0.3s | 2~3k tok/s |
| 4B BF16 (Qwen3.5-4B, Nemotron 3 Nano 4B) | ~8GB | 60~90 tok/s | ~0.5s | 1.2~2k tok/s |
| 4B int4/Q4 | ~2.5GB | 150~200 tok/s | ~0.25s | — |
| 9B int8 (중간 교사) | ~9GB | 40~60 tok/s | — | 학습 데이터 생성용 |
| 27B int4 (교사, Qwen3.6-27B) | ~15GB | 20~30 tok/s | — | 배치 시 200~400 tok/s |

프리필은 1k 토큰에 20~40ms라 프롬프트 길이는 문제가 아니고, 고정 prefix 캐싱까지 쓰면 더 준다. 결론적으로 System 2 결정 지연 0.3~0.5초, 이벤트 기반 ≤2Hz는 3090 한 장으로 여유 있고, 한 장이 배치로 에이전트 10~20개를 동시에 서빙할 수 있어 헤드리스 롤아웃·RL에도 충분하다. 결정 시 chain-of-thought는 쓰지 않는다(100~200토큰이면 1~2초, 교전 중엔 못 쓴다).

학습 메모리. 4B LoRA SFT는 한 장에서 무리 없다. 2B full FT는 AdamW 기준 ~32GB라 한 장엔 안 들어가지만 ZeRO-2/3로 4장에 나누면 된다(장당 ~8GB + 활성화). 8B/9B는 QLoRA 한 장. GRPO는 Unsloth 문서 기준 4B LoRA + colocated vLLM이 10~15GB라 3090 한 장에 들어가고, verl로는 롤아웃(vLLM, TP=1, DP)과 학습을 GPU로 분리한다. 3090은 페어 NVLink 브리지가 없으면 PCIe라서 TP는 피하고 DP로 간다.

GPU 배치안. 평상시 GPU0 = 정책 SLM 서빙(vLLM, 라이브 플레이·평가), GPU1 = 교사(27B int4) 데이터 생성 또는 두 번째 롤아웃 서버, GPU2~3 = SFT/KD 학습. RL 구간엔 GPU0~1 롤아웃(DP=2), GPU2~3 학습(FSDP/ZeRO). 게임 시뮬레이션 자체는 CPU다. 128×128 시나리오 영역에서 오프라인 Game 스텝은 가볍지만, 상대 봇까지 포함한 병렬 매치 수는 CPU 코어에 묶이므로 초반에 tick당 ms를 재 두자.

모델 선택. 출발점은 Qwen3.5-4B(Apache 2.0, 네이티브 멀티모달이라 나중에 vision 확장 시 계열 유지, 0.8B/2B/4B/9B 사다리로 KD 실험 가능) 또는 Nemotron 3 Nano 4B(Orak 동급 최고 주장, 크래프톤 스택과 같은 NVIDIA 계열). 둘 다 zero-shot으로 Ally v0를 돌려 보고 스킬 선택 정확도·JSON 준수율·지연으로 하나를 고른다. 최종 배포 크기는 2B~4B, 교사는 9B(계열 내) + 27B/API(계열 외) 이중으로 둔다.

## 7. 평가 — 사람다움을 어떻게 재는가

자동 지표 세 묶음을 매 빌드마다 돌린다. 하네스 메트릭 벡터(도움이 되는가: teammateResponseRate, isolationRate, avgAllyDistance, damage/survival), 텔 지표(3절 표: 반응시간 히스토그램, 거리별 명중률, 경로 최적성, 결정 간격, 정보 집합 위반율, 유휴 비율), 소통 지표(발화 지연·길이 분포, 관측 불일치율, LLM-judge 자연스러움).

관전자 튜링 테스트(BotPrize/Navigation Turing Test 방식)는 12주차의 핵심 산출물이다. survev 관전 모드로 듀오 매치 클립(30~60초, 탑다운)을 20~30개 만들고, 판정자 5~10명(지인)에게 "두 팀원 중 AI는 누구인가"와 확신도, 근거 한 줄을 받는다. 클립의 절반은 사람-사람 듀오(본인 + 지인 로그) 또는 스크립트 봇을 섞어 기저율을 만든다. 채점은 판정 정확도(50%에 가까울수록 좋음)와 판정자가 적은 근거의 분류(조준/이동/판단/소통)로 한다. 근거 분류가 다음 반복의 우선순위를 알려준다.

파트너 경험 설문은 PUBG Ally의 A/B 방식을 축소한 것이다. 사람이 CPC와 듀오 3~5판을 뛰고 Likert 5점으로 "사람 팀원 같았다 / 도움이 됐다 / 말이 자연스러웠다 / 짜증났던 순간"을 답한다. 빌드 두 개를 블라인드로 비교한다.

판별기(사람 vs 봇)는 사람 로그가 조금이라도 생기면 만든다. 250스텝 시퀀스에 대한 1D-CNN/GRU면 충분하고 계산량은 무시할 만하다. AAMAS 논문의 제안대로 "봇을 개선해 판별기를 속이고, 판별기를 재학습"하는 반복이 QA 루프가 된다.

## 8. 6주 스프린트 계획 (v0.2 — 3개월 로드맵을 대체)

원칙은 셋이다. 매주 금요일에 "플레이 가능한 빌드"가 있어야 하고(주 1 데모), 구현의 대부분은 Claude가 하고 준성은 리뷰·결정·플레이테스트를 맡으며, 6주차 끝에 Go/No-Go를 결정한다. 3개월 계획에 있던 GRPO·KD(9B→2B)·음성·vision·건물 맵은 6주 범위 밖이다.

| 주 | 스프린트 목표 | 금요일 데모 (완료 기준) |
|---|---|---|
| 1 | 0단계 마무리: S4 관측(가시 집합) · S5 나머지(loot/heal/revive 탭, shots_heard) · S6 JSONL(하네스 `validate_episode` 통과) · S7 브리지(`reset` / `step(actions, ticks)`) | Python이 브리지로 4개 봇을 조종해 에피소드를 끝내고, 그 JSONL에서 하네스 메트릭 벡터가 나온다. 처리량(S1) 수치 기록. 기존 스크립트 봇을 "보이는 적에게만 반응"으로 바꾼 전지적 vs 관측 비교 리플레이 1장 |
| 2 | **첫 co-playable 순간.** System 1 in TS: 스킬 7개(move_to · follow · loot · heal · engage · retreat · revive) + 휴머나이제이션 v0(반응 지연·조준 노이즈·경로 지터) + `liveHook` 정리(`pnpm cpc:live`) | 준성이 브라우저에서 스크립트 팀메이트와 듀오 1판을 뛴다(적은 스크립트 봇 듀오). 플레이 로그 + "이상했던 점" 목록 |
| 3 | System 2 zero-shot: vLLM 사이드카(Qwen3.5-4B vs Nemotron Nano 4B), 상태 블록 빌더(≤400 tok), 스킬 JSON guided decoding, commit/interrupt 루프, 클라이언트 텍스트 채팅 오버레이 | SLM이 스킬을 고르고 짧은 채팅을 하는 Ally v0와 듀오 1판. 결정 지연 분포·JSON 준수율·스킬 선택 로그 |
| 4 | 합성 데이터 + SFT: 헤드리스 롤아웃(브리지, Python 상대) → 교사 라벨(페르소나 3~5, 근거+스킬+발화) 5~10k → LoRA SFT 4B → Ally v1 | v0 vs v1 오프라인(교사 일치율·JSON·지연) + 온라인(메트릭 벡터) 비교표. 듀오 1판 |
| 5 | 사람다움 패스: 텔 검출기(반응시간·거리별 명중률·경로 최적성·결정 간격·정보 집합 위반) · 휴머나이제이션 튜닝 · 발화 근거성 필터 · 여유 시 RFT 1라운드 | v0/v1/v1+휴머나이제이션 리플레이 나란히 + 텔 대시보드. 지인 1~2명 플레이테스트 |
| 6 | 평가·결정: 미니 튜링 테스트(클립 10~20, 판정자 5+) · 파트너 설문(3~5판) · 데모 영상 · 1페이지 write-up | Go/No-Go: 다음 6주에 무엇을 할지(ONNX 모터 정책, RFT/GRPO, 본 맵, 음성) 결정 |

브리지 step 단위에 대한 결정. `step(actions, ticks)`로 둘 다 지원하되 기본은 결정 단위다. `ticks=1`은 틱 단위(100Hz, 디버그·end-to-end 실험용), `ticks=3`은 사람 클라이언트의 입력 주기(33Hz)와 같은 "사람 해상도", `ticks=10`(0.1초)이 정책 기본값이다. 서버는 마지막 입력을 다음 패킷까지 유지하므로(사람 키 입력과 같은 의미론) `ticks≤3`이면 사람 대비 충실도 손실이 없다. 틱 단위는 가능하지만 스텝마다 관측 직렬화 + 왕복 오버헤드(~0.3~1ms)가 100번 붙어 처리량이 3~10× 실시간 수준으로 떨어질 것으로 예상하고, 결정 단위는 시뮬레이션 비용이 지배해 30× 이상이 목표다. S1 처리량은 `ticks=10` 기준으로 잰다. System 1이 TS 안에 있으므로 Python이 틱마다 개입할 이유는 원래 없다.

결정 로그(2026-09-08). 0단계 맵은 CPC 전용 맵 정의를 만들지 않고 `test_normal` 들판 + 시드 기반 루트 배치(PR-S2 `duo2v2Field.ts`: 팀별 대칭 스타터 킷 + 중앙 contested 킷)로 시작한다. 엄폐물·건물·가스 스케줄은 1단계 이후 필요 시 cpc_dev 안에서 추가한다. PR-S2(field 시나리오)와 PR-S3(액션 어댑터 `applyCpcAction.ts`, `stepGame.ts`)는 같은 날 구현·테스트 완료(vitest 12/12, M2~M4 충족).

진행 현황(2026-09-08 저녁). S5 절반(`eventTaps.ts`: fire/damage/down/kill), S4(`observation.ts`: 가시 사각형 기준 관측 + 키 allowlist), S7(`episode.ts` + `bridgeServer.ts`, `pnpm cpc:bridge`, 프로토콜 v0는 `docs/survev-bridge-v0.md`)까지 구현·테스트(cpc_dev 21개). 스크립트 상대(`scriptedPolicy.ts` chaser)를 서버에 내장. Python 쪽은 `experiment/survev_rl/`(브리지 클라이언트·목 브리지·216차원 featurizer·MultiDiscrete [9,16,2,2]·보상 설정·벡터 env·PPO·train/eval·ONNX export, 테스트 53개). 실제 브리지 측정(2 vCPU): `ticks=10` 42× 실시간/env, `ticks=3` 33×, `ticks=1` 14×, 8 env 배치 ~136× 집계 → S1 충족. 남은 0단계: S5 나머지(loot/heal/revive 탭, shots_heard), S6(JSONL + 하네스 `validate_episode`, M8), M9 비결정성 목록, M10 인수 테스트. PPO 목 실험에서 "생존+HP" 기본 보상은 도망·은신 정책으로 수렴함이 확인됨(무장+idle 상대 대조군은 85% 승률) — 보상/커리큘럼이 PPO 트랙의 첫 과제.

진행 현황(2026-09-09). 실제 클라이언트 렌더 파이프라인이 섰다: 컨테이너에서 survev 풀스택(API·게임 서버·Vite) + 헤드리스 Chromium, 게임 서버의 scratch 훅(`liveHook.ts`, 미커밋)이 사람 접속 시 필드 시나리오를 띄우고 사람을 무적 카메라(자유 줌, 총알 비충돌)로 바꾼다. 게임 속도를 0.2×로 낮춰 게임 시간 0.5 s마다 정확히 캡처한다(`out/ep1` 스크립트 봇, `out/ep2` PPO v0, `out/ep2_armed` armed 커리큘럼; 각각 프레임 + contact_sheet + `live_episode.json`). 이 과정에서 브리지의 실제 버그를 잡았다: Python 클라이언트는 입력을 이름(`Interact`, `EquipPrimary`, `Reload`)으로 보내는데 TS 어댑터가 숫자만 인식해 조용히 버려서, 실제 브리지의 PPO는 총을 줍지도 장착·재장전도 못 했다(survev `5f118a6c`에서 `toInput()`으로 수정 + 회귀 테스트). 새로 추가한 `experiment/survev_rl/policy_server.py`(브리지의 역방향: 라이브 게임이 관측을 보내면 체크포인트가 `CpcAction`을 돌려줌)로 PPO를 실제 게임 안에서 뛰게 했다. 수정된 브리지에서의 결과(600k step, 8 env): v0 기본 보상은 여전히 도망 정책(생존 6→19.8 s, 0발, 맵 구석 (1,1)에서 사망)이고, `--loadout armed`는 승률 91 %지만 프레임을 보면 "제자리에서 동쪽 고정 조준 + t=0 연사"라는 스폰 기하 exploit이며 적이 접근하면 도주 모드로 바뀐다. 결론: 다음 반복은 스폰 위치·방향·루트 배치의 에피소드별 시드 무작위화, 가장 가까운 적 대비 조준 오차 피처(또는 조준 보조), chaser는 벤치마크가 아니라 커리큘럼 상대. 사이드 노트: 하네스 `MOVE_LABELS`는 화면 y-down 기준이라 survev 월드 좌표에서는 `up_left`가 남서쪽 이동이다(벡터 기준으로 읽을 것).

진행 현황(2026-09-09 밤, PPO 트랙 반복 2). 스폰 무작위화(survev `d10a1f71`: `layout: "random"` — 시드 각도 회전 + 중심 거리 24~44 u, 팀 대칭·중심 응시, 킷 동반 이동; Park-Miller 첫 출력이 시드에 선형이라 시드 믹싱 추가)와 goal-conditioned 정책(`evolutionary-ai-battle a382164`: 관측 goal 블록 6차원, `--goal center`, 보상 항 `goal_progress`/`goal_hold`/`enemy_at_goal`)을 구현하고 준성이 제안한 "글로벌 포인트" 보상을 검증했다. 결과(각 40만 step, `out/ep3`·`out/ep4`): progress+hold+적-포인트 페널티(v1)는 맨손/데미지 항/auto-pickup/armed 어느 조합에서도 "포인트로 돌진해 4.6 s에 사망"으로 수렴 — progress가 2 s 안에 적립되고 사망이 페널티 스트림을 끊어서 자살 돌진이 최적이 되며, 페널티 제거(적 사살)까지의 credit 경로가 너무 길어 사격은 학습되지 않는다. 적립 불가 항만 남기고 사망 −1을 넣은 v2(`configs/point_v2.json`)는 armed에서 후퇴 사격(kiting) 전투를 배우기 시작(데미지 8→68, 승률 18 %, 상승 중)하지만 포인트는 무시한다. 파밍은 어느 런에서도 우연 이상으로 나오지 않았다. 결론: 한 단계의 간결한 보상으로 "파밍→전투"는 안 나오고, 같은 2~4항을 유지한 3단계 커리큘럼(armed+random에서 전투 → resume 후 맨손+auto-pickup으로 파밍 → hold/enemy_at_goal 추가로 포인트 유지)이 근거 있는 최소 경로다. PPO 트랙은 GPU 머신 백그라운드로 돌리고, 2주차 co-playable 순간은 계획대로 TS System 1 스킬로 낸다.

진행 현황(2026-09-10 새벽, PPO 트랙 반복 3 — race 목표). 준성 제안대로 "글로벌 포인트 하나 → 먼저 닿은 팀 +1 → 새 랜덤 포인트, 60 s 고정, 전멸 후에도 계속"을 구현했다(survev `a0bb89a3` `objective.ts`/`endOnElimination`, evolutionary-ai-battle `8c6937b` `configs/race_v1.json` = capture 1.0 + damage 0.02). 결과(`out/ep5`): 맨손 60만 step에서 경주를 배워 팀 캡처 2.0/에피소드(스크립트 최적 페이스와 같은 ~4 s/개), 200만 step에서 2.8개·생존 10.8 s — 자살 최적이 사라지고 생존이 늘어나는 첫 보상 구조다. 그러나 싸움은 나오지 않는다: 맨손은 총보다 1 u/s 빨라서 달리는 게 최적이고, armed(60만)에서는 반대로 포인트를 무시하고 chaser와 싸운다(데미지 46, 킬 0.09). 둘 다 "쫓아오기만 하는 상대" 아래의 지역 최적이라, 다음 지렛대는 보상 항이 아니라 상대다: 포인트를 같이 다투는 스크립트 `racer`(적의 캡처 = 우리 손실, 적 사살 = 같은 에피소드 안의 캡처 이득)를 넣고, auto-pickup 또는 armed 단계로 워밍업한다. 보상은 두 항 유지.

진행 현황(2026-09-10, PPO 트랙 반복 4 — racer 상대·gun_pickup). 스크립트 `racer`(survev `4d93b59d`: chaser의 루팅·전투 단계를 공유하되 25 u 안의 적에게만 교전하고 그 외에는 포인트로 달림)와 팀 색 스킨(`170e7bd8`: team-a 파랑 outfitBlueLeader / team-b 빨강 outfitRed — 50v50 팩션 스킨을 아웃핏으로 적용, test_normal은 팩션 맵이 아니라 클라이언트 팀 패치가 안 나옴)을 넣었다. 결과(각 60만 step, `out/ep6`·`out/ep7`): racer 상대 + auto-pickup(H)은 캡처 1.6/에피소드로 경주는 하지만 여전히 맨손, armed 파이터 워밍업(I)은 생존은 길어도 경주를 안 함. 준성 지적("이것도 주울 생각을 안 한다")대로 원인은 총을 쥐기 전에 총의 가치를 말해주는 항이 없다는 것 → `gun_pickup`(첫 무장 시 1회 +1, `configs/race_v2.json`, evolutionary-ai-battle `1b1d512`)을 넣자 69 %가 스폰 1.3 s 만에 총을 줍고 사격이 3배가 됐지만 이번엔 포인트를 무시하고 싸운다. 정리: 항이 이름 붙인 행동(경주=capture, 파밍=gun_pickup, 전투=damage)은 각각 나오지만 60만 step·CPU 2코어에서는 세 가지가 합쳐지지 않는다. 다음은 GPU 머신에서 race_v2를 300~500만 step(브리지 프로세스 코어당 1개, 시드 4개)으로 돌리고, 그 체크포인트를 auto-pickup 없이 racer 상대로 `--resume`하는 것. 보상은 세 항(capture, gun_pickup, damage)에서 멈춘다.

진행 현황(2026-09-10, PPO 트랙 반복 5 — 조준 비대칭·상대 강도 노브·GPU 핸드오프). 준성의 질문("이 방향으로 GPU 가더라도 agent가 더 똑똑해질까")에 답하면서 원인을 좁혔다: 11개 run 전부 kills 0 — 정책의 조준은 절대각 16빈(명중률 6~12 %)이고 스크립트 상대는 정확 조준(60~70 %)이라 교전은 스텝 수와 무관하게 항상 지고, "적을 죽이면 남은 포인트를 독식한다"는 credit 경로를 한 번도 경험하지 못한다. GPU 자체는 지렛대가 아니다(정책은 작은 MLP, 벽시계는 브리지 CPU가 쓴다). 그래서 넘기기 전에 두 개를 넣었다: `--aim-assist`(fire를 누르는 동안 가장 가까운 보이는 적으로 조준 스냅 = 상대와 같은 원시 동작, evolutionary-ai-battle `985d4c1`)와 상대 강도 노브 `scriptedOptions`(survev `d80b5946`: 조준 노이즈 σ°, 반응 지연 s, racer 교전 거리; 정지 표적 둘을 chaser가 잡는 시간 6.1 s(정확) → 8.6(5°) → 14.3(10°) → 25.6(20°)). 파일럿 K0(race_v2, racer 10°, 30만 step, 2코어): 같은 step의 J 대비 데미지 3.5→48, 명중률 0.07→0.25, kills 0→0.18/에이전트, 생존 11→24 s — 프로젝트 최초의 킬이고, 정확 조준 상대에게도 절반쯤 전이된다; 레이스는 아직(팀 캡처 0.15). captures/team_captures/armed를 `progress.csv`와 eval 요약에 노출했다. GPU 서버 실행 계획은 `docs/cpc-handoff-gpu.md`: K1–K2(스텝만 늘린 대조군), K3–K4(aim assist + 10° racer, 400만 step), K5(상대 강도 커리큘럼 10°→5°→정확), L(auto-pickup 제거), M(armed 대조군), 판단 기준은 같은 step에서 K3 vs K1. 원칙 재확인: 이 트랙의 목적은 강도 조절이 되는 상대/기준선이고, K에서 합성이 안 나와도 2주차 System-1 스킬 + SLM으로 넘어간다 — 스킬 계층의 원시 동작도 상대와 대칭이어야 한다는 교훈을 가지고.

첫 2주 체크리스트(survev 쪽). (1) `applyCpcAction(player, {move:{x,y}, aim:{x,y}, fire, inputs[], useItem})` → InputMsg 조립 후 `player.handleInput(msg)` 호출, 8방향이 아니라 연속 벡터를 touchMoveDir/Len 경로로 넣을지 4키로 양자화할지 결정(사람은 4키 + 마우스이므로 4키 양자화가 오히려 사람답다). (2) 관측은 net update가 소켓에 보내는 것과 같은 컬링을 재사용해 만들고, 사람이 못 보는 필드(적 HP 정확값, 비가시 적 위치)는 스키마에서 아예 제외한다. (3) 이벤트 탭은 Player.damage / kill / knock / revive / loot pickup / heal / weapon fire 지점에 후크를 걸고 "shots heard"는 거리 기반으로 합성한다. (4) 시드 결정성은 맵 RNG를 시드하고 게임 로직의 Math.random을 시드 PRNG로 감싸되, 완전 결정성이 안 되면 통계적 평가(시드당 N회)로 대체한다. (5) 하네스의 공통 스키마 v0를 그대로 쓰고, 액션의 aim만 연속형으로 확장한다.

첫 2주 체크리스트(하네스/Python 쪽). 계층형 베이스라인의 `select_intent` + `create_global_plan_if_needed` 자리에 `SLMPolicy` 인터페이스(입력: state block, 출력: skill JSON)를 정의하고, 우선 규칙 기반 스텁으로 채워 파이프라인을 끝까지 관통시킨다. toy 환경은 스킬 컨트롤러의 단위 테스트와 RL 스모크 테스트 용도로 유지한다.

## 8-1. 0단계(브리지, 1~2주차) 성공 기준 — Definition of Done

원칙: 모든 기준은 테스트 하나 또는 명령 한 줄로 pass/fail이 갈려야 한다. M(must) 10개를 전부 통과해야 0단계가 끝나고, S(should)는 미달 시 수치를 기록한 채 1단계 백로그로 넘긴다.

Python이 붙는 계약(contract)은 다음 한 가지다.

```python
env = SurvevEnv("ws://127.0.0.1:8765", scenario="duo2v2_field", seed="cpc-duo2v2-seed-0",
                tick_dt=0.01, net_sync_every=3)
obs, info = env.reset()                                              # obs: {agent_id: AgentObservation}
obs, events, done, info = env.step({aid: action for aid in obs}, ticks=10)   # ticks=1 틱 단위 · 3 사람 입력 주기 · 10 기본
```

| ID | 기준 | 판정 방법 | PR |
|---|---|---|---|
| M1 | API 계약: 4개 에이전트(양 팀) 모두 Python에서 제어, reset/step/close 동작, 종료 시 `done=True`와 `info.winner_team`, `info.reason ∈ {elimination, time_limit}` | pytest e2e: 랜덤 정책 4개로 에피소드가 종료까지 진행 | S7 |
| M2 | 이동 충실도: 어댑터가 만든 InputMsg는 손으로 만든 InputMsg와 필드 동일하고, 장애물 없는 지면에서 1초 이동 시 변위 = 서버 실효 속도(`moveSpeed` 12 + 장비/상태 보정) × 1s ± 2%. 4키 양자화·연속(touchMove) 모드 모두 | vitest | S3 |
| M3 | 사격 충실도: 반자동 무기 `fire.start` → 총알 정확히 1발, `fire.hold` 1초 → 무기 fireDelay 기준 발수 ± 1, 총알 방향 = aim ± 무기 spread | vitest | S3 |
| M4 | 상호작용 충실도: `Interact`로 사거리 내 루트 픽업(인벤토리 변화), 다운된 팀원 옆 `Revive` 유지 8s(`reviveDuration`) 후 `downed=false`, `UseBandage` 등 사용 시 HP 증가 | vitest | S3 |
| M5 | 관측 신선도·동일성: 매 netSync 직후 관측의 오브젝트 id 집합 == 그 플레이어의 `visibleObjects` id 집합, 관측의 tick == 마지막 netSync tick, 사각형을 벗어난 오브젝트는 ≤3틱 안에 사라짐. 100 netSync 연속 불일치 0 | vitest | S4 |
| M6 | 정보 누출 0: 컬링 사각형 밖의 적은 관측에 없다(네거티브 테스트), 관측 JSON 키는 allowlist 스키마의 키만(미등록 키 발견 시 실패), 적의 정확 HP·인벤토리 등 클라이언트가 못 받는 필드 없음, 팀원은 groupStatus 수준(위치·HP·downed)만 | vitest + 스키마 allowlist 테스트 | S4 |
| M7 | 이벤트 정확성: "A가 정지한 B를 사격" 60게임초 시나리오에서 damage 이벤트 합 == B의 HP 손실(방어구 감산 후) ± 0.5, `down` → `kill` 순서로 각 1회, `fire` 이벤트 수 == 생성 총알 수, 가스 피해는 `source=gas`, loot/heal/revive 이벤트가 각각 1회 이상 나오는 스크립트 시나리오 통과 | vitest | S5 |
| M8 | 데이터·평가 연결: 에피소드 JSONL이 하네스 `experiment/core/schema_validation.validate_episode()`를 에러 0으로 통과하고, 하네스 메트릭 계산이 그 파일에서 돌아 combat(damageDealt>0)·survival·cooperation(isolationRate 등) 값이 나온다 | pytest (하네스 repo 임포트) | S6 |
| M9 | 재현성 최소선: 같은 seed → 같은 맵 장애물 해시·같은 스폰 위치. 완전 결정성은 요구하지 않되, 비결정성 원인(`Math.random` 사용처) 목록을 README에 기록 | vitest + README | S7 |
| M10 | 개발 위생: `pnpm cpc:episode -- --scenario duo2v2 --policy random --seconds 60 --out .tmp/ep.jsonl` 한 줄로 동작, vitest(S3~S6) + pytest(S7) 전체 ≤ 2분, cpc_dev/README에 API·스키마·처리량·비결정성 기록 | CI 러너 | 전체 |
| S1 | 처리량: 단일 프로세스, 4 에이전트, 128×128, `tick_dt=0.01`에서 60게임초 ≤ 3초 벽시계(≥20× 실시간). 미달 시 측정치 기록 | pytest 벤치 | S7 |
| S2 | 지연: reset ≤ 3초(맵 재생성 포함), step 왕복 오버헤드(시뮬 시간 제외) ≤ 5ms | pytest 벤치 | S7 |
| S3 | 병렬: 코어당 프로세스 1개로 N개 매치 동시 실행 스크립트(포트 자동 할당), N=8에서 집계 처리량 선형에 가깝게(≥ 6×) | 스크립트 + 로그 | S7 |
| S4 | 총성 이벤트: `shots_heard`는 거리 R 이내 에이전트에게만, 방향(8방위)·거리(near/mid/far) 버킷으로 제공, R 밖 에이전트에게는 전달되지 않음 | vitest | S5 |
| S5 | (stretch) 사람 플레이 슬롯: 실제 클라이언트로 접속한 매치도 같은 JSONL(InputMsg + 관측)로 기록 | 수동 1회 + 파일 검증 | — |

0단계에서 명시적으로 하지 않는 것: 스킬 컨트롤러·길찾기, SLM 연동, 클라이언트 채팅 오버레이, `Math.random` 시드 패치(완전 결정성), 실내 건물 처리. 최종 인수는 `pytest tests/test_phase0_acceptance.py` 하나로 M1~M10을 순서대로 검증한다: 브리지 기동 → reset(seed) → 스크립트 정책(추격+사격 vs 정지) 60게임초 → 이벤트/HP 정합 → JSONL 저장 → validate_episode → 메트릭 계산 → 처리량 로그.

## 9. 리스크와 열린 질문

가장 큰 리스크는 System 1 엔지니어링 비용이다. survev 맵의 건물·문·강·부시를 다루는 내비게이션과 루팅만으로 3~5주차를 다 쓸 수 있다. 완화책은 시나리오 영역을 128×128로 제한하고 건물 실내 진입을 1차에서 제외하는 것, 그리고 스킬을 11개 전부가 아니라 move_to / follow / loot / heal / engage / retreat / revive 7개로 시작하는 것이다.

두 번째는 합성 데이터의 "LLM스러움"이다. 교사가 만든 발화·결정이 사람이 아니라 가이드북처럼 들릴 수 있다. 페르소나·인지 제약·의도적 실수 주입, 발화 길이 상한(한국어 15자 내외), 그리고 12주차 튜링 테스트의 근거 분류로 교정한다. 근본 해결은 사람 데이터이므로 로거는 1주차부터 켠다.

세 번째는 RL의 시간 비용이다. 멀티턴 에피소드 GRPO는 환경 인프라(병렬 게임, 리셋, 보상 계산)가 무겁다. RFT로 먼저 이득을 확인하고 GRPO는 1회 실험으로 제한한다.

네 번째는 결정성·재현성이다. 완전 결정성이 어렵다면 통계적 평가로 가고, 대신 로그를 충분히 남긴다.

열린 질문들. 소통 채널을 텍스트 채팅으로 갈지 핑/이모트만으로 갈지(둘 다 준비하되 튜링 테스트는 어느 쪽으로?). 스킬 티어를 파트너에 맞춰 적응시킬지(사람 파트너의 K/D를 추정해 비슷한 티어로 맞추면 "같이 노는 맛"이 커진다). 페르소나를 매치마다 고정할지, 파트너 프로필 장기 기억을 어디까지 넣을지. 상대 팀도 CPC로 채울 때 "봇 vs 봇" 매치가 사람 분포에서 얼마나 멀어지는지.

## 10. 아이디어 파킹랏 (stretch)

End-to-end SLM 정책 실험: 0.8B~2B 모델이 5~10Hz로 저수준 액션 토큰을 직접 내는 "action-token SLM". 3090 한 장으로 에이전트 하나는 가능하지만 연속 조준의 사람다움은 낮을 것이다. 계층형과의 튜링 테스트 대조군으로 가치가 있다.

음성: 크래프톤이 2026-04 공개한 Raon-Speech(9B), Raon-SpeechChat(전이중 대화), Raon-OpenTTS가 오픈소스다. 텍스트 채팅이 자리 잡으면 SpeechChat을 붙여 "말로 하는 듀오"를 시연할 수 있다.

Vision: Qwen3.5 계열은 네이티브 멀티모달이라, 렌더된 클라이언트 프레임을 넣는 실험을 모델 교체 없이 할 수 있다. 정보 집합 동등성의 궁극 버전(픽셀만 본다)이지만 지연과 데이터 비용이 크므로 후순위.

핑/이모트 전용 언어: 원작의 제약 안에서 팀 핑과 이모트만으로 협업 의도를 전달하는 문제. 작고 예쁜 부문제이며 사람도 실제로 그렇게 논다.

파트너 적응: 사람 파트너의 플레이 스타일(공격성, 루팅 시간, 선호 무기)을 매치 내에서 추정해 CPC의 페르소나·거리 유지·발화 빈도를 맞춘다. Ally의 장기 기억(선호 무기, 드롭 위치)과 같은 축.

사람다움 보상: 사람 로그가 생기면 판별기를 보상으로 쓰는 GAIL-for-SLM. "정책이 판별기를 속이고 판별기가 재학습"하는 루프를 스킬 수준에서 돌린다.

정보 집합 위반 검사기를 CI에 넣기: 에이전트가 비가시 정보로 행동하는 순간을 자동 검출해 빌드를 실패시키는 것. 공정성과 사람다움을 동시에 지키는 가장 싼 장치.

## 참고 자료

- KRAFTON AI, From Workflow-Based SLM to Autonomous Agent: Evolving PUBG Ally's Architecture (2026-04-15): http://www.krafton.ai/blog/pubg_ally_nemotron/
- NVIDIA Technical Blog, Q&A: How KRAFTON Built PUBG Ally, a Co-Playable Character Powered by NVIDIA ACE: https://developer.nvidia.com/blog/how-krafton-built-pubg-ally-a-co-playable-character-powered-by-nvidia-ace/
- KRAFTON, PUBG Ally Beta Test (Ally Duo, 2026-06-17~07-01): https://www.krafton.com/en/news/press/krafton-introduces-pubg-ally-beta-test/
- Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games: https://arxiv.org/abs/2506.03610 · https://github.com/krafton-ai/Orak
- Krafton Raon 모델 패밀리 공개(2026-04): https://en.sedaily.com/news/2026/04/02/krafton-launches-ai-model-brand-raon-releases-four-open · https://github.com/krafton-ai
- The Many Challenges of Human-Like Agents in Virtual Game Environments (AAMAS 2025): https://arxiv.org/abs/2505.20011
- Navigation Turing Test (NTT): Learning to Evaluate Human-Like Navigation: https://arxiv.org/abs/2105.09637
- Counter-Strike Deathmatch with Large-Scale Behavioural Cloning: https://arxiv.org/abs/2104.04258
- Small Language Models are the Future of Agentic AI (NVIDIA): https://arxiv.org/abs/2506.02153
- Nemotron 3 Nano 4B: https://huggingface.co/blog/nvidia/nemotron-3-nano-4b · Qwen3.5 small series: https://huggingface.co/Qwen/Qwen3.5-0.8B
- RTX 3090 서빙 실측(nano-vLLM): https://dev.to/maomaoling/inside-nano-vllm-what-an-rtx-3090-reveals-about-llm-serving-1f97
- Unsloth Memory Efficient RL: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl · GRPO+LoRA with verl 핸드북: https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl
- 로컬 repo: C:\repos\evolutionary-ai-battle (docs/common-interface-v0.md, docs/evaluation-metrics.md, experiment/baselines/hierarchical_baseline) · C:\repos\survev (server/src/cpc_dev/README.md, shared/net/inputMsg.ts, shared/gameConfig.ts Input enum, config.ts gameTps/netSyncTps)
