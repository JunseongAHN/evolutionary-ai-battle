# CPC v0 — 베이스라인 (2026-09-11)

v0은 **미세조정 없는(zero-shot) SLM이 판단하는 첫 co-playable 팀메이트**다. 이후의 모든 개선(v1 = 4주차 합성 데이터 + LoRA)은
이 문서의 설정과 수치에 대해 비교한다. 두 리포 모두 `cpc-v0` 태그로 고정했다.

| repo | branch | tag | commit |
|---|---|---|---|
| survev | `feature/cpc-dev` | `cpc-v0` | `485a538a` |
| evolutionary-ai-battle | `feat/survev-rl` | `cpc-v0` | 이 문서를 담은 커밋 |

## 구성

- **System 2 (판단)**: Qwen3.5-4B Q4_K_M, 미세조정 없음. llama-server(OpenAI 호환), GPU 1, `127.0.0.1:8090`.
  결정당 스킬 JSON 하나 `{skill, params, commit_ms, say}`.
- **문법 제약**: `skillGrammar.ts`가 만든 GBNF. 스킬마다 자기 파라미터만, 에이전트 id는 닫힌 집합, `commit_ms`는 사다리
  (300–3000), `move_to`는 좌표가 아니라 이름 목표(`"point"` / 에이전트 id / `"loot:<종류>"`), `loot`·`heal`은 필드 없음.
- **스킬 마스크**: `skillMask.ts`가 질문마다 지금 가능한 스킬만 문법에 넣는다. 무장한 적이 25 m 안이면 `revive` 없음,
  힐 아이템 없으면 `heal` 없음, 총과 탄약이 있으면 `loot` 없음. 에이전트 자신의 관측만 읽는다.
- **commit/interrupt 루프**: `plannerLoop.ts`. 게임은 모델을 기다리지 않는다. 결정 유지 시간 만료, 스킬 완료·실패, 피격,
  새 적, 팀원 다운 때 다시 묻고, 그 사이·오류 때는 스크립트 선택기가 대신 움직인다. 즉시 끝나는 결정 뒤에는 1초 쉰다.
- **System 1 (실행)**: 스킬 7개(`move_to`·`follow`·`loot`·`heal`·`engage`·`retreat`·`revive`)를 매 틱 실행.
  할 일이 없으면 사람 파트너를 따라간다.
- **상태 블록**: `stateBlock.ts`, 관측에서만 만든 평문(실측 약 90토큰), `[can do: ...]` 줄 포함.
- **라이브 시계**: 시나리오를 세운 순간부터 흐르는 자체 시계. (게임 시작 전 10초 동안 멈춰 있는 `startedTime`을 쓰면
  반응 지연이 끝나지 않아 대치가 생긴다 — 두 번째 플레이테스트의 원인.)

## 재현에 필요한 식별값

| 항목 | 값 |
|---|---|
| 모델 | `Qwen3.5-4B-Q4_K_M.gguf` (unsloth/Qwen3.5-4B-GGUF), sha256 `00fe7986ff5f6b46…`, 2,740,937,888 bytes |
| 서버 | llama.cpp `df03399`, CUDA 12.2 / sm_86 빌드, `-DGGML_CUDA_NO_VMM=ON`, `libcuda` 스텁 링크 (드라이버 535에서 공식 이미지·vLLM 불가) |
| 서버 옵션 | `--n-gpu-layers 99 --ctx-size 4096 --cache-reuse 256` |
| 샘플링 | temperature 0.7, max_tokens 96, `chat_template_kwargs.enable_thinking=false`, `cache_prompt=true` |
| 프롬프트 | `plannerSystemPrompt("ko")` (survev `plannerPrompt.ts`), 덤프 sha256 `c0ec05d1b1217a53…`. 벤치 내장 사본과 바이트 단위 동일 |
| 평가 세트 | `experiment/slm/cases_v1.json`, sha256 `6c13118ff8e5b0f5…`, 9개 상황(마스크·문법 포함). **재생성 금지** (전투가 시드되지 않아 같은 이름에 다른 상황이 잡힌다) |
| CPC 휴머나이제이션 | `{"aimNoiseDeg":6,"reactionDelay":0.25,"pathJitterDeg":8}` |
| 적 | 스크립트 `chaser` 듀오, `{"aimNoiseDeg":10,"reactionDelay":0.5}` |
| 시나리오 | `duo2v2_field`, `test_normal`, 시드 `cpc-live-seed-0`, 목표 없음 |

실행 환경(`cpcProject/docker/`의 Dockerfile·compose)은 방침상 커밋하지 않는다. 핵심 빌드 플래그는 위 표에 적었다.

## 수치

### 평가 세트 `cases_v1` (상황당 10회, 게임 프롬프트, 마스크 적용)

| 상황 | 기대 | 결과 |
|---|---|---|
| spawn_unarmed | loot | 10/10 |
| enemy_in_view_unarmed | loot / retreat | 10/10 |
| enemy_in_view_armed (적 32–34 m) | engage | **0/10** (팀원 쪽 `move_to`) |
| taking_fire | engage / retreat | 10/10 |
| race_point_quiet | move_to "point" | 10/10 |
| teammate_downed_under_fire | engage / retreat | 10/10 |
| idle_armed_no_point | follow / 팀원에게 move_to | 5/10 (나머지는 두 번째 총) |
| revive_under_fire (플레이테스트 원문) | engage / retreat | 10/10 |
| **합계** | | **65/80 (81%)** |
| 대조군: revive_under_fire, 마스크 없음 | engage / retreat | 0/10 (10/10 `revive` — 플레이테스트 실패 재현) |

유효한 스킬 JSON 90/90. 결정 지연 중앙값 280 ms, p90 373 ms (단일 3090, 약 150 tok/s).

### 모델·언어 비교 (마스크 이전, 6개 상황 × 10회)

| | 적절 |
|---|---|
| Qwen3.5-4B, 한국어 발화 | 47/60 (78%) |
| Qwen3.5-4B, 영어 발화 | 46/60 (77%) |
| Nemotron-3-Nano-4B, 한국어 | 40/60 (67%) |
| Nemotron-3-Nano-4B, 영어 | 44/60 (73%) |

발화 언어는 Qwen의 판단을 바꾸지 않는다. 한국어가 기본, 영어는 `CPC_SAY_LANG=en`.

### 라이브

- 마스크 적용 후 헤드리스 한 판: 질문 23, 결정 23, 오류 0, 금지된 선택(사격 중 리바이브·아이템 없는 힐·무장 중 루팅) 0.
- 시계 수정 후: 플래너가 0.0 / 1.8 / 2.8 / 3.5 / 4.0 / 4.5초에 다시 묻고, 적이 사거리에 들어온 t≈4에 교전 시작.
- 준성 플레이테스트(2026-09-11): "처음에 아무것도 안 하고 대치" 해소 확인.

## 알려진 한계 (v1이 넘어야 할 것)

1. 무장했고 적이 32 m에 보이면 교전하지 않는다 (0/10).
2. 할 일이 없을 때 절반은 사람 곁을 떠나 두 번째 총을 주우러 간다 (5/10).
3. 발화가 어색하고 가끔 근거가 없다 ("술 치자", 동쪽 적을 "남쪽"). 게임 안 채팅창이 없어 서버 로그 `[cpc:say]`로만 보인다.
4. 첫 사격까지 약 4초 (줍기 1 + 꺼내기 0.75 + 재장전 2.5 + 반응 0.25). 맵의 총은 빈 탄창으로 들어온다 — 사람과 같은 규칙이라 유지.
5. 헤드리스 라이브 테스트의 "사람"은 가만히 서 있어 사실상 1대2다. 판단 품질은 사람 플레이로만 볼 수 있다.

## 재현

```bash
cd ~/repos/cpcProject/docker
docker compose run -d --name cpc-slm slm                  # Qwen3.5-4B on GPU 1, 127.0.0.1:8090
docker compose run -d --name cpc-web web                  # CPC_PLANNER_URL=http://127.0.0.1:8090
# 플레이: ssh -L 3000:127.0.0.1:3000 -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 <server> → http://127.0.0.1:3000

# 벤치 (게임 프롬프트 덤프 후)
docker compose run --rm train python experiment/slm/bench_planner.py --url http://127.0.0.1:8090 \
  --blocks experiment/slm/cases_v1.json --grammar /work/out/slm/skill.gbnf --runs 10 \
  --prompt-file /work/out/slm/prompt_ko.txt
```

## v1과 비교하는 법

같은 `cases_v1.json`, 같은 프롬프트 해시, 같은 샘플링으로 벤치를 돌리고, 같은 시드·적 강도로 라이브를 돌린다. 모델만 바꿨을
때의 차이가 v1의 기여다. 평가 세트를 늘릴 때는 `cases_v2.json`을 새로 만들고 v0도 그 위에서 다시 잰다.
