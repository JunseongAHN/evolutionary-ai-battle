# CPC PPO 핸드오프 — GPU 서버(Linux, 4×RTX 3090)에서 돌릴 것

작성 2026-09-10. 컨테이너(2 vCPU)에서 하던 실험을 **Linux GPU 서버(RTX 3090 ×4)** 로 옮기는 시점의 상태와,
그 다음에 **그대로 실행할 커맨드**를 적어 둔다. 개발·커밋은 Windows 머신(`C:\repos\…`)에서 했으므로 서버에서는 push된 브랜치를 clone해서 시작한다.
코드 설명은 `experiment/survev_rl/README.md`, 프로토콜은 `docs/survev-bridge-v0.md`, 서버 쪽은 survev `server/src/cpc_dev/README.md`에 있고 여기서는 반복하지 않는다.
아래 커맨드는 모두 bash 기준이고, 같은 형태를 Linux 컨테이너에서 실제로 돌려 검증했다(node 22 / pnpm 10 / Python 3.11 / glibc 2.39).

## 0. 한눈에

- 파이프라인은 끝까지 붙어 있다: survev 필드 시나리오(시드 랜덤 스폰·루트, 레이스 목표, 팀 스킨, `racer` 상대, 상대 강도 노브)
  ↔ 브리지 ↔ Python PPO(목표 6-d 관측 블록, 보상 3항, aim assist) ↔ 정책 서버 ↔ 실제 클라이언트 렌더(contact sheet).
- 2 코어에서 600k 스텝짜리 실험 11개(A–J)의 결론: **보상항 하나가 그 이름의 행동 하나를 만든다** — `capture`→레이스,
  `gun_pickup`→총 줍기, `damage_dealt`→사격. 셋의 **합성(줍고 → 싸우고 → 레이스)** 은 나오지 않았다.
- 왜 안 나왔는지도 안다: **11개 run 전부 kills = 0**. 정책의 조준은 절대각 16빈(22.5°)이라 명중률 6–12 %인데 스크립트 상대는
  정확 조준(60–70 %)이라 교전은 스텝 수와 무관하게 항상 진다 → "죽이면 남은 포인트 독식"이라는 credit path를 한 번도 경험 못 한다.
  GPU는 지렛대가 아니다(정책은 작은 MLP, 시간은 브리지 CPU가 쓴다). 스텝만 늘리면 "더 오래 살고 데미지 조금 더"가 상한이다.
- 그래서 넘기기 직전에 두 개를 넣었다. `--aim-assist`(fire를 누르고 있으면 가장 가까운 보이는 적으로 aim 스냅 = 상대와 같은
  원시 동작)와 브리지 `scriptedOptions`(`--opp-aim-noise`, `--opp-reaction`, `--opp-engage-dist` = 상대 강도 노브).
  **파일럿 K0**(300k, 2 코어, 3.5 분): 같은 스텝의 J 대비 데미지 3.5 → 48, 명중률 0.07 → 0.25, **kills 0 → 0.18/에이전트**,
  생존 11 → 24 s. 프로젝트 최초의 킬. 레이스는 아직 없음(team captures 0.15) — 그게 GPU에서 답할 질문이다.
- 실행 순서: **K1–K2**(스텝만 늘린 대조군) ‖ **K3–K4**(aim assist + 약한 상대, 진짜 기대) → **K5**(상대 강도 커리큘럼) →
  **L**(auto-pickup 제거) → 선택 **M**(armed 천장). 보상은 `configs/race_v2.json` 세 항으로 동결, 결과 전에는 항을 더하지 않는다.
- 이 PPO 트랙은 계획상 CPC의 두뇌가 아니라 **측정 가능한 상대/강도 기준선**이다. K에서 합성이 안 나와도 강도 노브가 달린 상대는
  이미 손에 있다; 그때는 2주차 System-1 스킬 + SLM으로 넘어간다(§10).

## 1. 브랜치 상태 (push는 Windows 머신에서 직접)

| repo | branch | head | 내용 |
|---|---|---|---|
| survev | `feature/cpc-dev` | `d80b5946` | named inputs(`5f118a6c`), 랜덤 레이아웃(`d10a1f71`), 레이스 목표(`a0bb89a3`), 팀 스킨(`170e7bd8`), `racer`(`4d93b59d`), README(`20b5b0a0`), **scriptedOptions 강도 노브(`d80b5946`)** |
| evolutionary-ai-battle | `feat/survev-rl` | 이 문서를 담은 커밋 | 정책 서버, goal 관측, 보상항(goal/capture/gun_pickup), `--layout/--objective/--scripted racer`, **`--aim-assist`, `--opp-*`, captures/armed 지표 노출**, 프리셋, 테스트 69개 |
| evolutionary-ai-battle | `docs/cpc-action-plan` | `a2c2fd4` | 액션플랜 + PPO 반복 1–5 로그 + 세션 브리프 |

주의할 점.

- 서버는 Windows 개발 머신과 다른 기계다. **Windows에서 두 브랜치를 push → 서버에서 clone/checkout**(§2). Windows 워킹트리의 특이 사항(`master` 체크아웃 + untracked 사본, `git checkout -f feat/survev-rl` 필요)은 서버의 새 clone에는 해당 없다.
- 브리지는 survev `d80b5946`이어야 `scriptedOptions`를 안다(그 전 브리지는 옵션을 조용히 무시 → 상대가 정확 조준). `5f118a6c` 이전은 named input도 버린다.
- Windows 워킹트리는 CRLF, 저장은 LF다. Linux 서버에서 커밋할 일이 생기면 그냥 LF로 쓰면 되고, Windows 쪽에서 리눅스 셸(VM)로 커밋할 때만 `git -c core.autocrlf=true add --renormalize`가 필요하다.
- 컨테이너 체크포인트 사본은 git에 없다(untracked `C:\repos\evolutionary-ai-battle\out\runs\`). 서버에서 warm start가 필요하면 `scp -r`로 옮긴다(§9).

## 2. 서버 환경 준비 (한 번)

```bash
# node >= 20.19 + pnpm (survev package.json: pnpm@10). nvm 등으로 node 22 권장
corepack enable && corepack prepare pnpm@10.33.0 --activate
node -v && pnpm -v

# survev
git clone <survev origin> ~/repos/survev && cd ~/repos/survev && git checkout feature/cpc-dev
pnpm install
(cd tests && pnpm test src/cpc_dev)                 # 32 tests (bridgeServer.test 포함 — uWebSockets.js 바이너리는 최근 glibc에서 뜬다)

# evolutionary-ai-battle + Python env (torch는 서버 드라이버에 맞는 CUDA 휠로)
git clone <eab origin> ~/repos/evolutionary-ai-battle && cd ~/repos/evolutionary-ai-battle && git checkout feat/survev-rl
conda create -n agentic-ai python=3.11 -y && conda activate agentic-ai     # 기존 env가 있으면 그것을 써도 된다
pip install torch --index-url https://download.pytorch.org/whl/cu128        # nvidia-smi 의 CUDA 버전에 맞출 것
pip install numpy "websockets>=13" pytest
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -m pytest experiment/survev_rl/tests -q     # 69 passed

# 스모크 (게임 서버 없이 30초; --device cuda 가 잡히는지까지 확인)
python -m experiment.survev_rl.train_ppo --mock --n-envs 8 --total-steps 20000 --objective race --scripted racer \
  --reward-json experiment/survev_rl/configs/race_v2.json --auto-pickup --aim-assist --opp-aim-noise 10 --device cuda --out runs/mock_smoke
```

Python은 리포 루트에서 `python -m experiment.survev_rl.…`로 실행한다(`experiment.core.cpc_actions` import 때문). 정책은 224→256→256 MLP라 GPU 메모리 1 GB 미만이고, 학습 프로세스 4개가 GPU 0 하나를 같이 써도 된다(`CUDA_VISIBLE_DEVICES`로 나눠도 된다).

## 3. 브리지 기동 — 학습 프로세스 1개당 브리지 1개

브리지는 TS 단일 프로세스이고 한 커넥션의 env 전부를 그 프로세스 안에서 시뮬레이션한다. **병목은 GPU가 아니라 브리지 CPU**다.
컨테이너(2 vCPU) 기준 `--n-envs 8`에서 900–1400 agent-step/s → 600k ≈ 10 분, 4M ≈ 50–70 분. 서버 코어가 더 빠르면 그 이하.

```bash
cd ~/repos/survev
for p in 8765 8766 8767 8768; do
  nohup pnpm cpc:bridge -- --port=$p > /tmp/bridge_$p.log 2>&1 &
done
sleep 5; grep -h "listening" /tmp/bridge_*.log      # CPC bridge listening on ws://127.0.0.1:876x (protocol v0)
ss -ltnp | grep 876                                  # 포트 확인. 종료: fuser -k 8765/tcp
```

- 브리지 수는 (`nproc` ÷ 2) 정도가 안전하다 — 학습 프로세스도 CPU를 쓴다. 코어가 넉넉하면 K1–K4를 동시에, 아니면 2개씩. `CPC_BRIDGE_PORT` 환경변수는 `--port`와 같은 뜻.
- TS 코드를 고치면 해당 브리지를 재시작해야 반영된다. "could not listen"이면 이전 브리지가 아직 살아 있는 것(`fuser -k <port>/tcp`).
- 브리지 하나에 학습 하나. 두 학습이 한 브리지에 붙으면 env id가 겹쳐 에피소드가 섞인다.
- 정책 서버(렌더용) 기본 포트도 8766이므로 렌더를 같이 할 때는 정책 서버를 `--port 8770`처럼 옮긴다.
- 오래 걸리는 것은 `tmux`(또는 `nohup … &` + `tail -f`)로 돌린다. 아래 커맨드는 전부 `nohup … &` 형태로 적었다.

## 4. 학습 계획

공통: `race_v2`(capture 1.0, gun_pickup 1.0, damage_dealt 0.02), `--layout random --objective race --scripted racer`, 60 초 고정
(`--objective race`가 `endOnElimination: false`와 목표 관측 블록을 자동으로 켠다), `--n-envs 16`, `--ticks 10`. 시드는 torch/numpy 시드이자
에피소드 시드의 베이스. 로그 한 줄은 `[ppo] upd 120/976 step 491520 sps 1100 … eps=200 win=0.05 surv=14.3s ret=1.9 tcap=0.31 armed=0.72`
(`tcap` = 최근 200 에피소드 평균 팀 캡처, `armed` = 총을 한 번이라도 든 비율). `sps`가 800 아래로 오래 머물면 그 브리지가 포화된 것 —
`--n-envs 8`로 내리고 브리지를 하나 더 띄운다.

```bash
conda activate agentic-ai
cd ~/repos/evolutionary-ai-battle
CFG=experiment/survev_rl/configs/race_v2.json
COMMON="--n-envs 16 --ticks 10 --layout random --objective race --scripted racer --reward-json $CFG --device cuda"
```

### K1–K2 — 대조군: 스텝만 늘린 J (aim assist 없음, 정확 조준 racer)

"스텝이 문제였나"에 답하는 run. 컨테이너 run J와 같은 설정에서 600k → 4M, env 8 → 16.

```bash
nohup python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8765 $COMMON --auto-pickup --total-steps 4000000 --seed 1 --out runs/k1_ctrl_s1 > runs/k1_ctrl_s1.log 2>&1 &
nohup python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8766 $COMMON --auto-pickup --total-steps 4000000 --seed 2 --out runs/k2_ctrl_s2 > runs/k2_ctrl_s2.log 2>&1 &
```

### K3–K4 — 본 실험: aim assist + 약한 racer(조준 노이즈 10°)

파일럿 K0의 연장. 상대 10°는 "이길 수 있지만 쉽지 않은" 지점(정확 조준 대비 TTK 6 → 14 s).

```bash
nohup python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8767 $COMMON --opp-aim-noise 10 --auto-pickup --aim-assist --total-steps 4000000 --seed 1 --out runs/k3_aim_n10_s1 > runs/k3_aim_n10_s1.log 2>&1 &
nohup python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8768 $COMMON --opp-aim-noise 10 --auto-pickup --aim-assist --total-steps 4000000 --seed 2 --out runs/k4_aim_n10_s2 > runs/k4_aim_n10_s2.log 2>&1 &
tail -f runs/k3_aim_n10_s1.log
```

### K5 — 상대 강도 커리큘럼 (K3 이어서: 10° → 5° → 정확)

`--total-steps`는 **누적** 기준이다(체크포인트의 update 번호에서 이어 가므로 4M에서 5M을 주면 1M을 더 돈다; `--n-envs`는 같은 16이어야 update 계산이 맞는다).
`--opp-aim-noise`를 빼면 정확 조준으로 돌아간다.

```bash
nohup python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8767 $COMMON --opp-aim-noise 5 --auto-pickup --aim-assist --total-steps 5000000 --seed 1 --resume runs/k3_aim_n10_s1/checkpoint_final.pt --out runs/k5a_aim_n5 > runs/k5a_aim_n5.log 2>&1 &
nohup python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8767 $COMMON --auto-pickup --aim-assist --total-steps 6000000 --seed 1 --resume runs/k5a_aim_n5/checkpoint_final.pt --out runs/k5b_aim_exact > runs/k5b_aim_exact.log 2>&1 &
```

K5b에서 `survival`과 `kills`가 K5a 대비 절반 아래로 무너지면 정확 조준 상대는 아직 이르다 — 5°에 1M 더 머문 뒤 다시 시도(`--opp-reaction 0.3`을 정확 조준의 중간 단계로 써도 된다).

### L — 보조 바퀴 떼기 (`--auto-pickup` 없이 resume, +2M)

K3–K5 중 가장 좋은 run을 이어서. `--aim-assist`는 유지한다(상대도 자동 조준이므로 최종 정책의 일부로 남긴다).

```bash
nohup python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8765 $COMMON --opp-aim-noise 10 --aim-assist --total-steps 6000000 --seed 1 --resume runs/k3_aim_n10_s1/checkpoint_final.pt --out runs/l_noassist > runs/l_noassist.log 2>&1 &
```

L에서 `armed`가 무너지면(예: 0.7 → 0.2) 줍기를 행동으로 배운 게 아니라 auto-pickup에 기대고 있었던 것이다. 그때는 `gun_pickup`을 키우기보다
K3를 처음부터 `--auto-pickup` 없이 한 시드 다시 돌려 비교한다(J에서 줍기 자체는 1.3 초 만에 배웠으므로 가능성이 있다).

### M — 선택: 무장 천장 (armed, 같은 보상, aim assist)

"총이 이미 있으면 이 보상으로 레이스와 전투를 합성하는가"를 K와 분리해서 보는 대조군.

```bash
nohup python -m experiment.survev_rl.train_ppo --bridge ws://127.0.0.1:8768 $COMMON --opp-aim-noise 10 --loadout armed --aim-assist --total-steps 3000000 --seed 1 --out runs/m_armed > runs/m_armed.log 2>&1 &
```

### 판단 기준 (각 run의 마지막 50 만 스텝 평균, `progress.csv`)

핵심 비교는 **같은 스텝에서 K3 vs K1**이다. K1이 K3를 따라오면 조준이 아니라 스텝이 문제였던 것이고, K3만 앞서면(예상) 조준 비대칭이 원인이었던 것.

| 관찰 | 해석 | 다음 행동 |
|---|---|---|
| K1–K2 `kills` ≈ 0 그대로 | 스텝은 답이 아니었다(예상) | K1–K2는 1M쯤에서 끊어도 된다(`kill <pid>`; 체크포인트는 10 update마다 저장됨) |
| K3–K4 `kills` ≥ 0.3, `survival` ≥ 25 s, `team_captures` < 0.5 | K0의 연장: 싸우지만 레이스 안 함 | `capture` 1.0 → 2.0 한 시드 재실행: `--reward-json experiment/survev_rl/configs/race_v2_cap2.json` |
| K3–K4 `team_captures` ≥ 1.5 이고 `kills` ≥ 0.3 | 합성 성립 | K5 → L로, 렌더로 확인 |
| `armed` < 0.5 (1M 이후에도) | 줍기가 자리 잡지 못함 | `gun_pickup` 1.0 → 2.0 한 시드 재실행 (`race_v2.json` 사본에서 그 항만 고칠 것 — 아래 주의) |
| `survival_time` < 10 s 가 계속 | 10 초 안에 racer에게 죽음 | 렌더(§7)로 왜 죽는지 본 뒤 결정 — 보상 손대지 말 것 |
| 시드 2개가 같은 행동 | 재현됨 | 그 행동을 결과로 기록 |

**`--reward-json`은 항을 덮어쓰는 게 아니라 `RewardConfig` 기본값 위에 얹는다.** 기본값은 v0 보상(`alive_per_step` 0.01,
`hp_delta` 0.01, `death` −1.0, `team_win` 1.0)이고 `race_v2.json`은 그 넷을 **명시적으로 0으로** 둔다. 그래서 인라인으로 세 항만 주면
7항 보상이 조용히 돌고(적립되는 생존 보상 + 죽음 페널티가 다시 붙는다) 3항 동결 원칙이 깨지며 K3/K4와 비교도 안 된다. 항을 바꿀 때는
인라인이 아니라 **`race_v2.json` 사본을 만들어 그 항만 고친다**(`race_v2_cap2.json`이 그 예). 실제로 확인하는 방법은
`runs/<run>/config.json`의 `meta.reward`에서 0이 아닌 항이 셋뿐인지 보는 것이다.

기준값 근거: 스크립트 racer 혼자면 포인트 하나에 ~4 초(60 초에 ~14개)이고, 지금까지 우리 팀 최고는 chaser 상대 2.8(run F+), racer 상대 1.6(run H),
K0 eval에서 racer 팀은 에피소드당 1.1개를 가져갔다. `damage_dealt` 20은 ak47 기준 2–3발 명중이다.

## 5. 평가

K/L/M 각각 끝나면 (a) 학습 때 상대 그대로 50 에피소드, (b) `--opp-exact`(정확 조준 racer)로 50 에피소드 = 전이 시험, (c) chaser 상대 50 에피소드.
에피소드 시드는 학습과 겹치지 않게 1000부터(기본값).

```bash
CK=runs/k3_aim_n10_s1/checkpoint_final.pt
python -m experiment.survev_rl.eval --checkpoint $CK --bridge ws://127.0.0.1:8765 --n-envs 4 --episodes 50 --out runs/k3_aim_n10_s1/eval_trained.jsonl
python -m experiment.survev_rl.eval --checkpoint $CK --bridge ws://127.0.0.1:8765 --n-envs 4 --episodes 50 --opp-exact --out runs/k3_aim_n10_s1/eval_exact.jsonl
python -m experiment.survev_rl.eval --checkpoint $CK --bridge ws://127.0.0.1:8765 --n-envs 4 --episodes 50 --scripted chaser --opp-exact --out runs/k3_aim_n10_s1/eval_chaser.jsonl
python -m experiment.survev_rl.eval --policy random --bridge ws://127.0.0.1:8765 --n-envs 4 --episodes 20 --layout random --scripted racer --out runs/eval_random_racer.jsonl   # 바닥선
```

eval은 체크포인트 meta에서 featurizer/goal/objective/layout/auto_pickup/aim_assist/scripted_options를 그대로 복원하므로 학습 플래그를 다시 줄 필요가 없다
(`--scripted`, `--layout`, `--loadout`, `--opp-*`, `--opp-exact`만 오버라이드). 콘솔 요약에 `captures / team_captures / armed_rate / gun_pickup_median` 줄이
추가로 나오고 같은 값이 `*.summary.json`에 들어간다. `--deterministic`(argmax)은 샘플링 정책과 행동이 꽤 달라질 수 있으니 둘 다 본다.

`eval*.jsonl`에는 네 에이전트 전부의 metrics가 있으므로 상대 팀 점수도 읽을 수 있다: `metrics["team-b-0"]["team_captures"]`. 점유율 = 우리 팀 캡처 / (우리 + 상대) 가 레이스 성패의 한 줄 요약이다.

## 6. 지표 읽는 법

- `runs/<run>/progress.csv` — update마다 한 줄, 최근 200 에피소드 창 평균. 볼 컬럼: `survival_time`, `damage_dealt`, `shots`, `hits_given`, `kills`,
  `captures`, `team_captures`, `armed`, `win_rate`, `episode_return`, `sps`, `entropy`.
- `runs/<run>/episodes.jsonl` — 에이전트·에피소드당 한 줄. 위 지표에 더해 `gun_pickup_time`(-1 = 끝까지 맨손), `reason`(`time_limit` | `controlled_dead` | `elimination`), `seed`.
- `runs/<run>/eval*.summary.json` — §5의 요약. `eval*.jsonl`은 스텝별 obs 요약·행동(`cpc.aim`에 스냅된 조준이 들어감)·보상·이벤트까지 있어 렌더 없이도 "언제 총을 들고 언제 죽었나"를 볼 수 있다.

빠른 확인용(마지막 2000 에피소드):

```bash
python - <<'EOF'
import json, collections
rows = [json.loads(l) for l in open("runs/k3_aim_n10_s1/episodes.jsonl")][-2000:]
m = lambda k: sum(r.get(k, 0) for r in rows) / len(rows)
print({k: round(m(k), 2) for k in ("survival_time", "damage_dealt", "shots", "hits_given", "kills", "captures", "team_captures", "armed")})
print(collections.Counter(r["reason"] for r in rows))
t = sorted(r["gun_pickup_time"] for r in rows if r.get("gun_pickup_time", -1) >= 0)
print("pickup median", t[len(t) // 2] if t else None)
EOF
```

## 7. 눈으로 확인 (렌더 킷)

숫자가 애매하면 한 에피소드를 실제 클라이언트로 찍는다. 킷은 Windows 머신의 `C:\repos\survev\.tmp\cpc_dev\live\`에 두었고(untracked, 커밋 금지; 서버에서 하려면 폴더째 복사)
순서는 그 안의 `README-live.md`에 있다. 컨테이너에서는 Linux + headless Chromium으로 같은 절차를 돌렸으니 서버에서도 된다. 요약:

1. `liveHook.ts`를 `server/src/cpc_dev/`에 복사하고 `server/src/game/gameProcess.ts` 첫 줄에 `import "../cpc_dev/liveHook.ts";` (둘 다 커밋하지 않는다), `survev-config.hjson`을 리포 루트에.
2. 정책 서버: `python -m experiment.survev_rl.policy_server --checkpoint runs/k3_aim_n10_s1/checkpoint_final.pt --port 8770` (aim assist는 체크포인트 meta로 자동 적용).
3. `pnpm dev:api`, `pnpm dev:client`, 그리고 env를 준 `pnpm dev:game`: `CPC_SPEED=0.2 CPC_LAYOUT=random CPC_OBJECTIVE=race CPC_SCRIPTED=racer CPC_SCRIPTED_OPTIONS='{"aimNoiseDeg":10}' CPC_SEED=cpc-duo2v2-seed-101 CPC_POLICY_URL=ws://127.0.0.1:8770 CPC_STATUS_PATH=… CPC_EPISODE_PATH=… pnpm dev:game` (상대 강도를 학습 때와 맞추는 것이 `CPC_SCRIPTED_OPTIONS`).
4. 브라우저(또는 `shoot_ep2.py`의 headless Chromium: `pip install playwright pillow && playwright install chromium`)로 Duo 입장 → 사람은 카메라, team-a는 정책, team-b는 racer. `shoot_ep2.py`가 0.5 초 간격 프레임을 `CPC_FRAMES`에 저장.
5. `python contact_sheet.py <frames> <live_episode.json> <out.png> "<제목>"` — 파랑 = team-a(정책), 빨강 = team-b, 캡션에 HP·행동·캡처 수.

지금까지 찍은 것(Windows `out\`): `ep2`(v0 도주), `ep2_armed`(스폰 방향 착취), `ep3`(point_v1 돌진), `ep4`(point_v2 카이팅), `ep5`(race_v1 레이스), `ep6`(racer 상대), `ep7`(race_v2, 총은 줍지만 배회). 게임 프로세스는 에피소드마다 재시작한다.

## 8. 알려진 이슈와 교훈

- **좌표**: `MOVE_LABELS`는 하네스 이름 그대로 화면 y-down이다(`up` = (0, −1)). survev 월드에서는 y가 화면 위로 자라므로 라벨 `up_left`는 월드 남서쪽 이동이다. 결정 로그를 읽을 때는 이름이 아니라 벡터를 본다.
- **비결정성**: 같은 시드라도 루트 드리프트(물리, `Math.random`)와 넷싱크 타이밍 때문에 프레임 단위로는 달라진다(상대의 조준 노이즈 자체는 시드로 결정적). 지표는 항상 에피소드 여러 개의 평균으로 본다.
- **보상 교훈**(A–E에서 지불한 수업료): 잠재함수형 진행 보상은 첫 2 초에 은행에 넣고 죽는 최적해를 만든다; 죽음이 페널티 스트림을 끝내면 자살 돌진이 최적해가 된다; 레이스 목표는 남은 포인트를 몰수하는 방식으로 죽음의 비용을 암묵적으로 만들기 때문에 `death` 항이 필요 없다. 새 항을 넣고 싶을 때 이 셋에 걸리는지 먼저 본다.
- **행동 공간 교훈**(J → K0): 보상이 아니라 원시 동작의 비대칭이 학습을 막고 있었다. 상대가 가진 원시 동작(정확 조준)을 정책에도 주어야 비교가 "결정"에 대한 것이 된다. 나중에 System-1 스킬을 설계할 때도 같은 원칙 — 스킬 계층의 원시 동작은 상대와 대칭이어야 한다.
- **auto-pickup은 커리큘럼 보조 장치**다(닿는 루트에 `Interact`를 자동으로 누름). 최종 정책은 L처럼 이것 없이 검증한다. aim assist는 반대로 최종 정책의 일부로 남긴다(상대도 자동 조준).
- **정책 서버/eval은 체크포인트 meta로 자기 설정을 복원**한다(featurizer·goal·action space·auto_pickup·aim_assist·scripted_options). 학습 코드를 바꿔 obs 차원이 달라지면 옛 체크포인트는 못 읽는다 — 그럴 때는 새로 학습.
- `pgrep -f train_ppo` 류는 자기 셸까지 잡는다(컨테이너에서 두 번 당함). 프로세스는 포트(`ss -ltnp`, `fuser -k <port>/tcp`)나 `nohup`이 남긴 PID로 관리.

## 9. 지금까지 결과 (2 코어, `--n-envs 8`; 자세한 표는 README)

| run | 설정 | steps | survival | captures/team | shots | hits/shot | dmg | kills | armed | 행동 |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 기본 보상, fists, chaser | 600k | 19.8 s | – | 0 | – | 0 | 0 | – | 남서쪽으로 도주, 구석에서 사망 |
| v0 armed | fixed layout | 600k | 6 s | – | 19 | 0.45 | 120 | – | (armed) | 동쪽 조준 고정 사격 = 스폰 방향 착취, 승 91 % |
| A–D `point_v1*` | random layout | 400k | 4–5 s | hold 0.4 | 1–17 | ~0.1 | 0.5–32 | 0 | – | 포인트로 돌진해 그 위에서 사망 |
| E `point_v2` armed | | 400k | 17.7 s | hold 0 | 67 | 0.08 | 68 | 0 | (armed) | 카이팅 사수, 포인트 무시, 승 18 % |
| F/F+ `race_v1` | fists, chaser | 600k/2M | 9–11 s | 1.0/2.0 → 1.4/2.8 | 1 | 0.05 | 0.7 | 0 | ~0 | 스크립트 속도로 레이스, 총 없음 |
| G `race_v1` armed | chaser | 600k | 10.2 s | 0.03/0.06 | 32 | 0.12 | 46 | 0.09 | (armed) | 싸우기만, 포인트 무시 |
| H `race_v1` | racer, auto-pickup | 600k | 9.6 s | 0.8/1.6 | 3 | 0.05 | 0.9 | 0 | 낮음 | 레이스, 가끔 사격, 승 56 % |
| I resume G | racer, auto-pickup | +600k | 15.8 s | 0.13/0.27 | 15 | 0.07 | 8.8 | 0 | (armed prior) | 카이팅 유지, 레이스 재학습 안 됨 |
| J `race_v2` | racer, auto-pickup, +gun_pickup | 600k | 14.3 s | 0.02/0.04 | 20 | 0.06 | 6.4 | 0 | 0.69 (중앙값 1.3 s) | 총은 줍고 싸우고 배회, 레이스 안 함 |
| **K0 `race_v2`** | **racer 10°, auto-pickup, aim assist** | **300k** | **24.4 s** | 0.08/0.15 | 20 | **0.25** | **48** | **0.18** | 0.67 | 줍고 싸운다(첫 킬), 레이스는 아직 |

K0를 정확 조준 racer로 평가하면 생존 13.5 s, 데미지 49, kills 0.15 — 10°에서 배운 것이 정확 조준 상대에게도 절반쯤 전이된다.

컨테이너 체크포인트 사본(`checkpoint_final.pt` + `config.json` + `progress.csv`)은 Windows 머신 `C:\repos\evolutionary-ai-battle\out\runs\`에 있다
(`race_v2_aim_pilot`=K0, `race_v2_racer`=J, `race_v1_racer`=H, `race_v1_2m`=F+, `race_v1_armed`=G, `point_v2_armed`=E, `ppo_ep2_armed`=v0 armed). untracked라 git으로는 안 오므로
서버에서 쓰려면 `scp -r "<windows>:/c/repos/evolutionary-ai-battle/out/runs" ~/repos/evolutionary-ai-battle/out/`. K3를 처음부터 돌리지 않고 K0에서 이어 가려면
`--resume out/runs/race_v2_aim_pilot/checkpoint_final.pt --n-envs 8`(K0는 env 8로 돌았으므로 update 계산을 맞추려면 8 유지).

## 10. K–L 다음

- PPO 결과는 액션플랜의 "상대/강도 베이스라인"이다: `eval.jsonl`의 metric vector(survival, HP, damage, team win, partner survival, captures)를 하네스 비교에 그대로 넣고,
  `scriptedOptions`(조준 노이즈·반응 지연)가 강도 축이 된다 — 정확 조준 = 최강, 20° = 초심자.
- 2주차 본론은 TS System-1 스킬(goto / loot / engage / retreat)과 SLM 플래너다. 이번 반복에서 나온 `racer`·`chaser` 스크립트, `RaceObjective`,
  aim assist가 스킬 계층의 첫 재료(`goto(point)`, `engage(dist)`, 조준 원시 동작)로 그대로 쓰인다. 계획은 `docs/cpc-survev-slm-action-plan.md`의 진행 로그와 `docs/cpc-session-brief-2026-09-10.md`를 따른다.
