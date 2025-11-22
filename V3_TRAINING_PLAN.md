# V3 Optimization Training - In Progress

## Training Started: November 21, 2025

### Objective
Fix the issues from V1 results to achieve ALL targets simultaneously:
- **Sharpe Ratio > 1.0** ✓
- **Max Drawdown < 10%** ✓
- **Annualized Return > 15%** ✓

---

## V1 Results (Baseline - FAILED to meet all targets)

### PPO V1 (config_daily_stoploss.yaml)
- **Sharpe Ratio**: 1.0574 ✓ (Target: >1.0)
- **Max Drawdown**: 18.50% ✗ (Target: <10%)
- **Ann. Return**: 15.89% ✓ (Target: >15%)
- **Issue**: Drawdown too high despite tiered stop-loss (2%, 4%, 6%, 8%, 10%)

### DDPG V1 (config_daily_stoploss.yaml)
- **Sharpe Ratio**: 0.9356 ✗ (Target: >1.0)
- **Max Drawdown**: 9.50% ✓ (Target: <10%)
- **Ann. Return**: 14.18% ✗ (Target: >15%)
- **Issue**: Only 0.06 below Sharpe target, 0.82% below return target

---

## V3 Optimizations (TRAINING NOW)

### 1. PPO V3 (config_ppo_v3.yaml)
**Goal**: Reduce drawdown from 18.5% to <10%, maintain Sharpe >1.0

**Changes**:
- ✅ **Risk Penalty**: 1.0 → **1.5** (50% increase to penalize volatility)
- ✅ **Stop-Loss Threshold**: 5% → **3%** (more aggressive trigger)
- ✅ **Protective Puts**: 30% → **40%** hedge ratio
- ✅ **Put Strike**: 97% → **98%** (tighter protection)
- ✅ **Covered Calls**: Reduced 20% → **15%** to keep more upside
- ✅ **Call Strike**: 105% → **106%** (higher for more upside capture)

**Status**: TRAINING (61,440 / 500,000 steps = 12.3%)
- Reward: 2.79M (stable)
- ETA: ~15 minutes

### 2. DDPG V2 (config_ddpg_v2.yaml)
**Goal**: Increase Sharpe from 0.94 to >1.0, maintain DD <10%

**Changes**:
- ✅ **Risk Penalty**: 1.0 → **0.8** (20% reduction for higher returns)
- ✅ **Turnover Penalty**: 0.001 → **0.0005** (allow more trading)
- ✅ **Max Position Size**: 20% → **25%** (allow concentration in winners)
- ✅ **Protective Puts**: 30% → **25%** (reduce hedge costs)
- ✅ **Put Strike**: 97% → **96%** (looser/cheaper protection)
- ✅ **Covered Calls**: 20% → **25%** (more income generation)
- ✅ **Call Strike**: 105% → **104%** (more aggressive call selling)
- ✅ **Learning Rate**: 5e-5 → **1e-4** (faster learning)
- ✅ **Action Noise**: 0.1 → **0.15** (more exploration)
- ✅ **Tau**: 0.005 → **0.01** (faster target network updates)

**Status**: TRAINING (downloading data)
- ETA: ~20 minutes

### 3. Environment Changes (portfolio_env.py)
**More Aggressive Stop-Loss System**:

Previous (V1):
- 2% DD → 90% exposure
- 4% DD → 70% exposure
- 6% DD → 50% exposure
- 8% DD → 30% exposure
- 10% DD → 10% exposure

New (V3):
- **1% DD → 85% exposure** (earlier intervention)
- **2.5% DD → 65% exposure** (steeper cut)
- **4% DD → 45% exposure** (more aggressive)
- **6% DD → 25% exposure** (emergency level)
- **8% DD → 10% exposure** (extreme defensive)
- **10% DD → 5% exposure** (max protection)

**Impact**: Should prevent deep drawdowns by cutting exposure earlier and more aggressively.

---

## Expected Outcomes

### PPO V3 (Conservative)
- **Sharpe**: ~1.0-1.1 (slight decrease from 1.06 due to risk penalty)
- **Drawdown**: **8-12%** (targeting <10%, aggressive stop-loss should help)
- **Return**: ~14-16% (slight decrease due to higher risk penalty)

**Likelihood of Success**: 
- Meeting Sharpe target: **HIGH** (historically achieved 1.06)
- Meeting DD target: **MEDIUM** (need aggressive stop-loss to work)
- Meeting Return target: **MEDIUM** (may trade return for DD reduction)

### DDPG V2 (Aggressive Returns)
- **Sharpe**: **1.0-1.1** (targeting improvement from 0.94)
- **Drawdown**: 9-11% (may increase slightly from 9.5%)
- **Return**: **16-18%** (targeting increase from 14.2%)

**Likelihood of Success**:
- Meeting Sharpe target: **HIGH** (only need 0.06 improvement)
- Meeting DD target: **MEDIUM-HIGH** (was at 9.5%, small margin)
- Meeting Return target: **HIGH** (only need 0.82% improvement)

---

## Risk Assessment

### Potential Issues

1. **PPO V3**: Risk penalty 1.5 might be too conservative
   - Could suppress returns below 15% threshold
   - Mitigation: Aggressive stop-loss + options should compensate

2. **DDPG V2**: Reduced risk penalty (0.8) might increase DD
   - Could push DD from 9.5% to 11-12%
   - Mitigation: Aggressive stop-loss system should contain it

3. **Both**: New aggressive stop-loss could be too reactive
   - Cutting exposure at 1% DD might miss recoveries
   - Could lead to opportunity cost and lower returns

### Backup Plan
If both fail to meet all targets:
- Use PPO V3 result if Sharpe >1.0 and DD is closer to 10%
- Use DDPG V2 result if Sharpe >1.0 and DD <10%
- Create hybrid config: risk_penalty=1.2 (middle ground)

---

## Training Commands

```bash
# PPO V3
nohup python scripts/train_with_options.py --agent ppo --config configs/config_ppo_v3.yaml --output-dir models_ppo_v3 > logs/train_ppo_v3.log 2>&1 &

# DDPG V2
nohup python scripts/train_with_options.py --agent ddpg --config configs/config_ddpg_v2.yaml --output-dir models_ddpg_v2 > logs/train_ddpg_v2.log 2>&1 &
```

## Monitoring

```bash
# Real-time monitoring
./scripts/monitor_v3_realtime.sh

# Check logs
tail -f logs/train_ppo_v3.log
tail -f logs/train_ddpg_v2.log

# Check processes
ps aux | grep train_with_options
```

## Evaluation (after training completes)

```bash
# PPO V3
python scripts/evaluate_with_options.py --agent ppo --model-path models_ppo_v3/ppo_options_*_final.zip --config configs/config_ppo_v3.yaml --output results_ppo_v3

# DDPG V2
python scripts/evaluate_with_options.py --agent ddpg --model-path models_ddpg_v2/ddpg_options_*_final.zip --config configs/config_ddpg_v2.yaml --output results_ddpg_v2
```

---

## Timeline

- **Start**: 6:04 PM (PPO), 6:06 PM (DDPG)
- **Expected Completion**: ~6:20-6:25 PM
- **Evaluation**: ~6:30 PM
- **Results Available**: ~6:35 PM

---

**Status**: Training in progress... Check with `./scripts/monitor_v3_realtime.sh`
