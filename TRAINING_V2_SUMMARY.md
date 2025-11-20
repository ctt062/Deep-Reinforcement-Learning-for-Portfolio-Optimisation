# Diversified Portfolio Training - Version 2

## Overview
This is an enhanced training run with improved configuration to achieve lower drawdown while maintaining good returns.

## Key Improvements from V1

### 1. Portfolio Composition (22 Assets - NO CRYPTO)
**Sector Diversification:**
- **Technology (15%)**: AAPL, MSFT, GOOGL
- **Healthcare (15%)**: JNJ, UNH, PFE  
- **Financials (10%)**: JPM, V
- **Consumer (10%)**: AMZN, WMT, PG
- **Energy (5%)**: XOM
- **Industrials (5%)**: BA, CAT
- **Broad Market ETFs (20%)**: SPY, QQQ, IWM
- **Fixed Income (15%)**: TLT, IEF, AGG
- **Commodities (5%)**: GLD, SLV

**Removed:** BTC-USD, ETH-USD (crypto assets that caused high volatility)

### 2. Enhanced Risk Management

**Reward Function:**
- `risk_penalty_lambda: 2.0` (increased from 0.5)
- Aggressive penalties for drawdowns
- Higher turnover penalties to reduce overtrading

**Position Constraints:**
- Max 25% in any single asset
- Min 2% to ensure diversification

### 3. Optimized Training Parameters

**All Agents:**
- Lower learning rates (5e-5 instead of 1e-4) for stability
- Deeper networks [512, 512, 256, 256]
- Larger batch sizes for better generalization
- 500k timesteps total

**PPO:**
- Higher entropy coefficient for better exploration
- Increased batch size to 256

**DDPG:**
- Extended warm-up period (20k steps)
- Reduced action noise (0.05)
- Larger batch size (512)
- Slower target network updates

**DQN:**
- Increased buffer size (500k)
- Larger batch size (256)
- More gradual exploration decay

## Training Status

### V1 Results (Baseline with 20 assets including crypto)
- **PPO**: Sharpe 0.8528, DD 16.28%, Return 12.21%
- **DDPG**: Sharpe 0.5104, DD 20.70%, Return 8.52%
- **DQN**: Sharpe 0.8562, DD 16.23%, Return 12.24%

**Best:** PPO and DQN with ~16% drawdown (still above 10% target)

### V2 Training (Current)
**Started:** November 19, 2025
**Configuration:** config_diversified_v2.yaml
**Models Directory:** models_diversified_v2/
**Results Directory:** results_diversified_v2/

**Training Progress:**
- PPO: ✓ Running (500k timesteps)
- DDPG: ✓ Running (500k timesteps)
- DQN: ✓ Running (500k timesteps)

**Expected Improvements:**
1. **Lower Drawdown**: Target < 15% (ideally < 10%)
2. **Better Risk-Adjusted Returns**: Higher Sharpe ratio
3. **More Stable Performance**: Reduced volatility
4. **Lower Turnover**: More consistent strategies

## Monitoring

Check training progress:
```bash
bash scripts/monitor_training_v2.sh
```

View logs:
```bash
tail -f logs/ppo_v2_training.log
tail -f logs/ddpg_v2_training.log  
tail -f logs/dqn_v2_training.log
```

## Automatic Evaluation

Once all three models finish training, run:
```bash
bash scripts/evaluate_when_ready_v2.sh
```

This will automatically:
1. Wait for all models to complete
2. Evaluate each model individually
3. Compare all models against benchmarks
4. Generate comprehensive visualizations

## Expected Timeline

Training duration (approximate):
- **PPO**: 2-3 hours
- **DDPG**: 6-8 hours  
- **DQN**: 1-2 hours

Total: ~8-10 hours for all three models

## Key Hypotheses

1. **Crypto Removal**: Should reduce drawdown by 10-15 percentage points
2. **Increased Risk Penalties**: Should lead to more conservative strategies
3. **Better Diversification**: 22 assets vs 20, more sector balance
4. **Fixed Income Allocation**: Should provide downside protection
5. **Deeper Networks**: Should capture more complex risk patterns

## Success Criteria

**Primary Goal:**
- Maximum Drawdown < 15% (stretch goal: < 10%)

**Secondary Goals:**
- Sharpe Ratio > 0.85
- Annualized Return > 12%
- Average Turnover < 5%

## Next Steps

1. Wait for training to complete (~8-10 hours)
2. Run automatic evaluation
3. Compare V2 vs V1 results
4. If drawdown still > 15%, consider:
   - Further increasing risk penalties
   - Adding volatility targeting
   - Implementing dynamic position sizing
   - Testing on different time periods

## Files Created

- `configs/config_diversified_v2.yaml` - Enhanced configuration
- `scripts/monitor_training_v2.sh` - Progress monitoring
- `scripts/evaluate_when_ready_v2.sh` - Automatic evaluation
- `logs/ppo_v2_training.log` - PPO training log
- `logs/ddpg_v2_training.log` - DDPG training log
- `logs/dqn_v2_training.log` - DQN training log

---

*Last Updated: November 19, 2025*
*Training Status: In Progress*
