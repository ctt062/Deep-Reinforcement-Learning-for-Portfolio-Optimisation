#!/bin/bash
# Monitor training progress for daily frequency with stop-loss configuration

echo "============================================================"
echo "Training Monitor - Daily Frequency with Stop-Loss"
echo "============================================================"
echo "Configuration: 18 assets, daily frequency, tiered stop-loss"
echo "Target: Sharpe > 1.0, Max Drawdown < 10%"
echo "Stop-loss: Tiered at 2%, 4%, 6%, 8%, 10% drawdown"
echo "============================================================"
echo ""

# Check if processes are running
echo "=== Training Processes ==="
ppo_proc=$(ps aux | grep "[t]rain_with_options.*ppo.*daily_stoploss" | wc -l)
ddpg_proc=$(ps aux | grep "[t]rain_with_options.*ddpg.*daily_stoploss" | wc -l)

if [ $ppo_proc -gt 0 ]; then
    echo "✓ PPO training is RUNNING"
    ppo_cpu=$(ps aux | grep "[t]rain_with_options.*ppo.*daily_stoploss" | awk '{print $3}')
    ppo_mem=$(ps aux | grep "[t]rain_with_options.*ppo.*daily_stoploss" | awk '{print $4}')
    echo "  CPU: ${ppo_cpu}% | Memory: ${ppo_mem}%"
else
    echo "✗ PPO training is NOT running"
fi

if [ $ddpg_proc -gt 0 ]; then
    echo "✓ DDPG training is RUNNING"
    ddpg_cpu=$(ps aux | grep "[t]train_with_options.*ddpg.*daily_stoploss" | awk '{print $3}')
    ddpg_mem=$(ps aux | grep "[t]rain_with_options.*ddpg.*daily_stoploss" | awk '{print $4}')
    echo "  CPU: ${ddpg_cpu}% | Memory: ${ddpg_mem}%"
else
    echo "✗ DDPG training is NOT running"
fi

echo ""
echo "=== PPO Training Progress ==="
if [ -f logs/train_ppo_daily_stoploss.log ]; then
    lines=$(wc -l < logs/train_ppo_daily_stoploss.log)
    echo "Log lines: $lines"
    
    # Get latest timestep and reward
    latest=$(grep -E "Steps:|time_elapsed|total_timesteps" logs/train_ppo_daily_stoploss.log | tail -10)
    if [ -n "$latest" ]; then
        echo "Latest progress:"
        echo "$latest" | grep -E "Steps:|total_timesteps|time_elapsed" | tail -5
    fi
    
    echo ""
    echo "Recent rewards:"
    grep "ep_rew_mean" logs/train_ppo_daily_stoploss.log | tail -3
else
    echo "No log file found"
fi

echo ""
echo "=== DDPG Training Progress ==="
if [ -f logs/train_ddpg_daily_stoploss.log ]; then
    lines=$(wc -l < logs/train_ddpg_daily_stoploss.log)
    echo "Log lines: $lines"
    
    # Get latest timestep and reward
    latest=$(grep -E "Steps:|time_elapsed|total_timesteps" logs/train_ddpg_daily_stoploss.log | tail -10)
    if [ -n "$latest" ]; then
        echo "Latest progress:"
        echo "$latest" | grep -E "Steps:|total_timesteps|time_elapsed" | tail -5
    fi
    
    echo ""
    echo "Recent rewards:"
    grep "ep_rew_mean" logs/train_ddpg_daily_stoploss.log | tail -3
else
    echo "No log file found"
fi

echo ""
echo "=== Model Files ==="
if [ -d models_daily_stoploss ]; then
    echo "Saved models:"
    ls -lh models_daily_stoploss/ | grep -E "ppo_|ddpg_" | tail -10
else
    echo "No models directory yet"
fi

echo ""
echo "============================================================"
echo "Training Target: Sharpe > 1.0, Drawdown < 10%"
echo "Estimated completion: ~20-30 minutes per agent (500k timesteps)"
echo "============================================================"
