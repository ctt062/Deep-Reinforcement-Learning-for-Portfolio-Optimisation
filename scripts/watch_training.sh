#!/bin/bash
# Continuous monitor with auto-refresh every 20 seconds

while true; do
    clear
    echo "============================================================"
    echo "  Final Benchmark Training Monitor (Auto-refresh: 20s)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""
    
    # Check if training process is running
    TRAINING_PID=$(ps aux | grep "train_and_evaluate_final.py" | grep -v grep | awk '{print $2}')
    
    if [ -z "$TRAINING_PID" ]; then
        echo "⚠️  Training process not found"
        
        # Check if completed
        if [ -f "results_final_benchmark/ddpg_options_final_metrics.json" ] && \
           [ -f "results_final_benchmark/ppo_options_final_metrics.json" ]; then
            echo "✅ Training COMPLETED!"
            break
        else
            echo "Training may have failed. Check logs/final_benchmark_training.log"
            break
        fi
    else
        echo "✓ Training running (PID: $TRAINING_PID)"
        
        # Get detailed stats
        CPU=$(ps -p $TRAINING_PID -o %cpu | tail -1 | tr -d ' ')
        MEM=$(ps -p $TRAINING_PID -o %mem | tail -1 | tr -d ' ')
        ELAPSED=$(ps -p $TRAINING_PID -o etime | tail -1 | tr -d ' ')
        echo "  CPU: ${CPU}% | Memory: ${MEM}% | Elapsed: ${ELAPSED}"
        
        # Convert elapsed time to seconds
        if [[ $ELAPSED =~ ([0-9]+)-([0-9]+):([0-9]+):([0-9]+) ]]; then
            ELAPSED_SEC=$((${BASH_REMATCH[1]} * 86400 + ${BASH_REMATCH[2]} * 3600 + ${BASH_REMATCH[3]} * 60 + ${BASH_REMATCH[4]}))
        elif [[ $ELAPSED =~ ([0-9]+):([0-9]+):([0-9]+) ]]; then
            ELAPSED_SEC=$((${BASH_REMATCH[1]} * 3600 + ${BASH_REMATCH[2]} * 60 + ${BASH_REMATCH[3]}))
        elif [[ $ELAPSED =~ ([0-9]+):([0-9]+) ]]; then
            ELAPSED_SEC=$((${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]}))
        else
            ELAPSED_SEC=0
        fi
        
        echo ""
        echo "============================================================"
        echo "  Training Progress:"
        echo "============================================================"
        
        # Estimate progress (30 min per model = 1800 sec)
        TOTAL_EXPECTED=3600
        PERCENT=$((ELAPSED_SEC * 100 / TOTAL_EXPECTED))
        if [ $PERCENT -gt 100 ]; then
            PERCENT=100
        fi
        
        # Determine phase
        if [ $ELAPSED_SEC -lt 1800 ]; then
            PHASE="Training DDPG (1/2)"
            MODEL_PERCENT=$((ELAPSED_SEC * 100 / 1800))
        elif [ $ELAPSED_SEC -lt 3600 ]; then
            PHASE="Training PPO (2/2)"
            ELAPSED_PPO=$((ELAPSED_SEC - 1800))
            MODEL_PERCENT=$((ELAPSED_PPO * 100 / 1800))
        else
            PHASE="Finalizing..."
            MODEL_PERCENT=100
        fi
        
        echo "📊 Current Phase: $PHASE"
        echo "  Model Progress: ~${MODEL_PERCENT}%"
        
        # Model progress bar
        FILLED=$((MODEL_PERCENT / 2))
        BAR=""
        for i in $(seq 1 50); do
            if [ $i -le $FILLED ]; then
                BAR="${BAR}█"
            else
                BAR="${BAR}░"
            fi
        done
        echo "  [${BAR}] ${MODEL_PERCENT}%"
        
        echo ""
        echo "📊 Overall Progress: ~${PERCENT}%"
        
        # Overall progress bar
        FILLED_OVERALL=$((PERCENT / 2))
        BAR_OVERALL=""
        for i in $(seq 1 50); do
            if [ $i -le $FILLED_OVERALL ]; then
                BAR_OVERALL="${BAR_OVERALL}█"
            else
                BAR_OVERALL="${BAR_OVERALL}░"
            fi
        done
        echo "  [${BAR_OVERALL}] ${PERCENT}%"
        
        # Time estimates
        ELAPSED_MIN=$((ELAPSED_SEC / 60))
        REMAINING_SEC=$((TOTAL_EXPECTED - ELAPSED_SEC))
        if [ $REMAINING_SEC -lt 0 ]; then
            REMAINING_SEC=0
        fi
        REMAINING_MIN=$((REMAINING_SEC / 60))
        
        echo ""
        echo "  ⏱️  Elapsed: ${ELAPSED_MIN} min | Remaining: ~${REMAINING_MIN} min"
    fi
    echo ""
    
    # Check model files
    echo "============================================================"
    echo "  Files Status:"
    echo "============================================================"
    if [ -d "models_final_benchmark" ]; then
        DDPG_MODEL=$(ls models_final_benchmark/ddpg*.zip 2>/dev/null | wc -l | tr -d ' ')
        PPO_MODEL=$(ls models_final_benchmark/ppo*.zip 2>/dev/null | wc -l | tr -d ' ')
        echo "  ✓ Models saved: DDPG=${DDPG_MODEL}/1, PPO=${PPO_MODEL}/1"
    else
        echo "  • Models: None yet"
    fi
    
    if [ -d "results_final_benchmark" ]; then
        RESULTS=$(ls results_final_benchmark/*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "  ✓ Results files: ${RESULTS}/6"
    else
        echo "  • Results: None yet"
    fi
    
    echo ""
    echo "============================================================"
    echo "  Press Ctrl+C to stop monitoring"
    echo "  Next refresh in 20 seconds..."
    echo "============================================================"
    
    sleep 20
done

# If training completed, show summary
if [ -f "results_final_benchmark/ddpg_options_final_metrics.json" ] && \
   [ -f "results_final_benchmark/ppo_options_final_metrics.json" ]; then
    echo ""
    echo "============================================================"
    echo "  📊 TRAINING COMPLETE - RESULTS SUMMARY"
    echo "============================================================"
    echo ""
    python3 -c "
import json

print('DDPG Results (2019-2020):')
with open('results_final_benchmark/ddpg_options_final_metrics.json') as f:
    m = json.load(f)
    print(f'  Sharpe Ratio:      {m[\"sharpe_ratio\"]:>8.4f}')
    print(f'  Total Return:      {m[\"total_return\"]*100:>7.2f}%')
    print(f'  Max Drawdown:      {m[\"max_drawdown\"]*100:>7.2f}%')
    print(f'  Volatility:        {m[\"volatility\"]*100:>7.2f}%')
    ddpg_sharpe = m['sharpe_ratio']

print()
print('PPO Results (2019-2020):')
with open('results_final_benchmark/ppo_options_final_metrics.json') as f:
    m = json.load(f)
    print(f'  Sharpe Ratio:      {m[\"sharpe_ratio\"]:>8.4f}')
    print(f'  Total Return:      {m[\"total_return\"]*100:>7.2f}%')
    print(f'  Max Drawdown:      {m[\"max_drawdown\"]*100:>7.2f}%')
    print(f'  Volatility:        {m[\"volatility\"]*100:>7.2f}%')
    ppo_sharpe = m['sharpe_ratio']

print()
print('Winner:', 'DDPG' if ddpg_sharpe > ppo_sharpe else 'PPO')
" 2>/dev/null || echo "Error reading metrics"
    
    echo ""
    echo "============================================================"
    echo "  Next Step: Generate Visualizations"
    echo "============================================================"
    echo "  python scripts/visualize_benchmark_comparison.py"
    echo "============================================================"
fi
