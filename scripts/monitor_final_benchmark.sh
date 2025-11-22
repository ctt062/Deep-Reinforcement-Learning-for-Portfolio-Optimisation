#!/bin/bash
# Monitor Final Benchmark Training Progress

echo "============================================================"
echo "  Final Benchmark Training Monitor"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# Check if training process is running
TRAINING_PID=$(ps aux | grep "train_and_evaluate_final.py" | grep -v grep | awk '{print $2}')

if [ -z "$TRAINING_PID" ]; then
    echo "⚠️  Training process not found - may have completed or failed"
    echo ""
else
    echo "✓ Training process running (PID: $TRAINING_PID)"
    
    # Get CPU and memory usage
    CPU=$(ps -p $TRAINING_PID -o %cpu | tail -1 | tr -d ' ')
    MEM=$(ps -p $TRAINING_PID -o %mem | tail -1 | tr -d ' ')
    ELAPSED=$(ps -p $TRAINING_PID -o etime | tail -1 | tr -d ' ')
    echo "  CPU: ${CPU}% | Memory: ${MEM}% | Elapsed: ${ELAPSED}"
    echo ""
fi

# Extract progress from log file
LOG_FILE="logs/final_benchmark_training.log"

echo "============================================================"
echo "  Training Progress Estimation:"
echo "============================================================"

# Estimate progress based on elapsed time
if [ ! -z "$TRAINING_PID" ]; then
    # Get elapsed time (format: HH:MM:SS or MM:SS)
    ELAPSED_TIME=$(ps -p $TRAINING_PID -o etime | tail -1 | tr -d ' ')
    
    # Convert to seconds (remove leading zeros to avoid octal interpretation)
    if [[ $ELAPSED_TIME =~ ([0-9]+)-([0-9]+):([0-9]+):([0-9]+) ]]; then
        # Days format: D-HH:MM:SS
        ELAPSED_SEC=$((10#${BASH_REMATCH[1]} * 86400 + 10#${BASH_REMATCH[2]} * 3600 + 10#${BASH_REMATCH[3]} * 60 + 10#${BASH_REMATCH[4]}))
    elif [[ $ELAPSED_TIME =~ ([0-9]+):([0-9]+):([0-9]+) ]]; then
        # Hours format: HH:MM:SS
        ELAPSED_SEC=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
    elif [[ $ELAPSED_TIME =~ ([0-9]+):([0-9]+) ]]; then
        # Minutes format: MM:SS
        ELAPSED_SEC=$((10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]}))
    else
        ELAPSED_SEC=0
    fi
    
    # Updated: 100k timesteps @ ~580/min = ~172 min per model = 10320 seconds per model
    # Total: 2 models = ~344 minutes = 20640 seconds
    TOTAL_EXPECTED=20640
    
    if [ ! -z "$ELAPSED_SEC" ]; then
        # Calculate percentage
        PERCENT=$((ELAPSED_SEC * 100 / TOTAL_EXPECTED))
        if [ $PERCENT -gt 100 ]; then
            PERCENT=100
        fi
        
        # Determine phase based on elapsed time (each model ~172 min = 10320 sec)
        if [ $ELAPSED_SEC -lt 10320 ]; then
            PHASE="Training DDPG (1/2) - 100k timesteps"
            MODEL_PERCENT=$((ELAPSED_SEC * 100 / 10320))
        elif [ $ELAPSED_SEC -lt 20640 ]; then
            PHASE="Training PPO (2/2) - 100k timesteps"
            ELAPSED_PPO=$((ELAPSED_SEC - 10320))
            MODEL_PERCENT=$((ELAPSED_PPO * 100 / 10320))
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
else
    echo "⚠️  Training process not running"
fi

# Check log file for actual output
if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    echo ""
    echo "Recent output:"
    echo "------------------------------------------------------------"
    grep -v "Gym has been unmaintained" "$LOG_FILE" | grep -v "^$" | tail -10
    echo "------------------------------------------------------------"
fi
echo ""

# Check for model files
echo "============================================================"
echo "  Model Files Status:"
echo "============================================================"
if [ -d "models" ]; then
    ls -lh models/*.zip 2>/dev/null && echo "" || echo "No model files yet"
else
    echo "models/ directory not created yet"
fi
echo ""

# Check for results
echo "============================================================"
echo "  Results Status:"
echo "============================================================"
if [ -d "results" ]; then
    echo "Results files:"
    ls -lh results/*.json 2>/dev/null | tail -10 || echo "No result files yet"
else
    echo "results/ directory not created yet"
fi
echo ""

# Show summary if training complete
if [ -f "results/ddpg_options_final_metrics.json" ] && [ -f "results/ppo_options_final_metrics.json" ]; then
    echo "============================================================"
    echo "  ✅ TRAINING COMPLETED!"
    echo "============================================================"
    echo ""
    echo "Quick Results Preview:"
    echo ""
    
    echo "DDPG Metrics:"
    python3 -c "
import json
with open('results/ddpg_options_final_metrics.json') as f:
    m = json.load(f)
    print(f\"  Sharpe Ratio: {m['sharpe_ratio']:.4f}\")
    print(f\"  Total Return: {m['total_return']*100:.2f}%\")
    print(f\"  Max Drawdown: {m['max_drawdown']*100:.2f}%\")
    print(f\"  Volatility: {m['volatility']*100:.2f}%\")
" 2>/dev/null || echo "  (Error reading metrics)"
    
    echo ""
    echo "PPO Metrics:"
    python3 -c "
import json
with open('results/ppo_options_final_metrics.json') as f:
    m = json.load(f)
    print(f\"  Sharpe Ratio: {m['sharpe_ratio']:.4f}\")
    print(f\"  Total Return: {m['total_return']*100:.2f}%\")
    print(f\"  Max Drawdown: {m['max_drawdown']*100:.2f}%\")
    print(f\"  Volatility: {m['volatility']*100:.2f}%\")
" 2>/dev/null || echo "  (Error reading metrics)"
    
    echo ""
    echo "============================================================"
    echo "  Next Step: Generate Visualizations"
    echo "============================================================"
    echo "  python scripts/visualize_benchmark_comparison.py"
    echo "============================================================"
else
    echo "============================================================"
    echo "  Training still in progress..."
    echo "  Run this script again to check status"
    echo "============================================================"
fi

echo ""
