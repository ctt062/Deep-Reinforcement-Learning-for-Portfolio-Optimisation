#!/bin/bash
# Monitor training progress for all three agents

echo "======================================================================"
echo "Training Progress Monitor - Diversified Portfolio V2"
echo "======================================================================"
echo ""

# Check if processes are running
ppo_running=$(ps aux | grep "train.py.*ppo.*config_diversified_v2" | grep -v grep | wc -l)
ddpg_running=$(ps aux | grep "train.py.*ddpg.*config_diversified_v2" | grep -v grep | wc -l)
dqn_running=$(ps aux | grep "train.py.*dqn.*config_diversified_v2" | grep -v grep | wc -l)

echo "Process Status:"
echo "  PPO:  $(if [ $ppo_running -gt 0 ]; then echo '✓ Running'; else echo '✗ Stopped'; fi)"
echo "  DDPG: $(if [ $ddpg_running -gt 0 ]; then echo '✓ Running'; else echo '✗ Stopped'; fi)"
echo "  DQN:  $(if [ $dqn_running -gt 0 ]; then echo '✓ Running'; else echo '✗ Stopped'; fi)"
echo ""

# PPO Progress
echo "======================================================================"
echo "PPO Training Progress"
echo "======================================================================"
if [ -f logs/ppo_v2_training.log ]; then
    tail -25 logs/ppo_v2_training.log | grep -E "Steps:|total_timesteps|ep_rew_mean|Training completed" | tail -5
else
    echo "Log file not found"
fi
echo ""

# DDPG Progress
echo "======================================================================"
echo "DDPG Training Progress"
echo "======================================================================"
if [ -f logs/ddpg_v2_training.log ]; then
    tail -25 logs/ddpg_v2_training.log | grep -E "Steps:|total_timesteps|ep_rew_mean|Training completed" | tail -5
else
    echo "Log file not found"
fi
echo ""

# DQN Progress
echo "======================================================================"
echo "DQN Training Progress"
echo "======================================================================"
if [ -f logs/dqn_v2_training.log ]; then
    tail -25 logs/dqn_v2_training.log | grep -E "Steps:|total_timesteps|ep_rew_mean|Training completed" | tail -5
else
    echo "Log file not found"
fi
echo ""

# Check for completed models
echo "======================================================================"
echo "Completed Models"
echo "======================================================================"
ls -lh models_diversified_v2/*_final.zip 2>/dev/null | awk '{print $9}' | xargs -I {} basename {} || echo "No completed models yet"
echo ""

echo "======================================================================"
echo "Training will continue in background. Run this script again to check progress."
echo "======================================================================"
