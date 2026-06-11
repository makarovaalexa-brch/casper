# Cell 15: RL Training with Parallel Episodes (10x FASTER)

from casper.training.parallel_rl_trainer import ParallelRLTrainer

# Configuration
PARALLEL_EPISODES = 10  # Run 10 episodes concurrently for 10x speedup

# Initialize parallel trainer
parallel_trainer = ParallelRLTrainer(
    agent=casper_agent,
    user_sim=user_sim,
    evaluator=evaluator,
    movie_catalog=movie_catalog,
    parallel_episodes=PARALLEL_EPISODES
)

# Check for existing checkpoints
start_episode = 0
resume_data = None

rl_checkpoints = sorted(CHECKPOINT_DIR.glob('rl_episode_*.pt'))
if rl_checkpoints:
    latest_checkpoint = rl_checkpoints[-1]
    print(f"Loading RL checkpoint from {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
    casper_agent.rl_agent.actor.load_state_dict(checkpoint['actor'])
    casper_agent.rl_agent.critic.load_state_dict(checkpoint['critic'])
    start_episode = checkpoint['episode']
    resume_data = {
        'rewards': checkpoint['rewards'],
        'ndcg': checkpoint['ndcg']
    }
    print(f"Resuming from episode {start_episode}")
    print(f"  Recent NDCG (last 10): {np.mean(checkpoint['ndcg'][-10:]):.3f}")

print(f"\nStarting parallel RL training...")
print(f"  Parallel episodes: {PARALLEL_EPISODES}")
print(f"  Expected speedup: ~{PARALLEL_EPISODES}x")
print(f"  Estimated time: {RL_EPISODES / PARALLEL_EPISODES * 6:.0f} minutes (vs {RL_EPISODES * 60:.0f} minutes sequential)")

# Train with parallelization
results = parallel_trainer.train(
    num_episodes=RL_EPISODES,
    max_turns=MAX_TURNS,
    checkpoint_dir=CHECKPOINT_DIR,
    start_episode=start_episode,
    resume_data=resume_data
)

episode_rewards = results['episode_rewards']
episode_ndcg = results['episode_ndcg']
train_losses = results['train_losses']

print(f"\nRL training complete")
print(f"  Episodes completed: {len(episode_rewards)}")
if len(episode_rewards) >= 100:
    print(f"  Final reward (last 100): {np.mean(episode_rewards[-100:]):.3f}")
    print(f"  Final NDCG (last 100): {np.mean(episode_ndcg[-100:]):.3f}")
else:
    print(f"  Final reward (all): {np.mean(episode_rewards):.3f}")
    print(f"  Final NDCG (all): {np.mean(episode_ndcg):.3f}")
