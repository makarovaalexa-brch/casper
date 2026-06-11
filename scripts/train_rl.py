"""
CASPER RL Training Script

Trains the DDPG actor-critic agent using the Concept model accuracy reward signal.
Runs end-to-end episodes: agent asks questions → user simulator responds → reward computed.

Usage:
    python scripts/train_rl.py                    # Train from scratch or resume
    python scripts/train_rl.py --episodes 500     # Override episode count
    python scripts/train_rl.py --fresh             # Force fresh start (ignore checkpoints)

Monitor:
    python scripts/monitor_rl.py                   # (if exists)
    tail -f experiments/checkpoints/training_progress.csv
"""

import sys
import os
import json
import random
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')


# ─── Config ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='CASPER RL Training')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of RL episodes (default: from config)')
    parser.add_argument('--max-turns', type=int, default=None,
                        help='Max conversation turns per episode')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (ignore existing checkpoints)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Print summary every N episodes')
    parser.add_argument('--max-profiles', type=int, default=500,
                        help='Max user profiles to load')
    parser.add_argument('--min-eval-targets', type=int, default=10,
                        help='Min rated movies in pool per user')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_config():
    with open(PROJECT_ROOT / 'config' / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)


# ─── Components ──────────────────────────────────────────────────────────────

def init_components(config, args):
    """Initialize all training components."""
    DATA_DIR = PROJECT_ROOT / 'data' / 'movielens'
    CHECKPOINT_DIR = PROJECT_ROOT / 'experiments' / 'checkpoints'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("CASPER RL Training (Concept Model Accuracy Reward)")
    print("=" * 60)

    # Data loader
    print("\n[1/5] Loading MovieLens data...")
    from casper.data.movielens_loader import MovieLensLoader
    loader = MovieLensLoader(data_path=str(DATA_DIR), min_ratings=20)
    train_users, test_users = loader.load_data(test_split=0.3)
    print(f"  Train: {len(train_users)}, Test: {len(test_users)}")

    # Embedding space
    print("\n[2/5] Loading SBERT encoder...")
    from casper.models.embedding_space import SentenceBERTEmbeddingSpace
    embedding_space = SentenceBERTEmbeddingSpace(str(DATA_DIR))

    # Agent
    print("\n[3/5] Initializing CASPER agent...")
    from casper.agents.casper_agent import CASPERAgent
    agent = CASPERAgent(
        movielens_data_path=str(DATA_DIR),
        load_recommender=True,
        embedding_space=embedding_space,
    )
    agent.set_state_encoder_type('lstm')
    print(f"  State encoder: {agent.state_encoder_type}")

    # Load pretrained actor if available
    pretrain_ckpt = CHECKPOINT_DIR / 'actor_pretrained_lstm.pt'
    if pretrain_ckpt.exists():
        print(f"  Loading pretrained actor: {pretrain_ckpt.name}")
        ckpt = torch.load(pretrain_ckpt, map_location='cpu')
        agent.rl_agent.actor.load_state_dict(ckpt['actor'])
        if 'state_encoder' in ckpt and agent.rl_agent.state_encoder is not None:
            agent.rl_agent.state_encoder.load_state_dict(ckpt['state_encoder'])
        print(f"  Pretrained from epoch {ckpt.get('epoch', '?')}, "
              f"val_loss={ckpt.get('val_loss', '?'):.4f}")

    # User simulator
    print("\n[4/5] Loading user simulator...")
    from casper.agents.user_simulator import UserSimulator
    user_sim = UserSimulator(
        movielens_data_path=str(DATA_DIR),
        min_ratings=config['models']['user_simulator']['min_ratings'],
        max_profiles=args.max_profiles,
    )
    print(f"  {len(user_sim.user_profiles)} user profiles")

    # Reward calculator
    print("\n[5/5] Loading Concept model reward...")
    from casper.evaluation.concept_reward import ConceptModelRewardCalculator
    reward_calc = ConceptModelRewardCalculator(
        data_path=str(DATA_DIR), top_n_movies=100,
    )

    # Filter users
    before = len(user_sim.user_profiles)
    user_sim.user_profiles = reward_calc.filter_users_by_eval_targets(
        user_sim.user_profiles, min_eval_targets=args.min_eval_targets,
    )
    print(f"  Users: {before} -> {len(user_sim.user_profiles)} "
          f"(min {args.min_eval_targets} rated movies)")

    # Episode runner
    from casper.training.episode_runner import EpisodeRunner
    max_turns = args.max_turns or config['environment']['max_turns']
    runner = EpisodeRunner(
        reward_calculator=reward_calc,
        user_simulator=user_sim,
        max_turns=max_turns,
        holdout_ratio=0.0,
        exclude_mentioned=False,
        ndcg_weight=1.0,
        new_pref_bonus=0.05,
        not_seen_penalty=0.05,
        use_baseline=True,
    )

    return {
        'agent': agent,
        'user_sim': user_sim,
        'runner': runner,
        'reward_calc': reward_calc,
        'checkpoint_dir': CHECKPOINT_DIR,
        'max_turns': max_turns,
        'train_users': train_users,
        'test_users': test_users,
    }


# ─── Checkpoint helpers ──────────────────────────────────────────────────────

def save_checkpoint(path, agent, episode, rewards, acc, losses):
    checkpoint = {
        'actor': agent.rl_agent.actor.state_dict(),
        'critic': agent.rl_agent.critic.state_dict(),
        'actor_target': agent.rl_agent.actor_target.state_dict(),
        'critic_target': agent.rl_agent.critic_target.state_dict(),
        'episode': episode,
        'rewards': rewards,
        'acc': acc,
        'losses': losses,
        'state_encoder_type': agent.state_encoder_type,
        'reward_type': 'concept_accuracy',
        'reward_stats': {
            'mean': agent.rl_agent.reward_mean,
            'std': agent.rl_agent.reward_std,
            'count': agent.rl_agent.reward_count,
        },
        'noise_scale': agent.rl_agent.noise_scale,
    }
    if agent.rl_agent.state_encoder is not None:
        checkpoint['state_encoder'] = agent.rl_agent.state_encoder.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, agent):
    ckpt = torch.load(path, map_location='cpu')
    if ckpt.get('state_encoder_type', 'sbert') != agent.state_encoder_type:
        return None
    agent.rl_agent.actor.load_state_dict(ckpt['actor'])
    agent.rl_agent.critic.load_state_dict(ckpt['critic'])
    if 'actor_target' in ckpt:
        agent.rl_agent.actor_target.load_state_dict(ckpt['actor_target'])
    if 'critic_target' in ckpt:
        agent.rl_agent.critic_target.load_state_dict(ckpt['critic_target'])
    if 'state_encoder' in ckpt and agent.rl_agent.state_encoder is not None:
        agent.rl_agent.state_encoder.load_state_dict(ckpt['state_encoder'])
    if 'reward_stats' in ckpt:
        s = ckpt['reward_stats']
        agent.rl_agent.reward_mean = s.get('mean', 0.0)
        agent.rl_agent.reward_std = s.get('std', 1.0)
        agent.rl_agent.reward_count = s.get('count', 0)
    if 'noise_scale' in ckpt:
        agent.rl_agent.noise_scale = ckpt['noise_scale']
    return ckpt


def store_experiences(agent, result):
    states = result.get('states', [])
    embeddings = result.get('embeddings', [])
    rewards = result.get('rewards', [])
    if len(states) < 2 or len(embeddings) < 2:
        return
    for i in range(len(embeddings)):
        if i >= len(rewards):
            break
        next_state = states[i + 1] if i + 1 < len(states) else states[i]
        done = (i == len(embeddings) - 1)
        agent.rl_agent.store_experience(
            states[i], embeddings[i], rewards[i], next_state, done,
        )


# ─── Logging ─────────────────────────────────────────────────────────────────

def save_progress_csv(path, rewards, acc, losses):
    with open(path, 'w') as f:
        f.write('episode,reward,acc,loss,reward_ma10,acc_ma10\n')
        for i in range(len(rewards)):
            r_ma = np.mean(rewards[max(0, i - 9):i + 1])
            a_ma = np.mean(acc[max(0, i - 9):i + 1])
            lo = losses[i] if i < len(losses) else 0
            f.write(f'{i+1},{rewards[i]:.4f},{acc[i]:.4f},{lo:.4f},'
                    f'{r_ma:.4f},{a_ma:.4f}\n')


def save_stats_json(path, rewards, acc, losses):
    stats = {
        'num_episodes': len(rewards),
        'reward_type': 'concept_accuracy',
        'rewards': rewards,
        'acc': acc,
        'losses': losses,
    }
    if len(rewards) >= 10:
        stats['recent_reward_mean'] = float(np.mean(rewards[-10:]))
        stats['recent_acc_mean'] = float(np.mean(acc[-10:]))
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)


def print_summary(ep, rewards, acc, losses):
    n = len(rewards)
    if n < 10:
        return
    recent_r = np.mean(rewards[-10:])
    recent_a = np.mean(acc[-10:])
    recent_l = np.mean(losses[-10:]) if losses else 0
    early_r = np.mean(rewards[:10])
    early_a = np.mean(acc[:10])

    print(f"\n{'=' * 65}")
    print(f"  PROGRESS (ep {ep+1}) | {'Early (1-10)':>14} {'Recent (last10)':>16} {'Delta':>10}")
    print(f"  {'-' * 57}")
    print(f"  Reward:            {early_r:>+14.4f} {recent_r:>+16.4f} {recent_r-early_r:>+10.4f}")
    print(f"  Accuracy:          {early_a:>14.4f} {recent_a:>16.4f} {recent_a-early_a:>+10.4f}")
    print(f"  Actor Loss:        {'':>14} {recent_l:>16.4f}")
    print(f"  Noise:             {'':>14} {'':<16}")
    print(f"  Replay buffer:     {'':>14} {'':<16}")
    print(f"{'=' * 65}\n")


# ─── Main training loop ─────────────────────────────────────────────────────

def train(args, config, components):
    agent = components['agent']
    runner = components['runner']
    user_sim = components['user_sim']
    ckpt_dir = components['checkpoint_dir']

    n_episodes = args.episodes or config['training_data']['reinforcement']['num_episodes']
    ckpt_interval = args.checkpoint_interval
    log_interval = args.log_interval

    stats_file = ckpt_dir / 'training_stats.json'
    progress_file = ckpt_dir / 'training_progress.csv'

    # State
    episode_rewards = []
    episode_acc = []
    train_losses = []
    start_episode = 0

    # Try to resume from checkpoint
    if not args.fresh:
        candidates = sorted(ckpt_dir.glob('rl_episode_*.pt'), reverse=True)
        for ckpt_path in candidates:
            ckpt = load_checkpoint(ckpt_path, agent)
            if ckpt:
                start_episode = ckpt['episode']
                episode_rewards = ckpt.get('rewards', [])
                episode_acc = ckpt.get('acc', ckpt.get('ndcg', []))
                train_losses = ckpt.get('losses', [])
                print(f"\nResumed from {ckpt_path.name} (episode {start_episode})")
                print(f"  Noise: {agent.rl_agent.noise_scale:.4f}, "
                      f"Buffer: {len(agent.rl_agent.memory)}")
                break
        else:
            print("\nNo compatible checkpoint found — starting fresh")
            agent.rl_agent.clear_replay_buffer()
    else:
        print("\n--fresh: Starting from scratch")
        agent.rl_agent.clear_replay_buffer()
        # Clear old logs
        for f in [ckpt_dir / 'training_conversations.jsonl',
                  ckpt_dir / 'training_detailed.jsonl', progress_file]:
            if f.exists():
                f.unlink()

    conv_log = open(ckpt_dir / 'training_conversations.jsonl', 'a', encoding='utf-8')
    detailed_log = open(ckpt_dir / 'training_detailed.jsonl', 'a', encoding='utf-8')

    print(f"\n{'-' * 60}")
    print(f"  Episodes: {start_episode} -> {n_episodes}")
    print(f"  Max turns: {components['max_turns']}")
    print(f"  Users: {len(user_sim.user_profiles)}")
    print(f"  Reward: Concept Model Accuracy (SNR=1.69)")
    print(f"  Checkpoints: every {ckpt_interval} episodes")
    print(f"  Progress CSV: {progress_file}")
    print(f"{'-' * 60}\n")

    t0 = time.time()
    pbar = tqdm(
        range(start_episode, n_episodes),
        desc="Training", unit="ep",
        initial=start_episode, total=n_episodes,
    )

    try:
        for ep in pbar:
            # Run episode
            profile = user_sim.sample_user()
            result = runner.run_episode(agent, profile)
            uid = result['user_ground_truth'].get('user_id', '?')

            # Store experiences
            store_experiences(agent, result)

            # Train
            metrics = agent.rl_agent.train_step()
            actor_loss = metrics.get('actor_loss', 0) if metrics else 0
            avg_q = metrics.get('avg_q', 0) if metrics else 0
            noise = agent.rl_agent.noise_scale

            # Track
            ep_reward = result['total_reward']
            ep_acc = result['final_ndcg']  # accuracy (EpisodeRunner naming)
            episode_rewards.append(ep_reward)
            episode_acc.append(ep_acc)
            train_losses.append(actor_loss)

            # Progress bar
            pbar.set_postfix({
                'r': f'{ep_reward:.3f}',
                'acc': f'{ep_acc:.3f}',
                'Q': f'{avg_q:.2f}',
                'n': f'{noise:.3f}',
                'buf': len(agent.rl_agent.memory),
            })

            # Log conversation
            conv_log.write(json.dumps({
                'episode': ep + 1, 'user_id': uid,
                'reward': ep_reward, 'accuracy': ep_acc,
                'loss': actor_loss, 'avg_q': avg_q, 'noise': noise,
                'n_eval': result['user_ground_truth'].get('num_eval_targets'),
                'prefs': agent.discovered_preferences,
            }) + '\n')
            conv_log.flush()

            # Detailed turn log
            detailed_log.write(json.dumps({
                'episode': ep + 1, 'user_id': uid,
                'turns': result['turn_logs'],
                'metrics': {
                    'reward': ep_reward, 'accuracy': ep_acc,
                    'loss': actor_loss, 'avg_q': avg_q,
                },
            }) + '\n')
            detailed_log.flush()

            # CSV every episode
            save_progress_csv(progress_file, episode_rewards, episode_acc, train_losses)

            # Checkpoint
            if (ep + 1) % ckpt_interval == 0:
                ckpt_path = ckpt_dir / f'rl_episode_{ep+1}.pt'
                save_checkpoint(ckpt_path, agent, ep + 1,
                                episode_rewards, episode_acc, train_losses)
                save_stats_json(stats_file, episode_rewards, episode_acc, train_losses)

            # Summary
            if (ep + 1) % log_interval == 0:
                print_summary(ep, episode_rewards, episode_acc, train_losses)

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving checkpoint...")
        save_checkpoint(
            ckpt_dir / f'rl_interrupted_{ep+1}.pt', agent, ep + 1,
            episode_rewards, episode_acc, train_losses,
        )
    finally:
        conv_log.close()
        detailed_log.close()
        save_stats_json(stats_file, episode_rewards, episode_acc, train_losses)
        save_progress_csv(progress_file, episode_rewards, episode_acc, train_losses)

    elapsed = time.time() - t0
    n_done = len(episode_rewards) - start_episode
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Episodes: {n_done} in {elapsed:.0f}s ({elapsed/max(n_done,1):.1f}s/ep)")
    if len(episode_rewards) >= 10:
        print(f"  Reward  (last 10): {np.mean(episode_rewards[-10:]):+.4f}")
        print(f"  Accuracy(last 10): {np.mean(episode_acc[-10:]):.4f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Progress CSV: {progress_file}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = load_config()

    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not set (needed for user simulator + question gen)")
        sys.exit(1)

    components = init_components(config, args)
    train(args, config, components)


if __name__ == '__main__':
    main()
