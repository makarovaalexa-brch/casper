"""
Logging and experiment tracking utilities for CASPER.

Handles metrics logging, experiment tracking, and results persistence.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class MetricsLogger:
    """
    Log and track evaluation metrics over time.

    Separate from metric computation (see evaluation/metrics.py).
    Used by training pipeline to persist results.
    """

    def __init__(self, log_dir: str = "experiments/logs"):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_evaluation(
        self,
        metrics: Dict,
        agent_name: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save evaluation metrics to file.

        Args:
            metrics: Evaluation metrics dictionary
            agent_name: Name of the agent being evaluated
            timestamp: Timestamp for this evaluation (defaults to now)
            metadata: Optional metadata (config, hyperparams, etc.)

        Returns:
            Path to saved log file
        """
        if timestamp is None:
            timestamp = datetime.now()

        log_entry = {
            'timestamp': timestamp.isoformat(),
            'agent': agent_name,
            'metrics': metrics
        }

        if metadata is not None:
            log_entry['metadata'] = metadata

        # Save to JSON file
        log_file = self.log_dir / f"{agent_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)

        print(f"Logged metrics to: {log_file}")
        return log_file

    def log_training_metrics(
        self,
        epoch: int,
        metrics: Dict,
        model_name: str,
        append: bool = True
    ) -> Path:
        """
        Log training metrics for a specific epoch.

        Args:
            epoch: Training epoch number
            metrics: Metrics for this epoch
            model_name: Name of the model being trained
            append: Append to existing log file or create new

        Returns:
            Path to log file
        """
        log_file = self.log_dir / f"{model_name}_training.jsonl"

        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        mode = 'a' if append else 'w'
        with open(log_file, mode) as f:
            f.write(json.dumps(log_entry) + '\n')

        return log_file

    def load_evaluation_logs(self, agent_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load all evaluation logs, optionally filtered by agent name.

        Args:
            agent_name: Filter logs for specific agent (None = all agents)

        Returns:
            DataFrame with all logged evaluations
        """
        log_files = list(self.log_dir.glob("*.json"))

        if agent_name is not None:
            log_files = [f for f in log_files if f.stem.startswith(agent_name)]

        all_logs = []
        for log_file in log_files:
            with open(log_file, 'r') as f:
                log_entry = json.load(f)
                # Flatten metrics into top-level dict
                flat_entry = {
                    'timestamp': log_entry['timestamp'],
                    'agent': log_entry['agent'],
                    **log_entry['metrics']
                }
                all_logs.append(flat_entry)

        if not all_logs:
            return pd.DataFrame()

        return pd.DataFrame(all_logs)

    def load_training_logs(self, model_name: str) -> pd.DataFrame:
        """
        Load training logs for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            DataFrame with training metrics per epoch
        """
        log_file = self.log_dir / f"{model_name}_training.jsonl"

        if not log_file.exists():
            print(f"No training logs found for {model_name}")
            return pd.DataFrame()

        logs = []
        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Flatten metrics
                flat_entry = {
                    'epoch': entry['epoch'],
                    'timestamp': entry['timestamp'],
                    **entry['metrics']
                }
                logs.append(flat_entry)

        return pd.DataFrame(logs)

    def compare_agents(
        self,
        agent_results: Dict[str, Dict],
        sort_by: str = 'mean_success_rate',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Create comparison table for multiple agents.

        Args:
            agent_results: Dict mapping agent_name -> metrics dict
            sort_by: Metric to sort by
            ascending: Sort order

        Returns:
            DataFrame with agent comparison
        """
        comparison_data = []

        for agent_name, metrics in agent_results.items():
            row = {
                'Agent': agent_name,
                'Success Rate': metrics.get('mean_success_rate', 0),
                'Avg Questions': metrics.get('mean_questions', 0),
                'Efficiency': metrics.get('mean_efficiency', 0),
                'Questions to Success': metrics.get('median_questions_to_success', float('inf')),
                'NDCG@10': metrics.get('mean_ndcg_10', 0)
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)

        return df

    def save_comparison(
        self,
        comparison_df: pd.DataFrame,
        filename: str = "agent_comparison.csv"
    ) -> Path:
        """
        Save agent comparison to CSV file.

        Args:
            comparison_df: Comparison DataFrame
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.log_dir / filename
        comparison_df.to_csv(output_path, index=False)
        print(f"Saved comparison to: {output_path}")
        return output_path
