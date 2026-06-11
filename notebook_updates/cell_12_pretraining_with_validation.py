# Cell 12: RL Actor Supervised Pretraining (WITH VALIDATION METRICS)

from casper.data.reddit_loader import RedditDataLoader
from tqdm import tqdm
import time

pretrain_checkpoint = CHECKPOINT_DIR / 'actor_pretrained.pt'

if pretrain_checkpoint.exists():
    print(f"Loading pretrained actor from {pretrain_checkpoint}")
    checkpoint = torch.load(pretrain_checkpoint)
    casper_agent.rl_agent.actor.load_state_dict(checkpoint['actor'])
    print(f"Actor loaded from pretrained checkpoint")
    print(f"  Training loss: {checkpoint.get('loss', 'N/A'):.4f}")
    if 'val_cos_sim' in checkpoint:
        print(f"  Val cosine similarity: {checkpoint['val_cos_sim']:.3f}")
else:
    if not reddit_data_dir.exists():
        raise RuntimeError(f"Reddit data directory not found: {reddit_data_dir}")

    print(f"Loading Reddit data from {reddit_data_dir}...")
    try:
        reddit_loader = RedditDataLoader(data_path=str(reddit_data_dir))
        reddit_data = reddit_loader.load()

        if len(reddit_data) == 0:
            raise RuntimeError(f"No Reddit data found in {reddit_data_dir}!")

        print(f"Loaded {len(reddit_data)} Reddit posts")
        print(f"Transforming Reddit posts into training examples...")

        from casper.models.preference_extractor import PreferenceExtractor
        pref_extractor = PreferenceExtractor()

        training_examples = []
        valid_posts = 0

        for post in tqdm(reddit_data, desc="Processing Reddit posts", unit="post"):
            if not post.get('title') or post.get('text') == '[removed]' or post.get('text') == '[deleted]':
                continue

            context = post['title']
            if post.get('text') and post['text'] not in ['[removed]', '[deleted]', '']:
                context = f"{post['title']}. {post['text']}"

            conversation = [f"User: {context}"]
            try:
                preferences = pref_extractor.extract_from_conversation(conversation)
                concepts = preferences.get("liked", []) + preferences.get("neutral", [])

                if concepts:
                    valid_posts += 1
                    for concept in concepts[:5]:
                        target_emb = casper_agent.embedding_space.get_embedding(concept)
                        if target_emb is not None:
                            training_examples.append({
                                'context': context[:500],
                                'concept': concept,
                                'target_embedding': target_emb
                            })
            except Exception as e:
                continue

        print(f"\nProcessing summary:")
        print(f"  Total Reddit posts: {len(reddit_data)}")
        print(f"  Valid posts with concepts: {valid_posts}")
        print(f"  Training examples created: {len(training_examples)}")

        if len(training_examples) == 0:
            print("\nWARNING: No valid training examples created from Reddit data!")
            print("  Skipping supervised pretraining - actor will train from scratch during RL.")
        else:
            # Train/val split
            val_split = 0.1
            n_val = int(len(training_examples) * val_split)
            n_train = len(training_examples) - n_val

            # Shuffle
            indices = np.random.permutation(len(training_examples))
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]

            train_examples = [training_examples[i] for i in train_indices]
            val_examples = [training_examples[i] for i in val_indices]

            print(f"\nCreated {len(train_examples)} training examples from {valid_posts} posts")
            print(f"Validation examples: {len(val_examples)}")
            print(f"\nTraining actor for {PRETRAIN_EPOCHS} epochs...")

            optimizer = torch.optim.Adam(casper_agent.rl_agent.actor.parameters(), lr=0.001)
            batch_size = 32

            best_val_cos_sim = 0.0

            for epoch in tqdm(range(PRETRAIN_EPOCHS), desc="Pretraining epochs", unit="epoch"):
                # Training
                epoch_loss = 0
                num_batches = 0

                shuffled = np.random.permutation(len(train_examples))

                for i in range(0, len(shuffled), batch_size):
                    batch_indices = shuffled[i:i+batch_size]
                    batch = [train_examples[idx] for idx in batch_indices]

                    contexts = [ex['context'] for ex in batch]
                    states = casper_agent.encoder.encode(
                        contexts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )

                    targets = np.array([ex['target_embedding'] for ex in batch])

                    states_tensor = torch.FloatTensor(states)
                    targets_tensor = torch.FloatTensor(targets)

                    predicted = casper_agent.rl_agent.actor(states_tensor)

                    predicted_norm = torch.nn.functional.normalize(predicted, dim=1)
                    targets_norm = torch.nn.functional.normalize(targets_tensor, dim=1)

                    loss = torch.nn.functional.mse_loss(predicted_norm, targets_norm)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(casper_agent.rl_agent.actor.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

                # Validation
                if val_examples:
                    with torch.no_grad():
                        val_contexts = [ex['context'] for ex in val_examples]
                        val_states = casper_agent.encoder.encode(
                            val_contexts,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                        val_targets = np.array([ex['target_embedding'] for ex in val_examples])

                        val_states_tensor = torch.FloatTensor(val_states)
                        val_targets_tensor = torch.FloatTensor(val_targets)

                        val_pred = casper_agent.rl_agent.actor(val_states_tensor)

                        val_pred_norm = torch.nn.functional.normalize(val_pred, dim=1)
                        val_targets_norm = torch.nn.functional.normalize(val_targets_tensor, dim=1)

                        val_loss = torch.nn.functional.mse_loss(val_pred_norm, val_targets_norm).item()
                        val_cos_sim = torch.nn.functional.cosine_similarity(val_pred, val_targets_tensor).mean().item()

                    # Track best
                    if val_cos_sim > best_val_cos_sim:
                        best_val_cos_sim = val_cos_sim
                        best_marker = " (best)"
                    else:
                        best_marker = ""

                    tqdm.write(
                        f"Epoch {epoch+1}/{PRETRAIN_EPOCHS}: "
                        f"train_loss={avg_loss:.4f} | "
                        f"val_loss={val_loss:.4f} | "
                        f"val_cos_sim={val_cos_sim:.3f}{best_marker}"
                    )
                else:
                    tqdm.write(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS}: loss={avg_loss:.4f}")

                if (epoch + 1) % 5 == 0:
                    torch.save({
                        'actor': casper_agent.rl_agent.actor.state_dict(),
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'val_cos_sim': val_cos_sim if val_examples else None,
                        'num_examples': len(train_examples)
                    }, pretrain_checkpoint)

            print(f"\nPretraining complete!")
            print(f"  Final train loss: {avg_loss:.4f}")
            if val_examples:
                print(f"  Best val cosine similarity: {best_val_cos_sim:.3f}")
            print(f"  Checkpoint saved: {pretrain_checkpoint}")

    except Exception as e:
        print(f"\nWARNING: Supervised pretraining failed: {e}")
        print(f"  Actor will train from scratch during RL.")
        import traceback
        traceback.print_exc()
