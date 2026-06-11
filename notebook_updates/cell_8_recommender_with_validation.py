# Cell 8: Train Shared Recommender (WITH VALIDATION METRICS)

import pandas as pd
from tqdm import tqdm

rec_checkpoint = CHECKPOINT_DIR / 'shared_recommender.pt'

RECOMMENDER_TRAIN_USERS = config['models']['recommender']['training_users']
RECOMMENDER_EPOCHS = config['models']['recommender']['epochs']
RECOMMENDER_BATCH_SIZE = config['models']['recommender']['batch_size']
RECOMMENDER_VAL_SPLIT = config['models']['recommender'].get('val_split', 0.1)  # Load from config

if rec_checkpoint.exists():
    print(f"Loading shared recommender checkpoint from {rec_checkpoint}")
    checkpoint = torch.load(rec_checkpoint)
    shared_recommender.recommender.user_tower.load_state_dict(checkpoint['user_tower'])
    shared_recommender.recommender.item_tower.load_state_dict(checkpoint['item_tower'])

    casper_agent.recommender.recommender.user_tower.load_state_dict(checkpoint['user_tower'])
    casper_agent.recommender.recommender.item_tower.load_state_dict(checkpoint['item_tower'])

    trained_users = checkpoint.get('num_users', RECOMMENDER_TRAIN_USERS)
    print(f"Shared recommender loaded (trained on {trained_users} users)")
else:
    print(f"Training shared recommender on {RECOMMENDER_TRAIN_USERS} users...")

    def create_recommender_training_data(ratings_path, movies_path, num_users):
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)

        conversations = []
        liked_movies = []

        user_counts = ratings_df['userId'].value_counts()
        sampled_users = user_counts.head(num_users).index.tolist()

        for user_id in tqdm(sampled_users, desc="Creating training examples", unit="user"):
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            high_ratings = user_ratings[user_ratings['rating'] >= 4.0]

            if len(high_ratings) < 3:
                continue

            sampled = high_ratings.sample(min(5, len(high_ratings)), random_state=42)
            movie_ids = sampled['movieId'].tolist()

            titles = []
            for mid in movie_ids[:3]:
                movie_row = movies_df[movies_df['movieId'] == mid]
                if len(movie_row) > 0:
                    title = movie_row.iloc[0]['title'].split('(')[0].strip()
                    titles.append(title)

            if len(titles) >= 2:
                conv = f"I enjoyed {titles[0]} and {titles[1]}"
                if len(titles) > 2:
                    conv += f", also {titles[2]}"
                conversations.append(conv)
                liked_movies.append(movie_ids)

        return conversations, liked_movies

    convs, likes = create_recommender_training_data(
        DATA_DIR / 'ratings.csv',
        DATA_DIR / 'movies.csv',
        num_users=RECOMMENDER_TRAIN_USERS
    )

    print(f"Training recommender on {len(convs)} examples for {RECOMMENDER_EPOCHS} epochs...")
    print(f"Validation split: {RECOMMENDER_VAL_SPLIT}")

    # IMPORTANT: Pass val_split parameter to enable validation metrics
    shared_recommender.train(
        conversations=convs,
        liked_movies=likes,
        epochs=RECOMMENDER_EPOCHS,
        batch_size=RECOMMENDER_BATCH_SIZE,
        val_split=RECOMMENDER_VAL_SPLIT  # THIS IS THE KEY LINE
    )

    torch.save({
        'user_tower': shared_recommender.recommender.user_tower.state_dict(),
        'item_tower': shared_recommender.recommender.item_tower.state_dict(),
        'num_users': RECOMMENDER_TRAIN_USERS,
        'epoch': RECOMMENDER_EPOCHS
    }, rec_checkpoint)

    casper_agent.recommender.recommender.user_tower.load_state_dict(
        shared_recommender.recommender.user_tower.state_dict()
    )
    casper_agent.recommender.recommender.item_tower.load_state_dict(
        shared_recommender.recommender.item_tower.state_dict()
    )

    print(f"Shared recommender trained and saved: {rec_checkpoint}")
