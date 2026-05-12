"""
generate_dataset.py
===================
PURPOSE: Creates a realistic synthetic movie/product review dataset.
         In a real project you'd download the IMDB or Amazon dataset,
         but this ensures everyone can run the project instantly.

WHY SYNTHETIC DATA?
- No download needed → works offline
- Controlled class balance → fair model training  
- Realistic enough for a portfolio demo
"""

import pandas as pd
import random
import os

# ── Seed for reproducibility (same "random" data every run) ──────────────────
random.seed(42)

# ── Review templates by sentiment ────────────────────────────────────────────
POSITIVE_REVIEWS = [
    "This movie was absolutely fantastic! The acting was superb and the story kept me hooked.",
    "One of the best films I have seen in years. Highly recommend to everyone.",
    "Amazing cinematography and a brilliant storyline. A must-watch!",
    "The characters were so well developed. I felt every emotion with them.",
    "Incredible performance by the entire cast. Loved every minute of it.",
    "A masterpiece of modern cinema. The director did an outstanding job.",
    "I was blown away by the special effects and the emotional depth of the story.",
    "This film exceeded all my expectations. Beautiful and moving.",
    "Wonderful script, great acting, and a story that stays with you long after.",
    "Absolutely loved it! The plot twists were unexpected and thrilling.",
    "One of the most touching movies I have ever watched. Truly inspiring.",
    "Brilliant from start to finish. The pacing was perfect.",
    "A feel-good movie that left me smiling for days. Highly enjoyable.",
    "The best movie of the year without a doubt. Stunning visuals.",
    "Heartwarming and funny in equal measure. A perfect family film.",
    "I was completely captivated from the opening scene to the final credits.",
    "The chemistry between the lead actors was electric. Very entertaining.",
    "An emotional rollercoaster in the best possible way. Loved it!",
    "Exceptional storytelling with a message that really resonates.",
    "This film restored my faith in Hollywood. Simply outstanding.",
    "Great product! Works exactly as described and arrived quickly.",
    "Excellent quality, very happy with my purchase. Will buy again.",
    "Super fast delivery and the item was even better than expected.",
    "Outstanding customer service and a top-quality product.",
    "Absolutely love this! It works perfectly and looks great.",
    "Best purchase I have made this year. Highly recommend.",
    "The quality is amazing for the price. Very impressed.",
    "Five stars all round. Fast shipping and perfect condition.",
    "Exactly what I needed. Works flawlessly. Very satisfied.",
    "Brilliant product. Does exactly what it says on the tin.",
]

NEGATIVE_REVIEWS = [
    "Terrible movie. Waste of two hours. The plot made no sense at all.",
    "Boring and predictable. I nearly fell asleep halfway through.",
    "Awful acting and a storyline that went nowhere. Very disappointing.",
    "One of the worst films I have ever seen. Do not bother watching.",
    "The special effects were cheap and the dialogue was cringeworthy.",
    "A complete disaster from beginning to end. Hugely disappointing.",
    "I want my money back. This movie was absolutely dreadful.",
    "Poorly written and badly directed. A total mess.",
    "The characters were flat and the story was painfully slow.",
    "Dreadful. Painful to sit through. Avoid at all costs.",
    "Nothing made sense in this film. Very frustrating to watch.",
    "The worst movie I have seen in years. Completely unwatchable.",
    "Terrible screenplay, wooden acting, and a plot full of holes.",
    "I was bored within the first 10 minutes. It never got better.",
    "A huge disappointment. The trailer was far more interesting than the film.",
    "Utterly forgettable and pointless. A waste of everyone's talent.",
    "Awful product. Broke after two days. Total waste of money.",
    "Terrible quality. Looks nothing like the photos. Very disappointed.",
    "Completely useless. Does not work as advertised. Avoid.",
    "Worst purchase ever. Cheap materials and poor craftsmanship.",
    "Arrived damaged and the company refused to help. Disgraceful service.",
    "Absolute rubbish. Stopped working after one week.",
    "Do not buy this. It is a scam. The quality is shocking.",
    "Very poor quality. Fell apart immediately. Total disappointment.",
    "Misleading description. The product is nothing like what was shown.",
    "Zero stars if I could. Dreadful product and even worse service.",
    "Not worth a single penny. Broke on first use.",
    "Appalling quality control. Would not recommend to anyone.",
    "Cheap and nasty. Looks terrible in real life.",
    "Returned immediately. Complete waste of money.",
]

NEUTRAL_REVIEWS = [
    "The movie was okay. Not great, but not terrible either.",
    "It was an average film. Some parts were good, others less so.",
    "Decent enough. I have seen better but also much worse.",
    "The plot was acceptable but the ending felt a bit rushed.",
    "Nothing special but a passable way to spend an evening.",
    "It was fine. Exactly what you would expect from this type of film.",
    "Some good moments but overall just average.",
    "The acting was okay. The story was reasonably entertaining.",
    "Not bad, not great. A mediocre experience overall.",
    "A middle-of-the-road film. Neither impressive nor terrible.",
    "It was watchable but I probably would not recommend it strongly.",
    "The film had potential but did not fully deliver on its promise.",
    "Some interesting ideas that were not fully explored. Just okay.",
    "A standard entry in the genre. Nothing to get excited about.",
    "Not particularly memorable but not offensive either.",
    "The product is okay. Does the job but nothing more.",
    "Average quality. It works but I expected better for the price.",
    "Reasonable product. Nothing special but does what it should.",
    "It is fine. Arrived on time and works as expected.",
    "Okay product. Not amazing but not terrible. Neutral feelings.",
    "Does the job adequately. Nothing exceptional to report.",
    "Average experience. Would not go out of my way to recommend.",
    "The quality is acceptable. Met my basic requirements.",
    "Satisfactory purchase. Could have been better, could have been worse.",
    "Functional and affordable. That is about all there is to say.",
    "It is alright. Middling quality but serves its purpose.",
    "Not bad for the money. Neutral experience overall.",
    "An ordinary product. Nothing to complain about, nothing to rave about.",
    "Gets the job done. No more, no less.",
    "Average all round. A solid three-star experience.",
]


def generate_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate a balanced synthetic review dataset.

    Parameters
    ----------
    n_samples : int
        Total number of reviews (split evenly across 3 sentiments).

    Returns
    -------
    pd.DataFrame with columns: review, sentiment, review_length
    """
    per_class = n_samples // 3
    reviews, labels = [], []

    for template_list, label in [
        (POSITIVE_REVIEWS, "positive"),
        (NEGATIVE_REVIEWS, "negative"),
        (NEUTRAL_REVIEWS, "neutral"),
    ]:
        for _ in range(per_class):
            # Pick a random template and add slight variation
            base = random.choice(template_list)
            reviews.append(base)
            labels.append(label)

    # Shuffle so sentiments are not in order
    combined = list(zip(reviews, labels))
    random.shuffle(combined)
    reviews, labels = zip(*combined)

    df = pd.DataFrame({"review": reviews, "sentiment": labels})
    df["review_length"] = df["review"].apply(len)
    df["word_count"] = df["review"].apply(lambda x: len(x.split()))
    df.index.name = "id"

    return df


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "data", "reviews.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = generate_dataset(n_samples=900)
    df.to_csv(output_path, index=True)

    print(f"✅ Dataset saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"\n   Sentiment distribution:")
    print(df["sentiment"].value_counts())
    print(f"\n   Sample reviews:")
    print(df.head(3)[["review", "sentiment"]])
