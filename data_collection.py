import pandas as pd
import praw
from tqdm import tqdm

reddit = praw.Reddit(
    client_id="Fx7jTBqWbi0Z2mdLr8tGhg",
    client_secret="pAQWFz_tjmlLnaz48gggoIz7TT1q2w",
    password="Drog1234%",
    user_agent="script:si630:v0.1 (by u/whoisjoe1)",
    username="whoisjoe1",
)

submission_ids = []
submission_text = []
titles = []
permalinks = []
comment_scores = []
comment_ids = []
comment_text = []

for submission in tqdm(reddit.subreddit("tifu+Showerthoughts+Jokes+oneliners").top(time_filter="all", limit=200), total=200):
    if submission.num_comments > 100:
        submission_ids.append(submission.id)
        titles.append(submission.title)
        submission_text.append(submission.selftext)
        permalinks.append(submission.permalink)

        submission.comments.replace_more(limit=0)

        comment_text_for_submission = []
        comment_ids_for_submission = []
        comment_scores_for_submission = []
        for top_level_comment in submission.comments:
            if top_level_comment.score > 100:
                comment_text_for_submission.append(top_level_comment.body)
                comment_ids_for_submission.append(top_level_comment.id)
                comment_scores_for_submission.append(top_level_comment.score)
        comment_scores.append(comment_scores)
        comment_ids.append(comment_ids_for_submission)
        comment_text.append(comment_text_for_submission)

data = pd.DataFrame({
    "submission_ids": submission_ids,
    "submission_text": submission_text,
    "titles": titles,
    "permalinks": permalinks,
    "comment_scores": comment_scores,
    "comment_ids": comment_ids,
    "comment_text": comment_text
})

data.to_csv("reddit.csv")