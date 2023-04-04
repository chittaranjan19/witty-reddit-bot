import praw
from datetime import datetime
import os

class RedditBot:
    def __init__(self):
        client_id = os.environ["CLIENT_ID"]
        client_secret = os.environ["CLIENT_ID"]
        password = os.environ["REDDIT_PASSWORD"]
        user_agent = os.environ["USER_AGENT"]
        username = os.environ["REDDIT_USERNAME"]

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            password=password,
            user_agent=user_agent,
            username=username,
        )


    def find_post(self):
        return next(self.reddit.subreddit("tifu+Showerthoughts+Jokes+oneliners").rising(limit=1))

    def comment(self, comment, submission_to_comment_on):
        submission_to_comment_on.reply(comment)
        self._log_comment(submission_to_comment_on.permalink)

    def _log_comment(self, permalink):
        f = open("commented.csv", "a+")
        f.write(str(datetime.now()) + "," + permalink + "\n")
        f.close()