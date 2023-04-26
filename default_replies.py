import numpy as np
class DefaultReplies:
    def __init__(self, replies=None):
        self.replies = [
            "I see what you did there",
            "That's what she said",
            "F",
            "* insert funny comment here *",
            "Am I the first comment?",
            "That's enough internet for today",
            "This is the good luck bot. 1 upvote = 1 good luck",
            "Nice"
        ]
        if replies:
            self.replies.extend(replies)
    def generate(self, text=None):
        return np.random.choice(self.replies)
