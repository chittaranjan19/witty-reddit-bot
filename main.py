from bot import RedditBot
from unigram_lm import UnigramLM
from default_replies import DefaultReplies
from zeroshot import ZeroShotCommenter
from finetuned_t5 import T5Commenter
import pandas as pd

if __name__ == "__main__":
    # baseline_model_unigram = UnigramLM(open("comment_corpus").read())
    # baseline_model_default_replies = DefaultReplies()
    zero_shot_commenter = ZeroShotCommenter()
    # t5_commenter = T5Commenter()

    bot = RedditBot()
    submission_to_comment_on = bot.find_post()
    # witty_comment = baseline_model_unigram.generate(submission_to_comment_on.title + "\n" + submission_to_comment_on.selftext)
    # witty_comment = baseline_model_default_replies.generate(submission_to_comment_on.title + "\n" + submission_to_comment_on.selftext)
    witty_comment = zero_shot_commenter.generate(submission_to_comment_on.title + "\n" + submission_to_comment_on.selftext)
    # witty_comment = t5_commenter.generate(submission_to_comment_on.title + "\n" + submission_to_comment_on.selftext)
    print(submission_to_comment_on.title + "\n" + submission_to_comment_on.selftext)
    print(witty_comment)
    bot.comment(witty_comment, submission_to_comment_on)

    # EVALUATION
    # eval_corpus = open("eval.txt").readlines()
    # eval_df = pd.read_csv("reddit_eval.csv")
    # ps, mean_ps = zero_shot_commenter.evaluate(eval_df)
    # ps, mean_ps = t5_commenter.evaluate(eval_df)
    # print(ps, mean_ps)


    # print(baseline_model.evaluate(eval_corpus))