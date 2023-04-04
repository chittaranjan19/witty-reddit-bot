from bot import RedditBot
from unigram_lm import UnigramLM

if __name__ == "__main__":
    baseline_model = UnigramLM(open("comment_corpus").read())
    # bot = RedditBot()
    # submission_to_comment_on = bot.find_post()
    # witty_comment = baseline_model.generate(submission_to_comment_on.selftext)
    # bot.comment(witty_comment, submission_to_comment_on)

    # EVALUATION
    eval_corpus = open("eval.txt").readlines()
    print(baseline_model.evaluate(eval_corpus))