############################################################
# Spam Filter
############################################################

import email
import math
import os
import Queue


def load_tokens(email_path):
    with open(email_path) as email_file:
        message = email.message_from_file(email_file)
    return [token for line in email.iterators.body_line_iterator(message)
            for token in line.split()]


def log_probs(email_paths, smoothing):
    count_dict = {}
    log_prob_dict = {}
    total_count = 0
    num_words = 0
    for email_path in email_paths:
        tokens = load_tokens(email_path)
        for token in tokens:
            if token in count_dict:
                count_dict[token] += 1
                total_count += 1
            else:
                count_dict[token] = 1
                num_words += 1
                total_count += 1
    denominator = total_count + (smoothing * (num_words + 1))
    for token in count_dict.keys():
        numerator = count_dict[token] + smoothing
        log_prob_dict[token] = math.log(numerator / denominator)
    log_prob_dict["<UNK>"] = math.log(smoothing / denominator)
    return log_prob_dict


class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        spam_filenames = [spam_dir + "/" + f for f in os.listdir(spam_dir)]
        ham_filenames = [ham_dir + "/" + f for f in os.listdir(ham_dir)]
        self.spam_dict = log_probs(spam_filenames, smoothing)
        self.ham_dict = log_probs(ham_filenames, smoothing)
        num_spam_files = float(len(spam_filenames))
        num_ham_files = float(len(ham_filenames))
        num_files = num_spam_files + num_ham_files
        self.log_p_spam = math.log(num_spam_files / num_files)
        self.log_p_ham = math.log(num_ham_files / num_files)

    def is_spam(self, email_path):
        tokens = load_tokens(email_path)
        spam_sum = 0
        ham_sum = 0
        for token in tokens:
            if token in self.spam_dict:
                spam_sum += self.spam_dict[token]
            else:
                spam_sum += self.spam_dict["<UNK>"]
            if token in self.ham_dict:
                ham_sum += self.ham_dict[token]
            else:
                ham_sum += self.ham_dict["<UNK>"]
        spam_sum += self.log_p_spam
        ham_sum += self.log_p_ham
        return spam_sum > ham_sum

    def most_indicative(self, n, log_dict):
        pq = Queue.PriorityQueue(n)
        for word in self.spam_dict:
            if word in self.ham_dict:
                p_w_spam = math.exp(self.spam_dict[word])
                p_w_ham = math.exp(self.ham_dict[word])
                p_w = p_w_spam + p_w_ham
                indication = log_dict[word] - math.log(p_w)
                new_item = (indication, word)
                if pq.qsize() < n:
                    pq.put(new_item)
                else:
                    curr_root = pq.get()
                    if indication > curr_root[0]:
                        pq.put(new_item)
                    else:
                        pq.put(curr_root)
        indicative_list = []
        while not pq.empty():
            indicative_list.append(pq.get()[1])
        return list(reversed(indicative_list))

    def most_indicative_spam(self, n):
        return self.most_indicative(n, self.spam_dict)

    def most_indicative_ham(self, n):
        return self.most_indicative(n, self.ham_dict)
