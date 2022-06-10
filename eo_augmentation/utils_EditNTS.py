# This code is based on https://github.com/YueDongCS/EditNTS/blob/master/label_edits.py

import numpy as np

def edit_distance(sent1, sent2, max_id=4999):
    m = len(sent1)
    n = len(sent2)
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j    # Min. operations = j
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
            elif sent1[i-1].lower() == sent2[j-1].lower():
                dp[i][j] = dp[i-1][j-1]
            else:
                edit_candidates = np.array([
                    dp[i][j-1], # Insert
                    dp[i-1][j] # Remove
                    ])
                dp[i][j] = 1 + min(edit_candidates)
    return dp

def sent2edit(sent1, sent2):
    dp = edit_distance(sent1, sent2)
    edits = []
    pos = []
    m, n = len(sent1), len(sent2)
    while m != 0 or n != 0:
        curr = dp[m][n]
        if m==0: #have to insert all here
            while n>0:
                left = dp[1][n-1]
                edits.append(sent2[n-1])
                pos.append(left)
                n-=1
        elif n==0:
            while m>0:
                top = dp[m-1][n]
                edits.append('DEL')
                pos.append(top)
                m -=1
        else: # we didn't reach any special cases yet
            diag = dp[m-1][n-1]
            left = dp[m][n-1]
            top = dp[m-1][n]
            if sent2[n-1].lower() == sent1[m-1].lower(): # keep
                edits.append('KEEP')
                pos.append(diag)
                m -= 1
                n -= 1
            elif curr == top+1: # INSERT preferred before DEL
                edits.append('DEL')
                pos.append(top)  # (sent2[n-1])
                m -= 1
            else: #insert
                edits.append(sent2[n - 1])
                pos.append(left)  # (sent2[n-1])
                n -= 1
    edits = edits[::-1]
    return edits


def edit2sent(sent, edits, last=False):
    new_sent = []
    sent_pointer = 0 #counter the total of KEEP and DEL, then align with original sentence
    if len(edits) == 0 or len(sent) ==0: # edit_list empty, return original sent
        return sent
    for i, edit in enumerate(edits):
        if len(sent) > sent_pointer: #there are tokens left for editing
            if edit =="KEEP":
                new_sent.append(sent[sent_pointer])
                sent_pointer += 1
            elif edit =="DEL":
                sent_pointer += 1
            else: #insert the word in
                new_sent.append(edit)
    if sent_pointer < len(sent):
        for i in range(sent_pointer,len(sent)):
            new_sent.append(sent[i])
    return new_sent