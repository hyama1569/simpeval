def extract_ad_spans(edits):
    ad_spans = []
    seen_a = [0 for i in range(len(edits))]
    for i in range(len(edits) - 1):
        if seen_a[i] != 1:
            if edits[i] != 'KEEP' and edits[i] != 'DEL':
                start = i
                j = i + 1
                flag = False
                while j < len(edits):
                    if edits[j] == 'DEL':
                        j += 1
                        flag = True
                    elif edits[j] != 'KEEP' and edits[j] != 'DEL':
                        if flag == False:
                            seen_a[j] = 1
                            j += 1
                        else:
                            break
                    else:
                        break
                if flag == True:
                    end = j - 1
                    if end - start > 0:
                        ad_spans.append((start, end))
    return ad_spans

def extract_d_spans(edits):
    d_spans = []
    flag = False
    seen = [0 for i in range(len(edits))]
    for i in range(len(edits)):
        if seen[i] != 1:
            seen[i] = 1
            if edits[i] != 'KEEP' and edits[i] != 'DEL':
                flag = True
            elif edits[i] == 'KEEP':
                flag = False
            else:
                if flag == True:
                    continue
                else:
                    start = i
                    j = i + 1
                    while j < len(edits):
                        if edits[j] == 'DEL':
                            seen[j] = 1
                            j += 1
                        elif edits[j] == 'KEEP':
                            break
                        else:
                            flag = True
                            break
                    end = j - 1
                    if end - start >= 0:
                        d_spans.append((start, end))
    return d_spans 

def extract_a_spans(edits):
    a_spans = []
    seen_a = [0 for i in range(len(edits))]
    for i in range(len(edits)):
        if seen_a[i] != 1:
            seen_a[i] = 1
            if edits[i] != 'KEEP' and edits[i] != 'DEL':
                start = i
                j = i + 1
                flag = False
                while j < len(edits):
                    if edits[j] != 'KEEP' and edits[j] != 'DEL':
                        seen_a[j] = 1
                        j += 1
                    elif edits[j] == 'DEL':
                        flag = True
                        break
                    else:
                        break
                end = j - 1
                if (flag == False) and (end - start >= 0):
                    a_spans.append((start, end))
    return a_spans

def extract_d_starts_from_ad_spans(edits, ad_spans):
    d_starts = []
    for span in ad_spans:
        a_start = span[0]
        d_start = a_start
        while d_start < len(edits):
            if edits[d_start] != 'DEL':
                d_start += 1
            else:
                break
        d_starts.append(d_start)
    return d_starts

def assign_ids_to_edits(edits, sent2_tok):
    edits_ids = []
    sent1_pointer = 0
    sent2_pointer = 0
    for i in range(len(edits)):
        if edits[i] == 'KEEP':
            edits_ids.append(sent1_pointer)
            sent1_pointer += 1
        elif edits[i] == 'DEL':
            edits_ids.append(sent1_pointer)
            sent1_pointer += 1
        else:
            while sent2_pointer < len(sent2_tok):
                if sent2_tok[sent2_pointer] == edits[i]:
                    edits_ids.append(sent2_pointer)
                    sent2_pointer += 1
                    break
                else:
                    sent2_pointer += 1
    return edits_ids

def extract_splr_ids(edits, ad_spans, sent2_tok):
    edits_ids = assign_ids_to_edits(edits, sent2_tok)
    splr_ids = []
    for ad_span_idx in range(len(ad_spans)):
        splr_flag = False
        for i in range(ad_spans[ad_span_idx][0], ad_spans[ad_span_idx][1] + 1):
            if edits[i] == '.':
                splr_flag = True
        if splr_flag == True:
            #sent1_span = [edits_ids[j] for j in range(ad_spans[ad_span_idx][0], ad_spans[ad_span_idx][1]+1) if edits[j] == 'KEEP' or edits[j] == 'DEL']
            #sent2_span = [edits_ids[j] for j in range(ad_spans[ad_span_idx][0], ad_spans[ad_span_idx][1]+1) if edits[j] != 'KEEP' and edits[j] != 'DEL']
            #splr_ids.append((sent1_span, sent2_span, ad_span_idx))
            splr_ids.append(['splr_span', ad_span_idx])
    return splr_ids

def extract_rep_ad_ids(edits, ad_spans, sent2_tok, aligns, splr_ids):
    edits_ids = assign_ids_to_edits(edits, sent2_tok)
    d_starts = extract_d_starts_from_ad_spans(edits, ad_spans)
    rep_ids = []
    ad_ids = []
    ad_spans_done = [i[1] for i in splr_ids]
    for ad_span_idx in range(len(ad_spans)):
        if ad_span_idx in ad_spans_done:
            continue
        else:
            now_span = ad_spans[ad_span_idx]
            d_start_in_now_span = d_starts[ad_span_idx]
            d_span_in_now_span = list(range(d_start_in_now_span, now_span[1]+1))
            a_span_in_now_span = list(range(now_span[0], d_start_in_now_span))
            sent1_ids_corresponding_d_span_in_now_span = [edits_ids[i] for i in d_span_in_now_span if edits[i] == 'DEL']
            sent2_ids_corresponding_a_span_in_now_span = [edits_ids[i] for i in a_span_in_now_span if edits[i] != 'DEL']

            added_words_to_sent1_by_a_span = []
            for i in range(len(sent2_ids_corresponding_a_span_in_now_span)):
                added_words_to_sent1_by_a_span.append(sent2_tok[sent2_ids_corresponding_a_span_in_now_span[i]])

            aligned_words_in_sent2 = []
            for i in range(len(sent1_ids_corresponding_d_span_in_now_span)):
                for j in range(len(aligns)):
                    if sent1_ids_corresponding_d_span_in_now_span[i] in aligns[j][0]:
                        for aligned_word_id in aligns[j][1]:
                            aligned_words_in_sent2.append(sent2_tok[aligned_word_id])

            rep_flag = False
            for word in aligned_words_in_sent2:
                if word in added_words_to_sent1_by_a_span:
                    rep_flag = True
            
            if rep_flag == True:
                rep_ids.append(['rep_span', ad_span_idx])
            else:
                ad_ids.append(['ad_span', ad_span_idx])
            
    return rep_ids, ad_ids

def extract_mvr_ids(edits, ad_spans, d_spans, a_spans, sent2_tok, aligns):
    edits_ids = assign_ids_to_edits(edits, sent2_tok)
    mvr_ids = []
    seen_a_span = [0 for i in range(len(a_spans))]
    seen_ad_span = [0 for i in range(len(ad_spans))]
    for d_span_idx in range(len(d_spans)):
        tmp_ids = ['mvr_span']
        d_span = list(range(d_spans[d_span_idx][0], d_spans[d_span_idx][1]+1))
        sent1_ids_corresponding_d_span = [edits_ids[i] for i in d_span if edits[i] == 'DEL']

        aligned_words_in_sent2_from_d_span = []
        for i in range(len(sent1_ids_corresponding_d_span)):
            for j in range(len(aligns)):
                if sent1_ids_corresponding_d_span[i] in aligns[j][0]:
                    for aligned_word_id in aligns[j][1]:
                        aligned_words_in_sent2_from_d_span.append(sent2_tok[aligned_word_id])
        
        if len(aligned_words_in_sent2_from_d_span) == 0:
            continue
        else:
            tmp_ids.append(('d_span', d_span_idx))

            # check a_spans
            for a_span_idx in range(len(a_spans)):
                a_span = list(range(a_spans[a_span_idx][0], a_spans[a_span_idx][1]+1))
                if d_span[-1] > a_span[0]:
                    continue
                else:
                    sent2_ids_corresponding_a_span = [edits_ids[i] for i in a_span if edits[i] != 'DEL']
                    added_words_to_sent1_by_a_span = []
                    for i in range(len(sent2_ids_corresponding_a_span)):
                        added_words_to_sent1_by_a_span.append(sent2_tok[sent2_ids_corresponding_a_span[i]])
                    
                    flag = False
                    for word in aligned_words_in_sent2_from_d_span:
                        if word in added_words_to_sent1_by_a_span:
                            flag = True
                    if flag == True:
                        if seen_a_span[a_span_idx] == 0:
                            tmp_ids.append(('a_span', a_span_idx))
                            seen_a_span[a_span_idx] = 1
            
            # check ad_span
            d_starts = extract_d_starts_from_ad_spans(edits, ad_spans)
            for ad_span_idx in range(len(ad_spans)):
                ad_span = list(range(ad_spans[ad_span_idx][0], ad_spans[ad_span_idx][1]+1))
                if d_span[-1] > ad_span[0]:
                    continue
                else:
                    d_start_in_ad_span = d_starts[ad_span_idx]
                    d_span_in_ad_span = list(range(d_start_in_ad_span, ad_span[-1]+1))
                    a_span_in_ad_span = list(range(ad_span[0], d_start_in_ad_span))
                    sent1_ids_corresponding_d_span_in_ad_span = [edits_ids[i] for i in d_span_in_ad_span if edits[i] == 'DEL']
                    sent2_ids_corresponding_a_span_in_ad_span = [edits_ids[i] for i in a_span_in_ad_span if edits[i] != 'DEL']
                    
                    added_words_to_sent1_by_a_span_in_ad_span = []
                    for i in range(len(sent2_ids_corresponding_a_span_in_ad_span)):
                        added_words_to_sent1_by_a_span_in_ad_span.append(sent2_tok[sent2_ids_corresponding_a_span_in_ad_span[i]])

                    aligned_words_in_sent2_from_ad_span = []
                    for i in range(len(sent1_ids_corresponding_d_span_in_ad_span)):
                        for j in range(len(aligns)):
                            if sent1_ids_corresponding_d_span_in_ad_span[i] in aligns[j][0]:
                                for aligned_word_id in aligns[j][1]:
                                    aligned_words_in_sent2_from_ad_span.append(sent2_tok[aligned_word_id])
                    
                    flag = False
                    if len(aligned_words_in_sent2_from_ad_span) == 0:
                        flag = True
                    for word in aligned_words_in_sent2_from_d_span:
                        if word in added_words_to_sent1_by_a_span_in_ad_span:
                            flag = True
                    if flag == True:
                        if seen_ad_span[ad_span_idx] == 0:
                            tmp_ids.append(('ad_span', ad_span_idx))
                            seen_ad_span[ad_span_idx] = 1

            mvr_ids.append(tmp_ids)
    return mvr_ids

def extract_d_ids(edits, d_spans, sent2_tok, aligns):
    edits_ids = assign_ids_to_edits(edits, sent2_tok)
    d_ids = []
    for d_span_idx in range(len(d_spans)):
        d_span = list(range(d_spans[d_span_idx][0], d_spans[d_span_idx][1]+1))
        sent1_ids_corresponding_d_span = [edits_ids[i] for i in d_span if edits[i] == 'DEL']

        aligned_words_in_sent2_form_d_span = []
        for i in range(len(sent1_ids_corresponding_d_span)):
            for j in range(len(aligns)):
                if sent1_ids_corresponding_d_span[i] in aligns[j][0]:
                    for aligned_word_id in aligns[j][1]:
                        aligned_words_in_sent2_form_d_span.append(sent2_tok[aligned_word_id])
        
        if len(aligned_words_in_sent2_form_d_span) == 0:
            d_ids.append(['d_span', d_span_idx])
    
    return d_ids

def extract_a_ids(a_spans, mvr_ids):
    a_ids = []
    a_spans_done = []
    for mvr_info in mvr_ids:
        for i in range(1, len(mvr_info)):
            if mvr_info[i][0] == 'a_span':
                a_spans_done.append(mvr_info[i][1])

    for a_span_idx in range(len(a_spans)):
        if a_span_idx in a_spans_done:
            continue
        else:
            a_ids.append(['a_span', a_span_idx])

    return a_ids