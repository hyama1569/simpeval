import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

class Features(object):
	def __init__(self, input_ids_a_and_b, input_ids_b_and_a, attention_mask, segment_ids_a_and_b, segment_ids_b_and_a, sent1_valid_ids, sent2_valid_ids, sent1_wordpiece_length, sent2_wordpiece_length, label_s2t, label_t2s):
		self.input_ids_a_and_b = input_ids_a_and_b
		self.input_ids_b_and_a = input_ids_b_and_a
		self.attention_mask = attention_mask
		self.segment_ids_a_and_b = segment_ids_a_and_b
		self.segment_ids_b_and_a = segment_ids_b_and_a
		self.sent1_valid_ids = sent1_valid_ids
		self.sent2_valid_ids = sent2_valid_ids
		self.sent1_wordpiece_length=sent1_wordpiece_length
		self.sent2_wordpiece_length=sent2_wordpiece_length
		self.label_s2t = label_s2t
		self.label_t2s = label_t2s

def convert_examples_to_features(data_examples, set_type, max_seq_length, tokenizer, args, pad_token=0, pad_token_segment_id=0):
	features = []
	len_examples=len(data_examples)
	for ex_index, example in enumerate(data_examples):
		if ex_index % 200 == 0:
			print("Processed "+set_type+" examples %d/%d" % (ex_index, len_examples))

		if len(example.text_a.split())>args.max_sent_length or len(example.text_b.split())>args.max_sent_length:
			if set_type=='train':
				continue
			else:
				example.text_a=' '.join(example.text_a.split()[:args.max_sent_length])
				example.text_b=' '.join(example.text_b.split()[:args.max_sent_length])
		inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_seq_length, truncation=True)
		input_ids_a_and_b, token_type_ids_a_and_b = inputs["input_ids"], inputs["token_type_ids"]
		# print(inputs)
		inputs = tokenizer.encode_plus(example.text_b, example.text_a, add_special_tokens=True, max_length=max_seq_length, truncation=True)
		input_ids_b_and_a, token_type_ids_b_and_a = inputs["input_ids"], inputs["token_type_ids"]
		# print(example)
		# print(input_ids_a_and_b)
		# print(input_ids_b_and_a)
		# sys.exit()
		attention_mask = inputs["attention_mask"] #[1] * len(input_ids_a_and_b)
		padding_length = max_seq_length - len(input_ids_a_and_b)
		input_ids_a_and_b = input_ids_a_and_b + ([pad_token] * padding_length)
		input_ids_b_and_a = input_ids_b_and_a + ([pad_token] * padding_length)
		attention_mask = attention_mask + ([0] * padding_length)
		token_type_ids_a_and_b = token_type_ids_a_and_b + ([pad_token_segment_id] * padding_length)
		token_type_ids_b_and_a = token_type_ids_b_and_a + ([pad_token_segment_id] * padding_length)
		sent1_valid_ids = []  #
		sent2_valid_ids = []
		sent1_wordpiece_length = 0
		sent2_wordpiece_length = 0
		total_length = 1  # [CLS]
		for idx, word in enumerate(example.text_a.split()):
			token = tokenizer.tokenize(word)
			sent1_valid_ids.append(total_length)
			total_length += len(token)
			sent1_wordpiece_length += len(token)
		total_length += 1  # [SEP]
		for idx, word in enumerate(example.text_b.split()):
			token = tokenizer.tokenize(word)
			sent2_valid_ids.append(total_length)
			total_length += len(token)
			sent2_wordpiece_length += len(token)
		# print(total_length, sum(attention_mask))
		# sys.exit()
		sent1_valid_ids = sent1_valid_ids + ([0] * (max_seq_length - len(sent1_valid_ids)))
		sent2_valid_ids = sent2_valid_ids + ([0] * (max_seq_length - len(sent2_valid_ids)))
		assert len(input_ids_a_and_b) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids_a_and_b), max_seq_length)
		assert len(attention_mask) == max_seq_length, "Error with input length {} vs {}".format(len(attention_mask), max_seq_length)
		assert len(token_type_ids_a_and_b) == max_seq_length, "Error with input length {} vs {}".format(len(token_type_ids_a_and_b), max_seq_length)
		assert len(sent1_valid_ids) == max_seq_length, "Error with input length {} vs {}".format(len(sent1_valid_ids), max_seq_length)
		assert len(sent2_valid_ids) == max_seq_length, "Error with input length {} vs {}".format(len(sent2_valid_ids), max_seq_length)
		label_s2t=np.zeros((args.max_sent_length, args.max_span_size), dtype=int) -1
		label_t2s=np.zeros((args.max_sent_length, args.max_span_size), dtype=int) -1
		left2right = {}
		for pair in example.label.split():
			left, right = pair.split('-')
			left, right = int(left), int(right)
			if left in left2right:
				left2right[left].append(right)
			else:
				left2right[left] = [right]
		for key in left2right:
			left2right[key]=sorted(left2right[key])
			# print(target_span_list, target_span_list[:target_i+1])
		lenA=len(example.text_a.split())
		lenB=len(example.text_b.split())
		alignment_pair=[]
		new_key = []
		for i in range(lenA):
			if i in left2right:
				if len(new_key) == 0:
					new_key = [i]
				elif left2right[new_key[-1]] == left2right[i] and i == (new_key[-1] + 1):
					new_key.append(i)
				else:
					alignment_pair.append((new_key, left2right[new_key[-1]]))
					new_key = [i]
		alignment_pair.append((new_key, left2right[new_key[-1]]))
		source_id_tagged=set()
		target_id_tagged=set()
		for pair in alignment_pair:
			source_id_list, target_id_list = pair
			# source_id_list=source_id_list[:args.max_span_size]
			# target_id_list=target_id_list[:args.max_span_size]
			source_id_tagged = source_id_tagged | set(source_id_list)
			target_id_tagged = target_id_tagged | set(target_id_list)
			tag_idx=0
			for d in range(min(len(target_id_list), args.max_span_size)):
				if d==len(target_id_list)-1 or d==args.max_span_size-1:
					tag_idx+=target_id_list[0]+1
				else:
					tag_idx+=lenB-d
			if len(source_id_list)>args.max_span_size:
				label_s2t[source_id_list[args.max_span_size-1]][args.max_span_size-1]=tag_idx
				for my_idx in range(args.max_span_size, len(source_id_list)):
					label_s2t[source_id_list[my_idx]][0]=tag_idx
			else:
				label_s2t[source_id_list[-1]][len(source_id_list)-1]=tag_idx
			tag_idx=0
			for d in range(min(len(source_id_list), args.max_span_size)):
				if d==len(source_id_list)-1 or d==args.max_span_size-1:
					tag_idx+=source_id_list[0]+1
				else:
					tag_idx+=lenA-d
			if len(target_id_list)>args.max_span_size:
				label_t2s[target_id_list[args.max_span_size-1]][args.max_span_size - 1] = tag_idx
				for my_idx in range(args.max_span_size, len(target_id_list)):
					label_t2s[target_id_list[my_idx]][0]=tag_idx
			else:
				label_t2s[target_id_list[-1]][len(target_id_list)-1]=tag_idx
		for i in range(lenA):
			if i not in source_id_tagged:
				label_s2t[i][0]=0
		for i in range(lenB):
			if i not in target_id_tagged:
				label_t2s[i][0]=0
		# print(example)
		# print(label_s2t)
		# print(label_t2s)
		# sys.exit()
		features.append(Features(input_ids_a_and_b=input_ids_a_and_b, input_ids_b_and_a=input_ids_b_and_a, attention_mask=attention_mask, segment_ids_a_and_b=token_type_ids_a_and_b,
		                         segment_ids_b_and_a=token_type_ids_b_and_a,sent1_valid_ids=sent1_valid_ids, sent2_valid_ids=sent2_valid_ids, sent1_wordpiece_length=sent1_wordpiece_length,
		                         sent2_wordpiece_length=sent2_wordpiece_length,label_s2t=label_s2t, label_t2s=label_t2s))
	return features

def create_Data_Loader(data_examples, args, set_type='train', batchsize=16, max_seq_length=128, tokenizer=None):
	data_features = convert_examples_to_features(data_examples, set_type, max_seq_length, tokenizer, args)
	all_input_ids_a_and_b = torch.tensor([f.input_ids_a_and_b for f in data_features], dtype=torch.long)
	all_input_ids_b_and_a = torch.tensor([f.input_ids_b_and_a for f in data_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.attention_mask for f in data_features], dtype=torch.long, requires_grad=False)
	all_segment_ids_a_and_b = torch.tensor([f.segment_ids_a_and_b for f in data_features], dtype=torch.long)
	all_segment_ids_b_and_a = torch.tensor([f.segment_ids_b_and_a for f in data_features], dtype=torch.long)
	all_sent1_valid_ids = torch.tensor([f.sent1_valid_ids for f in data_features], dtype=torch.long)
	all_sent2_valid_ids = torch.tensor([f.sent2_valid_ids for f in data_features], dtype=torch.long)
	all_sent1_wordpiece_length = torch.tensor([f.sent1_wordpiece_length for f in data_features], dtype=torch.long)
	all_sent2_wordpiece_length = torch.tensor([f.sent2_wordpiece_length for f in data_features], dtype=torch.long)
	if set_type == 'train':
		all_label_s2t = torch.tensor([f.label_s2t for f in data_features])
		all_label_t2s = torch.tensor([f.label_t2s for f in data_features])
		dataset = TensorDataset(all_input_ids_a_and_b, all_input_ids_b_and_a, all_input_mask, all_segment_ids_a_and_b, all_segment_ids_b_and_a, all_sent1_valid_ids, all_sent2_valid_ids, all_sent1_wordpiece_length, all_sent2_wordpiece_length, all_label_s2t, all_label_t2s)
		data_sampler = RandomSampler(dataset)
	else:
		dataset = TensorDataset(all_input_ids_a_and_b, all_input_ids_b_and_a, all_input_mask, all_segment_ids_a_and_b, all_segment_ids_b_and_a, all_sent1_valid_ids, all_sent2_valid_ids, all_sent1_wordpiece_length, all_sent2_wordpiece_length)
		data_sampler = SequentialSampler(dataset)
	dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batchsize)
	return dataloader

def put_a_number_in_bin_and_binary(a, return_fired_idx = False):
	bins = np.array([-11, -6, -4, -3, -2, -1, 0, 1, 2, 3, 5, 10])
	fired_idx = np.digitize(a, bins, right = True)

	if return_fired_idx == True:
		return fired_idx

	d = np.zeros(13)
	d[fired_idx] = 1

	return d

def convert_stateID_to_spanID(stateID, sent_length, args): # 0 is NULL state
	stateID=stateID-1
	if stateID<0:
		return (-1, -1)
	else:
		for span_length in range(1, args.max_span_size+1):
			lower_bound=(span_length-1)*sent_length-int((span_length-1)*(span_length-2)/2)
			upper_bound=span_length*sent_length-int(span_length*(span_length-1)/2)
			if stateID>=lower_bound and stateID<upper_bound:
				return (stateID-lower_bound, span_length) # return (spanID, span_Length)
