import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
from torch import logsumexp
import torch.nn.functional as F
from torch.autograd import Variable
from neural_jacana.utils import *

def normalize_1d_tensor_to_list(tensor):
    nom_list = []
    for i in range(len(tensor)):
        val = tensor[i]
        if 0 <= val <= 127:
            nom_list.append(val)
        elif val < 0:
            nom_list.append(0)
        elif 127 < val:
            nom_list.append(127)
            
    return nom_list

class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super().__init__()
		self.gamma = nn.Parameter(torch.ones(features))
		self.beta = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta

class NeuralWordAligner(nn.Module):
	def __init__(self, args):
		super(NeuralWordAligner, self).__init__()
		self.args = args
		#self.my_device = torch.device('cpu')
		self.my_device = args.my_device
		self.bert_model = BertModel.from_pretrained('neural_jacana/spanbert_hf_base', output_hidden_states=True, output_attentions=True)
		self.attn_proj = nn.Linear(768, 100)
		self.attn_embd = nn.Embedding(1, 100)
		self.default_tag = -1
		self.layer_norm = LayerNorm(768 * 3)
		self.sim_feature_num=12
		span_representation_dim = 768
		self.span_representation_dim=span_representation_dim
		self.mlp1 = nn.Sequential(nn.Linear(self.sim_feature_num, self.sim_feature_num), nn.PReLU(),nn.Linear(self.sim_feature_num, 1))
		self.FF_kernel_spanbert = nn.Sequential(*[nn.Linear((span_representation_dim*3)*4, (span_representation_dim*3)), nn.PReLU(),nn.Linear((span_representation_dim*3), (span_representation_dim*3)), nn.PReLU(), nn.Linear((span_representation_dim*3), self.sim_feature_num)])
		self.mlp2 = nn.Sequential(nn.Linear(self.args.distance_embedding_size, 1))
		# self.mlp2 = nn.Sequential(nn.Linear(args.distance_embedding_size, args.distance_embedding_size), nn.PReLU(), nn.Linear(args.distance_embedding_size,1))
		self.distance_embedding = nn.Embedding(13, self.args.distance_embedding_size)
		self.distance_embedding.weight.requires_grad = False
		self.transition_matrix_dict = {}
		for len_B in range(self.args.max_span_size, self.args.max_sent_length + 1, 1):
			extended_length_B = self.args.max_span_size * len_B - int(self.args.max_span_size * (self.args.max_span_size - 1) / 2)

			tmp_dist_idx_list = np.zeros((extended_length_B + 1, extended_length_B + 1))
			if self.args.use_transition_layer=='True':
				for j in range(extended_length_B + 1):  # 0 is NULL state
					for k in range(extended_length_B + 1):  # k is previous state, j is current state
						tmp_dist_idx_list[j][k]=put_a_number_in_bin_and_binary(self.distortionDistance(k, j, len_B), return_fired_idx=True)
			self.transition_matrix_dict[extended_length_B] = tmp_dist_idx_list

	def _input_likelihood(self,
	                      logits: torch.Tensor,
	                      T: torch.Tensor,
	                      text_mask: torch.Tensor,
	                      hamming_cost: torch.Tensor,
	                      tag_mask: torch.Tensor) -> torch.Tensor:
		"""
		Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible segmentations.
		:param logits: shape (batch_size, sequence_length, max_span_width, num_tags)
		:param T: shape (batch_size, num_tags, num_tags)
		:param text_mask: shape (batch_size, sequence_length)
		:param hamming_cost: shape (batch_size, sequence_length, max_span_width, num_tags)
		:param tag_mask: shape (batch_size, num_tags, num_tags)
		:return: (batch_size,) denominator score
		"""
		batch_size, sequence_length, max_span_width, num_tags = logits.size()
		logits = logits.permute(1,2,0,3).contiguous() # shape: (sequence_length, max_span_width, batch_size, num_tags)
		hamming_cost = hamming_cost.permute(1,2,0,3).contiguous() # shape: (sequence_length, max_span_width, batch_size, num_tags)

		if self.args.use_transition_layer == "False":
			alpha = Variable(torch.FloatTensor([[0.0 for _ in range(batch_size)]]).to(self.my_device), requires_grad=True) # 1 * batch_size
		else:
			alpha = Variable(torch.FloatTensor(np.zeros((1, batch_size, num_tags))).to(self.my_device), requires_grad=True) # 1 * batch_size * num_tags

		# For each j we compute logits for all the segmentations of length j.
		for j in range(sequence_length):
			width = max_span_width
			if j < max_span_width - 1:
				width = j + 1

			# Reverse the alpha so it gets added to the correct logits.
			idx = Variable(torch.LongTensor([i for i in range(j, j - width, -1)]).to(self.my_device))
			reversed_alpha = alpha.index_select(dim=0, index=idx)
			# Tensorize and broadcast along the max_span_width dimension.
			broadcast_alpha = reversed_alpha.view(width, batch_size, -1) # shape: (max_span_width, batch_size, num_tags)
			logits_at_j = logits[j]
			start_indices = Variable(torch.LongTensor(range(width)).to(self.my_device))
			span_factors = logits_at_j.index_select(dim=0, index=start_indices)
			span_costs = hamming_cost[j].index_select(dim=0, index=start_indices)


			# if j==5 and False:
			# 	print(broadcast_alpha.size()) # span_width * bsz * 1 or span_width * bsz * num_tags
			# 	print(span_factors.size()) # span_width * bsz * num_tags
			# 	print(span_costs.size()) # span_width * bsz * num_tags
			# 	sys.exit()
			if self.args.use_transition_layer == "False":
				# Logsumexp the scores over the num_tags axis.
				alpha_along_arglabels = logsumexp(broadcast_alpha + (span_factors + span_costs) * tag_mask[:,0, :], dim=-1)
				# Logsumexp the scores over the width axis.
				alpha_at_j = logsumexp(alpha_along_arglabels, dim=0).view(1, batch_size)
				alpha = torch.cat([alpha, alpha_at_j], dim=0)
			else:
				alpha_along_arglabels = []
				for k in range(num_tags):
					emit_score = ((span_factors[:,:,k] + span_costs[:,:,k]) * tag_mask[:, 0, k]).unsqueeze(-1).repeat(1, 1, num_tags)
					tran_score = T[:,k,:]*tag_mask[:,k,:] # bsz * num_tags
					if j<width:
						tran_score = tran_score.unsqueeze(0).repeat(width, 1, 1)
						tran_score[j:,:,:]=0
					# Logsumexp the scores over the num_tags axis.
					alpha_along_arglabels.append(logsumexp(broadcast_alpha+tran_score+emit_score, dim=-1, keepdim=True))
				alpha_along_arglabels = torch.cat(alpha_along_arglabels, dim=-1)
				# Logsumexp the scores over the width axis.
				alpha_at_j = logsumexp(alpha_along_arglabels, dim=0, keepdim=True) # 1*bsz *num_tags
				alpha = torch.cat([alpha, alpha_at_j], dim=0)

		if self.args.use_transition_layer == "True":
			alpha = logsumexp(alpha, dim=-1)
		# Get the last positions for all alphas in the batch.
		actual_lengths = torch.sum(text_mask, dim=1).view(1, batch_size)
		# Finally we return the alphas along the last "valid" positions.
		# shape: (batch_size)
		partition = alpha.gather(dim=0, index=actual_lengths)
		return partition

	def _joint_likelihood(self,
	                      logits: torch.Tensor,
	                      T: torch.Tensor,
	                      tags: torch.Tensor,
	                      text_mask: torch.LongTensor) -> torch.Tensor:
		"""
		Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
		:param logits: shape (batch_size, sequence_length, max_span_width, num_tags)
		:param T: shape (batch_size, num_tags, num_tags)
		:param tags: golden label shape (batch_size, sequence_length, max_span_width)
		:param text_mask: shape (batch_size, sequence_length)
		:return: (batch_size,) numerator score
		"""
		batch_size, sequence_length, max_span_width, num_tags = logits.shape
		logits = logits.permute(1,2,0,3).contiguous() # shape: (sequence_length, max_span_width, batch_size, num_tags)
		# Transpose to shape: (sequence_length, batch_size)
		text_mask = text_mask.float().transpose(0, 1).contiguous()
		# Transpose to shape: (sequence_length, max_span_width, batch_size)
		tags = tags.permute(1, 2, 0).contiguous()

		default_tags = Variable(self.default_tag * torch.ones(batch_size).long().to(self.my_device))

		numerator = 0.0
		for j in range(sequence_length):
			for d in range(min(max_span_width, j+1)):
				current_tag = tags[j][d]
				valid_tag_mask = (current_tag != default_tags).long()
				current_tag = current_tag*valid_tag_mask
				# Reshape for gather operation to follow.
				current_tag = current_tag.view(batch_size, 1)
				# The score for using current_tag
				emit_score = logits[j][d].gather(dim=1, index=current_tag).squeeze(1) * valid_tag_mask * text_mask[j]
				numerator += emit_score
				if j>=1 and self.args.use_transition_layer=='True' and j-(d+1) >=0 and valid_tag_mask.sum()>0:
					for last_d in range(min(max_span_width, j)):
						last_tag = tags[j-(d+1)][last_d]
						last_valid_tag_mask = (last_tag != default_tags).long()
						# print(last_tag, last_valid_tag_mask)
						# print(last_tag, last_valid_tag_mask, current_tag.view(-1), valid_tag_mask)
						# print(last_valid_tag_mask)
						last_tag = last_tag*last_valid_tag_mask
						last_tag = last_tag.view(batch_size, 1, 1)
						last_tag = last_tag.repeat(1, num_tags, 1)
						trans_score = T.gather(dim=2, index=last_tag).squeeze(-1)
						trans_score = trans_score.gather(dim=1, index=current_tag).squeeze(1)
						trans_score = trans_score * last_valid_tag_mask * valid_tag_mask * text_mask[j]
						# print(trans_score)
						numerator += trans_score
		# print(tags)
		# sys.exit()
		return numerator

	def viterbi_tags(self,
	                 logits: Variable,
	                 T: Variable,
	                 text_mask: Variable,
	                 tag_masks: Variable,
	                 reverse = False):
		"""
		:param logits: shape (batch_size, sequence_length, max_span_width, num_tags)
		:param T: shape (batch_size, num_tags, num_tags)
		:param text_mask: shape (batch_size, sequence_length)
		:param tag_mask: shape (batch_size, num_tags, num_tags)
		:return:
		"""
		batch_size, max_seq_length, max_span_width, num_classes = logits.size()

		# Get the tensors out of the variables
		tag_masks = tag_masks[:,0,:] # shape (batch_size, num_tags)
		logits, text_mask, tag_masks = logits.data, text_mask.data, tag_masks.data
		sequence_lengths = torch.sum(text_mask, dim=-1)

		all_tags = []
		for logits_ex, tag_mask, sequence_length, transition_matrix in zip(logits, tag_masks, sequence_lengths, T):

			# logits_ex: max_seq_length * max_span_width * num_classes
			emission_matrix=[]
			for d in range(max_span_width):
				extracted_logit=logits_ex[d:sequence_length,d:d+1,:int(sum(tag_mask))] # num_spans * 1 * num_tags
				extracted_logit=extracted_logit.squeeze(1)
				extracted_logit=torch.cat([extracted_logit[:,1:], extracted_logit[:,0:1]], 1)
				emission_matrix.append(extracted_logit)
			emission_matrix=torch.cat(emission_matrix, dim=0)
			# We pass the logits to ``viterbi_decoder``.
			optimal_sequence = self.viterbi_decoder(emission_matrix, transition_matrix, sequence_length, sum(tag_mask)-1)
			predA = set()
			target_sent_length = int((self.args.max_span_size - 1) / 2 + (sum(tag_mask) - 1) / self.args.max_span_size)
			for i, stateID in enumerate(optimal_sequence):
				if stateID==0:
					continue
				stateID-=1
				for d in range(1, self.args.max_span_size+1):
					lower_bound=(d-1)*target_sent_length - int((d-1)*(d-2)/2)
					upper_bound = d * target_sent_length - int(d * (d - 1) / 2)
					if stateID >= lower_bound and stateID < upper_bound:
						spanID, spanLength = stateID - lower_bound, d
						for kk in range(spanLength):
							if reverse:
								predA.add(str(spanID + kk) + '-' + str(i))
							else:
								predA.add(str(i) + '-' + str(spanID + kk))
						break
			all_tags.append(predA)

		return all_tags

	def viterbi_decoder(self, emission_matrix, transition_matrix, len_A, extended_length_B):
		"""
		:param emission_matrix:  extended_length_A * (extended_length_B + 1), word/phrase pair interaction matrix
		:param transition_matrix: (extended_length_B + 1) * (extended_length_B + 1), state transition matrix
		:param len_A: source sentence length
		:param len_B: target sentence length
		:return: optimal sequence
		"""
		emission_matrix=emission_matrix.data.cpu().numpy()
		transition_matrix=transition_matrix.data.cpu().numpy()
		len_A=len_A.data.cpu().numpy()
		extended_length_B=int(extended_length_B.data.cpu().numpy())
		T1=np.zeros((len_A, extended_length_B + 1), dtype=float)
		T2=np.zeros((len_A, extended_length_B + 1), dtype=int)
		T3=np.zeros((len_A, extended_length_B + 1), dtype=int)
		for j in range(extended_length_B+1):
			# T1[0][j]=start_transition[j] + emission_matrix[0][j-1]
			T1[0][j] = emission_matrix[0][j - 1]
			T2[0][j]=-1
			T3[0][j]=1 # span size

		for i in range(1, len_A):
			global_max_val = float("-inf")
			for j in range(extended_length_B+1):
				max_val = float("-inf")
				for span_size in range(1, min(i + 1, self.args.max_span_size) + 1):
					for k in range(extended_length_B+1):
						if i-span_size>=0:
							cur_val = T1[i-span_size][k] + transition_matrix[j][k] + emission_matrix[i - (span_size-1) +(span_size-1)*len_A-int((span_size-1)*(span_size-2)/2)][j-1]
							# if i==len_A-1:
							# 	cur_val+=end_transition[j]
						else:
							cur_val = emission_matrix[i - (span_size-1) + (span_size-1)*len_A - int((span_size-1)*(span_size-2)/2)][j-1]
						if cur_val>max_val:
							T1[i][j]=cur_val
							T2[i][j]=k
							T3[i][j]=span_size
							max_val=cur_val
				if max_val>global_max_val:
					global_max_val=max_val
		optimal_sequence=[]
		max_val=float("-inf")
		max_idx=-1
		for j in range(extended_length_B+1):
			if T1[len_A-1][j]>max_val:
				max_idx=j
				max_val=T1[len_A-1][j]
		# optimal_sequence = [max_idx] + optimal_sequence
		# for i in range(len_A - 1, 0, -1):
		# 	optimal_sequence = [T2[i][max_idx]] + optimal_sequence
		# 	max_idx = T2[i][max_idx]
		i=len_A-1
		while i>=0:
			optimal_element=[max_idx]*T3[i][max_idx]
			optimal_sequence=optimal_element+optimal_sequence
			new_i = i - T3[i][max_idx]
			new_max_idx = T2[i][max_idx]
			i=new_i
			max_idx=new_max_idx

		return optimal_sequence

	def _compose_span_representations(self, seq1, seq2):
		"""
		:param seq1: bsz*max_sent1_length*768, unigram representations arranged first, then followed by CLS representations
		:param seq2: bsz*max_sent2_length*768, unigram representations arranged first, then followed by CLS representations
		:return: (seq1, seq2): bsz* num_all_spans *768
		"""
		seq1_spans=[]
		seq2_spans=[]
		bsz, max_sent1_length, _ = seq1.size()
		bsz, max_sent2_length, _ = seq2.size()
		idx = Variable(torch.LongTensor([0])).to(self.my_device)
		for d in range(self.args.max_span_size):
			seq1_spans_d=[]
			for i in range(max_sent1_length-d):
				current_ngrams=seq1[:,i:i+d+1,:] # bsz * current_span_size * 768
				alpha = torch.matmul(self.attn_proj(current_ngrams), self.attn_embd(idx).view(-1,1)) # bsz * span_size * 1
				alpha = F.softmax(alpha, dim=1)
				seq1_spans_d.append(torch.cat([current_ngrams[:,:1,:], current_ngrams[:,-1:,:], torch.matmul(alpha.permute(0, 2, 1), current_ngrams)], dim=-1))
			seq1_spans_d = [torch.zeros((bsz, 1, self.span_representation_dim*3)).to(self.my_device)]* (max_sent1_length - len(seq1_spans_d)) + seq1_spans_d
			seq1_spans_d=torch.cat(seq1_spans_d, dim=1) # bsz * max_sent1_length * dim
			seq1_spans.append(seq1_spans_d)
			seq2_spans_d=[]
			for i in range(max_sent2_length-d):
				current_ngrams=seq2[:,i:i+d+1,:]
				alpha = torch.matmul(self.attn_proj(current_ngrams), self.attn_embd(idx).view(-1, 1))  # bsz * span_size * 1
				alpha = F.softmax(alpha, dim=1)
				seq2_spans_d.append(torch.cat([current_ngrams[:, :1, :], current_ngrams[:, -1:, :], torch.matmul(alpha.permute(0, 2, 1), current_ngrams)], dim=-1))
			seq2_spans_d = [torch.zeros((bsz, 1, self.span_representation_dim*3)).to(self.my_device)]* (max_sent2_length-len(seq2_spans_d)) + seq2_spans_d
			seq2_spans_d=torch.cat(seq2_spans_d, dim=1) # bsz * max_sent2_length * dim
			seq2_spans.append(seq2_spans_d)
		return torch.cat(seq1_spans, dim=1), torch.cat(seq2_spans, dim=1)

	def compute_sim_cube_FF_spanbert(self, seq1, seq2):
		'''
		:param seq1: bsz * len_seq1 * dim
		:param seq2: bsz * len_seq2 * dim
		:return:
		'''
		def compute_sim_FF(prism1, prism2):
			features = torch.cat([prism1, prism2, torch.abs(prism1 - prism2), prism1 * prism2], dim=-1)
			FF_out = self.FF_kernel_spanbert(features)
			return FF_out.permute(0,3,1,2)

		def compute_prism(seq1, seq2):
			prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
			prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
			prism1 = prism1.permute(1, 2, 0, 3).contiguous()
			prism2 = prism2.permute(1, 0, 2, 3).contiguous()
			return compute_sim_FF(prism1, prism2)

		sim_cube = torch.Tensor(seq1.size(0), self.sim_feature_num, seq1.size(1), seq2.size(1)) # bsz * feature_num * len_seq1 * len_seq2
		sim_cube = sim_cube.to(self.my_device)
		sim_cube[:, 0:self.sim_feature_num] = compute_prism(seq1, seq2)
		return sim_cube

	def create_pad_cube(self, sent1_lengths, sent2_lengths):
		pad_cube = []
		max_len1 = max(sent1_lengths)
		max_len2 = max(sent2_lengths)

		for s1_length, s2_length in zip(sent1_lengths, sent2_lengths):
			pad_mask = np.ones((max_len1, max_len2))
			pad_mask[:s1_length, :s2_length] = 0
			pad_cube.append(pad_mask)

		pad_cube = np.array(pad_cube)
		return torch.from_numpy(pad_cube).float().to(self.my_device).unsqueeze(0)

	def distortionDistance(self, state_i, state_j, sent_length):
		start_i, size_i=convert_stateID_to_spanID(state_i, sent_length, self.args)
		start_j, size_j=convert_stateID_to_spanID(state_j, sent_length, self.args)
		return start_j - (start_i + size_i - 1) - 1

	def forward(
			self,
			input_ids_a_and_b=None,
			input_ids_b_and_a=None,
			attention_mask=None,
			token_type_ids_a_and_b=None,
			token_type_ids_b_and_a=None,
			sent1_valid_ids=None,
			sent2_valid_ids=None,
			sent1_wordpiece_length=None,
			sent2_wordpiece_length=None,
			labels_s2t=None,
			labels_t2s=None,
	):
		sent1_lengths = torch.sum(sent1_valid_ids > 0, dim=1).cpu().numpy()
		sent2_lengths = torch.sum(sent2_valid_ids > 0, dim=1).cpu().numpy()
		max_len1 = max(sent1_lengths)
		max_len2 = max(sent2_lengths)
		# print(sent1_lengths)
		# print(sent2_lengths)
		outputs = self.bert_model(
			input_ids=input_ids_a_and_b,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids_a_and_b,
		)
		all_layer_hidden_states, all_layer_attention_weights = outputs[-2:]
		# print(outputs[0].size()) # bsz*128*768
		# print(all_layer_hidden_states[-1].size()) # bsz*128*768
		# print(all_layer_attention_weights[-1].size()) # bsz*12*128*128
		all_hidden_states = all_layer_hidden_states[-1]  # the last layer, bsz*128*768
		batch_size, max_sequence_length, hidden_dim = outputs[0].size()
		#print(batch_size)
		#print(all_hidden_states)
		#print(outputs[0]) # the same thing as all_hidden_states
		outputs_b_and_a = self.bert_model(
			input_ids=input_ids_b_and_a,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids_b_and_a,
		)
		all_layer_hidden_states_b_and_a, all_layer_attention_weights_b_and_a = outputs_b_and_a[-2:]
		all_hidden_states_b_and_a = all_layer_hidden_states_b_and_a[-1]
		seq1 = []
		seq2 = []
		for i in range(batch_size):
			for j in range(sent1_lengths[i]-1):
				diff=sent1_valid_ids[i][j+1]-sent1_valid_ids[i][j]
				if diff>1:
					position=sent1_valid_ids[i][j]
					# print(j,position)
					all_hidden_states[i][position:position+1,:]=torch.mean(all_hidden_states[i][position:position+diff, :], dim=0, keepdim=True)
					position=sent1_valid_ids[i][j]+sent2_wordpiece_length[i]+1
					# print(j,position)
					# print(sent2_wordpiece_length)
					# sys.exit()
					all_hidden_states_b_and_a[i][position:position+1,:]=torch.mean(all_hidden_states_b_and_a[i][position:position+diff, :], dim=0, keepdim=True)
			# print(sent1_valid_ids[i])
			# print(sent1_valid_ids[i]+sent2_wordpiece_length[i]+1)
			sent1_valid_ids_part2 = (sent1_valid_ids[i]+sent2_wordpiece_length[i]+1)*(sent1_valid_ids[i]>0).int()
			sent1_valid_ids_part2 = torch.tensor(normalize_1d_tensor_to_list(sent1_valid_ids_part2)).to(self.my_device) ###### add this
			# print(sent1_valid_ids_part2)
			# sys.exit()
			seq1.append(torch.index_select(all_hidden_states[i], 0, sent1_valid_ids[i]).unsqueeze(0) + torch.index_select(all_hidden_states_b_and_a[i], 0, sent1_valid_ids_part2).unsqueeze(0))
			for j in range(sent2_lengths[i]-1):
				diff=sent2_valid_ids[i][j+1]-sent2_valid_ids[i][j]
				if diff>1:
					position=sent2_valid_ids[i][j]
					all_hidden_states[i][position:position+1, :]=torch.mean(all_hidden_states[i][position:position+diff,:], dim=0, keepdim=True)
					position=sent2_valid_ids[i][j]-sent1_wordpiece_length[i]-1
					all_hidden_states_b_and_a[i][position:position+1,:]=torch.mean(all_hidden_states_b_and_a[i][position:position+diff, :], dim=0, keepdim=True)
			sent2_valid_ids_part2 = (sent2_valid_ids[i]-sent1_wordpiece_length[i]-1)*(sent2_valid_ids[i]>0).int()
			sent2_valid_ids_part2 = torch.tensor(normalize_1d_tensor_to_list(sent2_valid_ids_part2)).to(self.my_device) ###### add this
			seq2.append(torch.index_select(all_hidden_states[i], 0, sent2_valid_ids[i]).unsqueeze(0) + torch.index_select(all_hidden_states_b_and_a[i], 0, sent2_valid_ids_part2).unsqueeze(0))
		seq1 = torch.cat(seq1, 0) # bsz*128*768
		seq2 = torch.cat(seq2, 0) # bsz*128*768
		seq1 = seq1[:, :max_len1, :]
		seq2 = seq2[:, :max_len2, :]
		seq1_spans, seq2_spans = self._compose_span_representations(seq1, seq2) # bsz * num_all_spans * dim
		# print(seq1_spans.size())
		# print(seq2_spans.size())

		# bsz * feature_num * num_all_source_spans * num_all_target_spans
		simCube_context_spanbert = self.compute_sim_cube_FF_spanbert(self.layer_norm(seq1_spans), self.layer_norm(seq2_spans))

		simCube_context_spanbert_A2B = F.pad(simCube_context_spanbert, (1, 0), 'constant', 0)
		output_both_A2B = self.mlp1(simCube_context_spanbert_A2B.permute(0, 2, 3, 1)).squeeze(-1) # bsz * num_all_source_spans * (num_all_target_spans + 1), 0 is inserted at the head position

		simCube_context_spanbert_B2A = F.pad(simCube_context_spanbert.permute(0, 1, 3, 2), (1, 0), 'constant', 0)
		output_both_B2A = self.mlp1(simCube_context_spanbert_B2A.permute(0, 2, 3, 1)).squeeze(-1) # bsz * num_all_target_spans * (num_all_source_spans + 1)

		num_tags_B = output_both_A2B.size(-1)
		num_tags_A = output_both_B2A.size(-1)

		tag_B_valid_ids=[[0] for i in range(batch_size)]
		tag_A_valid_ids=[[0] for i in range(batch_size)]
		for i in range(batch_size):
			for d in range(1, self.args.max_span_size+1, 1):
				tag_B_valid_ids[i]+=[k+(d-1)*max_len2 for k in range(d, sent2_lengths[i]+1, 1)]
				tag_A_valid_ids[i]+=[k+(d-1)*max_len1 for k in range(d, sent1_lengths[i]+1, 1)]
			tag_B_valid_ids[i]+=(num_tags_B-len(tag_B_valid_ids[i]))*[num_tags_B-1]
			tag_A_valid_ids[i]+=(num_tags_A-len(tag_A_valid_ids[i]))*[num_tags_A-1]
		# print(sent1_lengths)
		# print(sent2_lengths)
		# print(tag_A_valid_ids)
		# print(tag_B_valid_ids)
		# sys.exit()
		tag_B_valid_ids = torch.LongTensor(tag_B_valid_ids).to(self.my_device).view(batch_size, 1, num_tags_B)
		tag_A_valid_ids = torch.LongTensor(tag_A_valid_ids).to(self.my_device).view(batch_size, 1, num_tags_A)
		tag_B_valid_ids = tag_B_valid_ids.repeat(1, output_both_A2B.size(1), 1)
		tag_A_valid_ids = tag_A_valid_ids.repeat(1, output_both_B2A.size(1), 1)
		output_both_A2B = output_both_A2B.gather(dim=-1, index=tag_B_valid_ids)
		output_both_B2A = output_both_B2A.gather(dim=-1, index=tag_A_valid_ids)

		logits_A2B = output_both_A2B.view(batch_size, self.args.max_span_size, -1, num_tags_B)
		logits_A2B = logits_A2B.permute(0,2,1,3) # shape (batch_size, sequence_length, max_span_width, num_tags)
		logits_B2A = output_both_B2A.view(batch_size, self.args.max_span_size, -1, num_tags_A)
		logits_B2A = logits_B2A.permute(0,2,1,3)

		# print(logits_A2B.size())
		# print(logits_B2A.size())

		batch_dist_idx_list_B = np.zeros((batch_size, num_tags_B, num_tags_B), dtype=int) # the last embedding dim is for padding
		tag_masks_sentB = np.zeros((batch_size, num_tags_B, num_tags_B), dtype=int)
		batch_dist_idx_list_A = np.zeros((batch_size, num_tags_A, num_tags_A), dtype=int)
		tag_masks_sentA = np.zeros((batch_size, num_tags_A, num_tags_A), dtype=int)
		for d in range(batch_size):
			if sent2_lengths[d] < self.args.max_span_size:
				extended_length_B = self.args.max_span_size*self.args.max_span_size-int(self.args.max_span_size*(self.args.max_span_size-1)/2)
			else:
				extended_length_B = self.args.max_span_size*sent2_lengths[d]-int(self.args.max_span_size*(self.args.max_span_size-1)/2)

			batch_dist_idx_list_B[d,:(extended_length_B+1),:(extended_length_B+1)] = self.transition_matrix_dict[extended_length_B]
			tag_masks_sentB[d,:(extended_length_B+1),:(extended_length_B+1)]=np.ones((extended_length_B+1, extended_length_B+1))

			extended_length_A = self.args.max_span_size*sent1_lengths[d]-int(self.args.max_span_size*(self.args.max_span_size-1)/2)
			batch_dist_idx_list_A[d,:(extended_length_A+1),:(extended_length_A+1)] = self.transition_matrix_dict[extended_length_A]
			tag_masks_sentA[d,:(extended_length_A+1),:(extended_length_A+1)]=np.ones((extended_length_A+1, extended_length_A+1))

		extracted_tensor_idx_B = Variable(torch.from_numpy(batch_dist_idx_list_B).type(torch.LongTensor)).to(self.my_device)
		transition_matrix_sentB = self.mlp2(self.distance_embedding(extracted_tensor_idx_B)).squeeze(-1)
		extracted_tensor_idx_A = Variable(torch.from_numpy(batch_dist_idx_list_A).type(torch.LongTensor)).to(self.my_device)
		transition_matrix_sentA = self.mlp2(self.distance_embedding(extracted_tensor_idx_A)).squeeze(-1)
		if self.args.use_transition_layer == "False":
			transition_matrix_sentB = transition_matrix_sentB*0
			transition_matrix_sentA = transition_matrix_sentA*0
		tag_masks_sentB = Variable(torch.from_numpy(tag_masks_sentB).type(torch.FloatTensor)).to(self.my_device) # bsz * num_tags_B * num_tags_B
		tag_masks_sentA = Variable(torch.from_numpy(tag_masks_sentA).type(torch.FloatTensor)).to(self.my_device) # bsz * num_tags_A * num_tags_A
		text_masks_sentA=(sent1_valid_ids>0).int()[:,:max_len1] # bsz * max_sent1
		text_masks_sentB=(sent2_valid_ids>0).int()[:,:max_len2] # bsz * max_sent2

		# print(transition_matrix_sentB.size())
		# print(transition_matrix_sentA.size())
		# print(tag_masks_sentB.size())
		# print(tag_masks_sentA.size())
		# print(text_masks_sentB.size())
		# print(text_masks_sentA.size())
		# sys.exit()

		if self.training:
			gold_tags_sentB = labels_s2t[:,:max_len1,:]
			# print(gold_tags_sentB.type()) # torch.LongTensor
			# sys.exit()
			numerator_A2B = self._joint_likelihood(logits_A2B, transition_matrix_sentB, gold_tags_sentB, text_masks_sentA)
			# print(numerator_A2B)
			zeros = Variable(torch.zeros(batch_size, max_len1, self.args.max_span_size, num_tags_B).float().to(self.my_device))
			gold_tags_sentB_without_negative_one = gold_tags_sentB * (gold_tags_sentB > -1)
			scattered_tags = zeros.scatter_(-1, gold_tags_sentB_without_negative_one.unsqueeze(dim=-1), 1)
			hamming_cost = (1-scattered_tags)
			# negative_ones_only = (gold_tags_sentB.unsqueeze(-1).repeat(1,1,1,num_tags_B)==-1).float()
			# hamming_cost += negative_ones_only
			# hamming_cost = (hamming_cost > 0).float()
			denominator_A2B = self._input_likelihood(logits_A2B, transition_matrix_sentB, text_masks_sentA, hamming_cost, tag_masks_sentB)
			# print(denominator_A2B)
			loss_A2B = (denominator_A2B-numerator_A2B).sum()#/batch_size

			gold_tags_sentA = labels_t2s[:, :max_len2, :]
			numerator_B2A = self._joint_likelihood(logits_B2A, transition_matrix_sentA, gold_tags_sentA, text_masks_sentB)
			zeros = Variable(torch.zeros(batch_size, max_len2, self.args.max_span_size, num_tags_A).float().to(self.my_device))
			gold_tags_sentA_without_negative_one = gold_tags_sentA * (gold_tags_sentA > -1)
			scattered_tags = zeros.scatter_(-1, gold_tags_sentA_without_negative_one.unsqueeze(dim=-1), 1)
			hamming_cost = (1 - scattered_tags)
			# negative_ones_only = (gold_tags_sentA.unsqueeze(-1).repeat(1, 1, 1, num_tags_A) == -1).float()
			# hamming_cost += negative_ones_only
			# hamming_cost = (hamming_cost > 0).float()
			denominator_B2A = self._input_likelihood(logits_B2A, transition_matrix_sentA, text_masks_sentB, hamming_cost, tag_masks_sentA)
			loss_B2A = (denominator_B2A-numerator_B2A).sum()#/batch_size

			return loss_A2B+loss_B2A

		else:
			optimal_sequence_A2B = self.viterbi_tags(logits_A2B, transition_matrix_sentB, text_masks_sentA, tag_masks_sentB, reverse=False)
			optimal_sequence_B2A = self.viterbi_tags(logits_B2A, transition_matrix_sentA, text_masks_sentB, tag_masks_sentA, reverse=True)
			return [optimal_sequence_A2B[i] & optimal_sequence_B2A[i] for i in range(batch_size)]
