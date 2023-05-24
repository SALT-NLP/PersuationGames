import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import TensorDataset

Strategies = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]

strategies2id = {"No Strategy": 0, "Identity Declaration": 1, "Accusation": 2, "Interrogation": 3, "Call for Action": 4, "Defense": 5, "Evidence": 6}
role2id = {'Moderator': 0, 'Villager': 1, 'Werewolf': 2, 'Seer': 3, 'Robber': 4, 'Troublemaker': 5, 'Tanner': 6, 'Drunk': 7, 'Hunter': 8, 'Mason': 9,  'Insomniac': 10, 'Minion': 11, 'Doppelganger': 12}


def read_data(args, logger, strategy, tokenizer, mode):
	all_input_ids = []
	all_input_mask = []
	all_label = []
	all_video_features = []
	if isinstance(args.dataset, str):
		args.dataset = (args.dataset,)
	for dataset in args.dataset:
		logger.info(f'{dataset} dataset:')
		file_path = os.path.join('data', dataset, f'{mode}.json')
		with open(file_path, 'r') as f:
			games = json.load(f)

		id = 0
		video_features = None
		for game_idx, game in enumerate(games):
			dialogues = game.pop("Dialogue")
			context = [[]] * args.context_size
			if args.video:
				if dataset == 'Ego4D':
					# video_features = np.load(os.path.join('data', args.dataset, 'video_feature', f'{game["EG_ID"]}_{game["Game_ID"]}.npy'))
					video_features = np.load(os.path.join(args.video_path, f'{game["EG_ID"]}_{game["Game_ID"]}.npy'))
					# logger.info(f'{game["EG_ID"]}_{game["Game_ID"]}.npy, {video_features.shape}')
				elif dataset == 'Youtube':
					video_features = np.load(os.path.join(args.video_path, f'{game["video_name"]}_{game["Game_ID"]}.npy'))
				else:
					raise NotImplementedError
				logger.info(f'Loading video features from {args.video_path}')

			for rid, record in enumerate(dialogues):
				id += 1
				label = 1 if strategy in record['annotation'] else 0
				utterance = record['utterance']

				tokens = [tokenizer.cls_token]
				if args.context_size != 0:
					for cxt in context[-args.context_size:]:
						tokens += cxt + ['<end of text>']
					tokens += [tokenizer.sep_token]
				context.append(tokenizer.tokenize(utterance))
				tokens += context[-1] + [tokenizer.sep_token]
				if len(tokens) > args.max_seq_length:
					logger.info(f'too long, {len(tokens)}')
					tokens = [tokenizer.cls_token] + tokens[-args.max_seq_length + 1:]
					logger.info(len(tokens), tokens)

				input_ids = tokenizer.convert_tokens_to_ids(tokens)
				input_mask = [1] * len(input_ids)

				assert len(tokens) <= args.max_seq_length, f"{len(tokens)}, {utterance}"

				padding_length = args.max_seq_length - len(input_ids)
				input_ids += [tokenizer.pad_token_id] * padding_length
				input_mask += [0] * padding_length

				assert len(input_ids) == args.max_seq_length
				assert len(input_mask) == args.max_seq_length

				if id % 2000 == 1:
					logger.info("*** Example ***")
					logger.info(f"guid: {id}")
					logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
					logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
					logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
					logger.info(f"label: {label}")

				all_input_ids.append(input_ids)
				all_input_mask.append(input_mask)
				all_label.append(label)

				if args.video:
					# video_feature = video_features[record["Rec_Id"] - 1][1]  # use center crop
					video_feature = video_features[record["Rec_Id"] - 1]  # use three crops

					# Use video context
					# if args.context_size > 0:
					# 	features = [video_feature]
					# 	for i in range(1, args.context_size + 1):
					# 		feat = video_features[max(record["Rec_Id"] - 1 - i, 0)]
					# 		features.insert(0, feat)
					# 	video_feature = np.concatenate(features, axis=0)

					all_video_features.append(video_feature)

	if args.video:
		Dataset = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long),
								torch.tensor(all_input_mask, dtype=torch.long),
								torch.tensor(all_label, dtype=torch.long),
								torch.tensor(all_video_features, dtype=torch.float32))
	else:
		Dataset = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long),
								torch.tensor(all_input_mask, dtype=torch.long),
								torch.tensor(all_label, dtype=torch.long))
	return Dataset


def read_data_for_deduction(args, logger, mode, utterance_encodings):
	all_inputs = []
	all_label = []
	guid = 0
	for dataset in args.dataset:
		file_path = os.path.join('data', dataset, f'{mode}.json')
		with open(file_path, 'r') as f:
			games = json.load(f)
		id = 0
		for game in games:
			lst_id = id
			player2start_role = {}
			dialogues = game.pop("Dialogue")
			logger.info(f'{game}')
			for player, start_role in zip(game['playerNames'], game['startRoles']):
				player2start_role[player] = role2id[start_role]
			for i, voter in enumerate(game['playerNames']):
				if game['startRoles'][i] == 'Moderator':
					continue
				for j, voted in enumerate(game['playerNames']):
					if game['startRoles'][j] == 'Moderator':
						continue
					guid += 1
					label = 1 if game['votingOutcome'][i] == j else 0
					inputs = []
					id = lst_id
					for record in dialogues:
						speaker_encoding = [1 if record['speaker'] == voter else 0, 1 if record['speaker'] == voted else 0]
						strategy_encoding = [0] * len(Strategies)
						for k, strategy in enumerate(Strategies):
							if strategy in record['annotation']:
								strategy_encoding[k] = 1
						start_role_encoding = [0] * len(role2id)
						if record['speaker'] not in player2start_role:
							if record['speaker'] not in ("Automated", "Game Audio", "Audio", "Siri Voice", "Siri", "Timer", "Twitch Alert", "Voiceover", ) and not record['speaker'].startswith("Speaker "):
								logger.warning(f"non-exist speaker {record['speaker']}, give role of moderator")
							start_role_encoding[0] = 1
						else:
							start_role_encoding[player2start_role[record['speaker']]] = 1
						inputs.append(utterance_encodings[id] + speaker_encoding + strategy_encoding + start_role_encoding)

						if guid % 1000 == 1 and (id - lst_id) % 1000 == 1:
							logger.info("*** Example ***")
							logger.info(f"guid: {guid}")
							logger.info(record)
							logger.info(f"speaker encoding: {speaker_encoding}")
							logger.info(f"strategy encoding: {strategy_encoding}")
							logger.info(f"start role encoding: {start_role_encoding}")
							# logger.info(f"inputs: {inputs}")
							logger.info(f"label: {label}")
						id += 1
					all_inputs.append(inputs)
					all_label.append(label)

	logger.info(f"total datapoints: {guid}")
	all_inputs = [torch.tensor(inputs, dtype=torch.float32) for inputs in all_inputs]
	all_inputs = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True)
	all_label = torch.tensor(all_label, dtype=torch.long)
	# print(all_inputs.shape, all_label.shape)
	logger.info(f"{all_inputs.shape}, {all_label.shape}")
	Dataset = TensorDataset(all_inputs, all_label)
	return Dataset


def read_data_for_deduction_simple(args, logger, mode):
	all_inputs = []
	all_label = []
	guid = 0
	for dataset in args.dataset:
		file_path = os.path.join('data', dataset, f'{mode}.json')
		with open(file_path, 'r') as f:
			games = json.load(f)
		for game in games:
			player2start_role = {}
			player_strategy_dist = {}
			dialogues = game.pop("Dialogue")
			# logger.info(f'{game}')
			for player, start_role in zip(game['playerNames'], game['startRoles']):
				player2start_role[player] = role2id[start_role]
				player_strategy_dist[player] = [0, 0, 0, 0, 0, 0, 0]
			for record in dialogues:
				if record['speaker'] not in player2start_role:
					if record['speaker'] not in (
					"Automated", "Game Audio", "Audio", "Siri Voice", "Siri", "Timer", "Twitch Alert",
					"Voiceover",) and not record['speaker'].startswith("Speaker "):
						logger.warning(f"non-exist speaker {record['speaker']}, ignored")
					continue
				for strategy in record['annotation']:
					player_strategy_dist[record['speaker']][strategies2id[strategy]] += 1
			feature = []
			for i, player in enumerate(game['playerNames']):
				tot = np.sum(player_strategy_dist[player])
				strategy_dist = [x / tot for x in player_strategy_dist[player]]
				start_role_encoding = [0] * len(role2id)
				start_role_encoding[player2start_role[player]] = 1
				feature += strategy_dist + start_role_encoding
			feature += [0] * ((6 - len(game['playerNames'])) * (7 + len(role2id)))
			assert len(feature) == 6 * (7 + len(role2id)), f'{len(feature)}'
			all_inputs.append(feature)
			label = [6 if x in ('NA', 'N/A') else x for x in game['votingOutcome']]
			label += [-100] * (6 - len(game['playerNames']))
			all_label.append(label)
			guid += 1
			if guid % 100 == 1:
				logger.info("*** Example ***")
				logger.info(f"guid: {guid}")
				logger.info(f"feature: {feature}")
				logger.info(f"label: {label}")

	logger.info(f"total datapoints: {guid}")
	all_inputs = torch.tensor(all_inputs, dtype=torch.float32)
	all_label = torch.tensor(all_label, dtype=torch.long)
	logger.info(f"{all_inputs.shape}, {all_label.shape}")
	Dataset = TensorDataset(all_inputs, all_label)
	return Dataset


def read_data_for_deduction_simple_paired(args, logger, mode, role_embed=False, organize_in_dataset=True):
	all_inputs = []
	all_label = []
	guid = 0
	for dataset in args.dataset:
		file_path = os.path.join('data', dataset, f'{mode}.json')
		with open(file_path, 'r') as f:
			games = json.load(f)
		for game in games:
			player2start_role = {}
			player_strategy_dist = {}
			player_feature = {}
			dialogues = game.pop("Dialogue")
			# logger.info(f'{game}')
			for player, start_role in zip(game['playerNames'], game['startRoles']):
			# for player, start_role in zip(game['playerNames'], game['endRoles']):
			# 	if start_role not in role2id:
			# 		start_role = 'Moderator'
				player2start_role[player] = role2id[start_role]
				player_strategy_dist[player] = [0, 0, 0, 0, 0, 0, 0]

			for record in dialogues:
				if record['speaker'] not in player2start_role:
					if record['speaker'] not in ("Automated", "Game Audio", "Audio", "Siri Voice", "Siri", "Timer", "Twitch Alert", "Voiceover",) \
							and not record['speaker'].startswith("Speaker "):
						logger.warning(f"non-exist speaker {record['speaker']}, ignored")
					continue
				for strategy in record['annotation']:
					player_strategy_dist[record['speaker']][strategies2id[strategy]] += 1

			for i, player in enumerate(game['playerNames']):
				tot = np.sum(player_strategy_dist[player])
				if tot == 0:
					logger.warning(f"Said nothing!!! {player}, {player_strategy_dist}")
					strategy_dist = player_strategy_dist[player].copy()
				else:
					strategy_dist = [x / tot for x in player_strategy_dist[player]]
				if np.isnan(strategy_dist).any():
					logger.warning(f"nan!!! {player}, {player_strategy_dist}")

				if role_embed:
					start_role_encoding = [0] * len(role2id)
					start_role_encoding[player2start_role[player]] = 1
					player_feature[player] = strategy_dist + start_role_encoding
				else:
					player_feature[player] = strategy_dist

			for i, voter in enumerate(game['playerNames']):
				if player2start_role[voter] == 0:
					continue
				for j, voted in enumerate(game['playerNames']):
					if player2start_role[voted] == 0:
						continue
					guid += 1
					label = 1 if game['votingOutcome'][i] == j else 0
					feature = player_feature[voter] + player_feature[voted][:7]  # we only use strategy embedding for the voter
					assert len(feature) == 14 if not role_embed else 40

					assert not np.isnan(feature).any()
					all_inputs.append(feature)
					all_label.append(label)
					if guid % 1000 == 1:
						logger.info("*** Example ***")
						logger.info(f"guid: {guid}")
						logger.info(f"feature: {feature}")
						logger.info(f"label: {label}")

	logger.info(f"total datapoints: {guid}")
	all_inputs = torch.tensor(all_inputs, dtype=torch.float32)
	all_label = torch.tensor(all_label, dtype=torch.long)
	logger.info(f"{all_inputs.shape}, {all_label.shape}")
	if organize_in_dataset is True:
		Dataset = TensorDataset(all_inputs, all_label)
		return Dataset
	else:
		return all_inputs, all_label
