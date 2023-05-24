import copy

from transformers.models.bert.modeling_bert import *
from transformers.models.roberta.modeling_roberta import *
from torch import nn


class BertForSequenceClassificationWithVideo(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # --------------------------------------------------------------
        # map context and video feature into 3x768
        # self.compress = nn.Linear(config.hidden_size * 18, config.hidden_size * 3)  # compress video feature from 18 to 3
        # self.fuser = nn.Linear(config.hidden_size * 4, config.hidden_size)

        # map context to 3x768 and concatenate it with video feature
        # self.compress = nn.Linear(config.hidden_size * 15, config.hidden_size)  # compress video feature from 18 to 3
        # self.fuser = nn.Linear(config.hidden_size * 5, config.hidden_size)  # use three crops + context

        # flatten all context or video features
        # self.fuser = nn.Linear(config.hidden_size * 2, config.hidden_size)  # only use center crop
        self.fuser = nn.Linear(config.hidden_size * 4, config.hidden_size)  # use three crops
        # self.fuser = nn.Linear(config.hidden_size * 19, config.hidden_size)  # use context and video crops

        # use self-attention layer
        # self.modal_embed = nn.Parameter(torch.zeros(1, 4, 768))
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # self.att1 = MultiScaleBlock(
        #     dim=768,
        #     dim_out=768,
        #     num_heads=8,
        #     mlp_ratio=4.0,
        #     qkv_bias=True,
        #     drop_rate=0.0,
        #     drop_path=0.1,
        #     norm_layer=norm_layer,
        #     kernel_q=[1, 1, 1],
        #     kernel_kv=[1, 1, 1],
        #     stride_q=[1, 1, 1],
        #     stride_kv=[1, 1, 1],
        #     mode="conv",
        #     has_cls_embed=False,
        #     pool_first=False,
        # )
        # self.att2 = copy.deepcopy(self.att1)
        # self.att3 = copy.deepcopy(self.att1)
        # self.norm = norm_layer(768)
        # --------------------------------------------------------------

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        # No video context
        video_features = video_features.view(video_features.shape[0], -1)  # flatten features of the three crops
        x = torch.cat((pooled_output, video_features), 1)

        # map context and video feature into 3x768
        # video_features = video_features.view(video_features.shape[0], -1)  # flatten features of the three crops
        # video_features = self.dropout(video_features)
        # video_features = self.compress(video_features)
        # x = torch.cat((pooled_output, video_features), 1)
        # x = torch.tanh(x)

        # map context to 3x768 and concatenate it with video feature
        # video_context = video_features[:, :15, :]
        # video_features = video_features[:, 15:, :]
        # video_context = video_context.view(video_context.shape[0], -1)  # flatten context of the three crops
        # video_features = video_features.view(video_features.shape[0], -1)  # flatten features of the three crops
        # video_context = self.compress(video_context)
        # x = torch.cat((pooled_output, video_features, video_context), 1)


        # use fuser layer
        x = self.dropout(x)
        x = self.fuser(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)


        # use self-attention layer
        # features = torch.cat([pooled_output.unsqueeze(1), video_features], dim=1)
        # features += self.modal_embed
        # features, _ = self.att1(features, thw_shape=None)
        # features, _ = self.att2(features, thw_shape=None)
        # features, _ = self.att3(features, thw_shape=None)
        # features = self.norm(features)
        # features = features[:, 0, :]
        # logits = self.classifier(features)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForSequenceClassificationWithVideo(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # --------------------------------------------------------------
        # map context and video feature into 3x768
        # self.compress = nn.Linear(config.hidden_size * 18, config.hidden_size * 3)  # compress video feature from 18 to 3
        # self.fuser = nn.Linear(config.hidden_size * 4, config.hidden_size)

        # map context to 3x768 and concatenate it with video feature
        # self.compress = nn.Linear(config.hidden_size * 15, config.hidden_size)  # compress video feature from 18 to 3
        # self.fuser = nn.Linear(config.hidden_size * 5, config.hidden_size)  # use three crops + context

        # flatten context or video features
        # self.fuser = nn.Linear(config.hidden_size * 2, config.hidden_size)  # only use center crop
        self.fuser = nn.Linear(config.hidden_size * 4, config.hidden_size)  # use three crops
        # self.fuser = nn.Linear(config.hidden_size * 19, config.hidden_size)  # use context and video crops

        # use self-attention layer
        # self.modal_embed = nn.Parameter(torch.zeros(1, 4, 768))
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # self.att1 = MultiScaleBlock(
        #     dim=768,
        #     dim_out=768,
        #     num_heads=8,
        #     mlp_ratio=4.0,
        #     qkv_bias=True,
        #     drop_rate=0.0,
        #     drop_path=0.1,
        #     norm_layer=norm_layer,
        #     kernel_q=[1, 1, 1],
        #     kernel_kv=[1, 1, 1],
        #     stride_q=[1, 1, 1],
        #     stride_kv=[1, 1, 1],
        #     mode="conv",
        #     has_cls_embed=False,
        #     pool_first=False,
        # )
        # self.att2 = copy.deepcopy(self.att1)
        # self.att3 = copy.deepcopy(self.att1)
        # self.norm = norm_layer(768)
        # --------------------------------------------------------------

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        video_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        # No video context
        video_features = video_features.view(video_features.shape[0], -1)  # flatten features of the three crops
        x = torch.cat((pooled_output, video_features), 1)

        # map context and video feature into 3x768
        # video_features = video_features.view(video_features.shape[0], -1)  # flatten features of the three crops
        # video_features = self.dropout(video_features)
        # video_features = self.compress(video_features)
        # x = torch.cat((pooled_output, video_features), 1)
        # x = torch.tanh(x)

        # map context to 3x768 and concatenate it with video feature
        # video_context = video_features[:, :15, :]
        # video_features = video_features[:, 15:, :]
        # video_context = video_context.view(video_context.shape[0], -1)  # flatten context of the three crops
        # video_features = video_features.view(video_features.shape[0], -1)  # flatten features of the three crops
        # video_context = self.compress(video_context)
        # x = torch.cat((pooled_output, video_features, video_context), 1)

        # use fuser layer
        x = self.dropout(x)
        x = self.fuser(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        # use self-attention layer
        # features = torch.cat([pooled_output.unsqueeze(1), video_features], dim=1)
        # features += self.modal_embed
        # features, _ = self.att1(features, thw_shape=None)
        # features, _ = self.att2(features, thw_shape=None)
        # features, _ = self.att3(features, thw_shape=None)
        # features = self.norm(features)
        # # features = features[:, 0, :]
        # features = features.mean(dim=1)
        # logits = self.classifier(features)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LSTMPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_labels):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs, labels):
        output, (hn, cn) = self.lstm(inputs)
        # print(inputs.shape, output.shape, hn.shape)
        x = hn.view(-1, self.hidden_dim)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        # print(logits)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

class Deduction_simple(nn.Module):

    def __init__(self, input_dim, num_labels):
        super(Deduction_simple, self).__init__()
        self.num_labels = num_labels
        self.fc1 = nn.Linear(input_dim, num_labels * (num_labels - 1))

    def forward(self, inputs, labels):
        logits = self.fc1(inputs)
        logits = logits.view(-1, self.num_labels, self.num_labels - 1)
        # print(logits)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels.view(-1, self.num_labels - 1))
        return loss, logits

class Deduction_simple_paired(nn.Module):

    def __init__(self, input_dim, num_labels):
        super(Deduction_simple_paired, self).__init__()
        self.num_labels = num_labels
        self.fc1 = nn.Linear(input_dim, num_labels)

    def forward(self, inputs, labels):
        logits = self.fc1(inputs)
        logits = logits.view(-1, self.num_labels)
        # print(logits)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels.view(-1))
        return loss, logits