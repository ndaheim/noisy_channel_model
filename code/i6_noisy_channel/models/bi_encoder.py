from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel, AutoModel, RobertaPreTrainedModel, RobertaModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class BiEncoderOutput(ModelOutput):
  """
    Base class for outputs of sentence classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`):
        distance (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`):
    """

  loss: Optional[torch.FloatTensor] = None
  last_hidden_state: torch.FloatTensor = None
  margin: torch.FloatTensor = None


class RobertaForBiEncoder(RobertaPreTrainedModel):

  def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.roberta = RobertaModel(config, add_pooling_layer=False)
    self.init_weights()

  def forward(
      self,
      input_ids,
      attention_mask=None,
      positive_document_input_ids=None,
      negative_document_input_ids=None,
      labels=None,
      return_dict=None,
  ):
    batch_size = input_ids.shape[0]

    query_output: BaseModelOutput = self.roberta(input_ids, attention_mask=attention_mask)
    query_embedding = query_output.last_hidden_state[:, 0]

    if positive_document_input_ids is not None:
      positive_document_output: BaseModelOutput = self.roberta(positive_document_input_ids)
      negative_document_output: BaseModelOutput = self.roberta(negative_document_input_ids)

      positive_document_embedding = positive_document_output.last_hidden_state[:, 0]
      negative_document_embedding = negative_document_output.last_hidden_state[:, 0]

      loss_type = 'triplet'

      if loss_type == 'triplet':
        positive_distance = torch.norm(query_embedding - positive_document_embedding, p=2.0, dim=-1)
        negative_distance = torch.norm(query_embedding - negative_document_embedding, p=2.0, dim=-1)
        margin = positive_distance - negative_distance
        loss_func = torch.nn.TripletMarginLoss(margin=self.config.triplet_loss_margin, p=2.0)
        loss = loss_func(query_embedding, positive_document_embedding, negative_document_embedding)
      else:
        assert False
    else:
      loss = torch.tensor(0.0)
      margin = torch.tensor([0.0] * batch_size)

    if labels is None:
      loss = None

    if not return_dict:
      if loss is None:
        return query_embedding, margin
      return loss, query_embedding, margin

    return BiEncoderOutput(loss, query_embedding, margin)


class BiEncoderModel(PreTrainedModel):

  def __init__(self, config, query_model=None, document_model=None):
    super(BiEncoderModel, self).__init__(config)

    if config.encoder_shared:
      if query_model is not None:
        self.model = query_model
      else:
        self.model = AutoModel.from_config(config)
        self.model.resize_token_embeddings(config.vocab_size)
    else:
      raise NotImplementedError()

    self.config = config

    if config.model_parallel_encoders:
      assert torch.cuda.device_count() == 2
      self.is_parallelizable = True
      self.model_parallel = True

      self.model.to(torch.device('cuda:0'))
      self.model.to(torch.device('cuda:1'))

  def forward(
      self,
      input_ids,
      attention_mask,
      positive_document_input_ids=None,
      negative_document_input_ids=None,
      labels=None,
  ):
    batch_size = input_ids.shape[0]

    if self.config.model_parallel_encoders:
      input_ids = input_ids.to(self.model.device)
      attention_mask = attention_mask.to(self.model.device)
    query_output: BaseModelOutput = self.model(input_ids, attention_mask=attention_mask)
    query_embedding = query_output.last_hidden_state[:, 0]

    if positive_document_input_ids is not None:
      if self.config.model_parallel_encoders:
        positive_document_input_ids = positive_document_input_ids.to(self.model.device)
        negative_document_input_ids = negative_document_input_ids.to(self.model.device)

      positive_document_output: BaseModelOutput = self.model(positive_document_input_ids)
      negative_document_output: BaseModelOutput = self.model(negative_document_input_ids)

      positive_document_embedding = positive_document_output.last_hidden_state[:, 0]
      negative_document_embedding = negative_document_output.last_hidden_state[:, 0]

      if self.config.model_parallel_encoders:
        positive_document_embedding = positive_document_embedding.to(self.model.device)
        negative_document_embedding = negative_document_embedding.to(self.model.device)

      loss_type = 'triplet'

      if loss_type == 'triplet':
        positive_distance = torch.norm(query_embedding - positive_document_embedding, p=2.0, dim=-1)
        negative_distance = torch.norm(query_embedding - negative_document_embedding, p=2.0, dim=-1)
        margin = positive_distance - negative_distance
        loss_func = torch.nn.TripletMarginLoss(margin=self.config.triplet_loss_margin, p=2.0)
        loss = loss_func(query_embedding, positive_document_embedding, negative_document_embedding)
      else:
        assert False
    else:
      loss = torch.tensor(0.0)
      margin = torch.tensor([0.0] * batch_size)

    return BiEncoderOutput(loss, query_embedding, margin)
