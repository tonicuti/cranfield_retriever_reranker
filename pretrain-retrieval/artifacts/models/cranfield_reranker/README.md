---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:5260
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['has anyone investigated the effect of shock generated vorticity on heat transfer to a blunt body .', 'vibrations of infinitely long cylindrical shells under initial stress . the general bending theory of shells under the influence of initial stress presented recently by herrmann and armenakas is applied in this investigation to study the effect of initial uniform circumferential stress, uniform bending moment and uniform radial shear on the dynamic response of an infinitely long cylindrical shell .'],
    ['references on the methods available for accurately estimating aerodynamic heat transfer to conical bodies for both laminar and turbulent flow .', 'formulae and approximations for aerodynamic heating rates in high speed flight . this note gives formulae and approximations suitable for making preliminary estimates of aerodynamic heating rates in high speed flight . the formulae are based on the /intermediate enthalpy/ approximation which has given good agreement with theoretical and experimental evidence . in the general flight case they could be used in conjunction with an analogue computer or a step by step method of integration to predict the variations of heat flow and skin temperature with time . in the restricted case of flight at constant altitude and mach number, simple analytical methods and results are given which include the effects of radiation and can be applied to /thick/ as well as /thin/ skins where h is the aerodynamic heat transfer factor, and g, d and k are the heat capacity, thickness and thermal conductivity of the skin . if 0.1 the skin is approximately /thin/, i.e. temperature gradients across its thickness may be neglected .'],
    ['theoretical studies of creep buckling .', 'note on creep buckling of columns . this paper is concerned with the solution of the creep buckling of columns .  instantaneous elastic and plastic deformations, as well as the transient and secondary creep, are considered .  formulae for the critical time at which a column fails are presented for integral values of the exponents appearing in the creep law .'],
    ['how can the analytical solution of the buckling strength of a uniform circular cylinder loaded in axial compression be refined so as to lower the buckling load .', 'heat transfer to flat plate in high temperature rarefied ultra high mach number flow . an investigation was conducted in a hypersonic shock tunnel to determine the local heat transfer rates for a sharp leading edge flat plate .  the free stream mach number range was 7.95 to 25.1 with stagnation temperatures of approximately 2550 and 6500 r .  for these temperature and mach number conditions, the strong interaction parameter, varied from 2.35 to 826 .  the corresponding knudsen numbers, based on the ratio of the free stream mean free path and the leading edge thickness, varied from 0.38 to 85.5 . for free stream mach numbers greater than 10, knudsen numbers of approximately unity, and perfect gas conditions, the calculated heat transfer coefficients were found to vary as as predicted by the noninsulated flat plate theory of li and nagamatsu .  for the case of, the leading edge slip phenomenon drastically reduced the local heat transfer coefficients as compared to the theoretical values predicted with no slip at the surface .  for the extreme case of and, the measured local heat transfer rate was an order of magnitude less than the analytical value .  both the knudsen number and the free stream mach number are important physical parameters that determine the extent of the slip flow region .'],
    ['what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .', 'principles of creep buckling weight strength analysis . the relation of the time dependent tangent modulus load  as conceived by shanley  to actual column capacity is clarified .  it may be interpreted as a limiting case of the conservative estimate .  the time dependent tangent modulus load is, therefore, an approximation to a conservative estimate .  the approximation, however, may be either conservative or nonconservative when applied to imperfect or real columns .  typical cases are discussed and experimental results for two alloys are cited .'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'has anyone investigated the effect of shock generated vorticity on heat transfer to a blunt body .',
    [
        'vibrations of infinitely long cylindrical shells under initial stress . the general bending theory of shells under the influence of initial stress presented recently by herrmann and armenakas is applied in this investigation to study the effect of initial uniform circumferential stress, uniform bending moment and uniform radial shear on the dynamic response of an infinitely long cylindrical shell .',
        'formulae and approximations for aerodynamic heating rates in high speed flight . this note gives formulae and approximations suitable for making preliminary estimates of aerodynamic heating rates in high speed flight . the formulae are based on the /intermediate enthalpy/ approximation which has given good agreement with theoretical and experimental evidence . in the general flight case they could be used in conjunction with an analogue computer or a step by step method of integration to predict the variations of heat flow and skin temperature with time . in the restricted case of flight at constant altitude and mach number, simple analytical methods and results are given which include the effects of radiation and can be applied to /thick/ as well as /thin/ skins where h is the aerodynamic heat transfer factor, and g, d and k are the heat capacity, thickness and thermal conductivity of the skin . if 0.1 the skin is approximately /thin/, i.e. temperature gradients across its thickness may be neglected .',
        'note on creep buckling of columns . this paper is concerned with the solution of the creep buckling of columns .  instantaneous elastic and plastic deformations, as well as the transient and secondary creep, are considered .  formulae for the critical time at which a column fails are presented for integral values of the exponents appearing in the creep law .',
        'heat transfer to flat plate in high temperature rarefied ultra high mach number flow . an investigation was conducted in a hypersonic shock tunnel to determine the local heat transfer rates for a sharp leading edge flat plate .  the free stream mach number range was 7.95 to 25.1 with stagnation temperatures of approximately 2550 and 6500 r .  for these temperature and mach number conditions, the strong interaction parameter, varied from 2.35 to 826 .  the corresponding knudsen numbers, based on the ratio of the free stream mean free path and the leading edge thickness, varied from 0.38 to 85.5 . for free stream mach numbers greater than 10, knudsen numbers of approximately unity, and perfect gas conditions, the calculated heat transfer coefficients were found to vary as as predicted by the noninsulated flat plate theory of li and nagamatsu .  for the case of, the leading edge slip phenomenon drastically reduced the local heat transfer coefficients as compared to the theoretical values predicted with no slip at the surface .  for the extreme case of and, the measured local heat transfer rate was an order of magnitude less than the analytical value .  both the knudsen number and the free stream mach number are important physical parameters that determine the extent of the slip flow region .',
        'principles of creep buckling weight strength analysis . the relation of the time dependent tangent modulus load  as conceived by shanley  to actual column capacity is clarified .  it may be interpreted as a limiting case of the conservative estimate .  the time dependent tangent modulus load is, therefore, an approximation to a conservative estimate .  the approximation, however, may be either conservative or nonconservative when applied to imperfect or real columns .  typical cases are discussed and experimental results for two alloys are cited .',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 5,260 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                         | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                             | float                                                          |
  | details | <ul><li>min: 39 characters</li><li>mean: 110.74 characters</li><li>max: 266 characters</li></ul> | <ul><li>min: 161 characters</li><li>mean: 963.76 characters</li><li>max: 1729 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.26</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>has anyone investigated the effect of shock generated vorticity on heat transfer to a blunt body .</code>                                              | <code>vibrations of infinitely long cylindrical shells under initial stress . the general bending theory of shells under the influence of initial stress presented recently by herrmann and armenakas is applied in this investigation to study the effect of initial uniform circumferential stress, uniform bending moment and uniform radial shear on the dynamic response of an infinitely long cylindrical shell .</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>0.0</code> |
  | <code>references on the methods available for accurately estimating aerodynamic heat transfer to conical bodies for both laminar and turbulent flow .</code> | <code>formulae and approximations for aerodynamic heating rates in high speed flight . this note gives formulae and approximations suitable for making preliminary estimates of aerodynamic heating rates in high speed flight . the formulae are based on the /intermediate enthalpy/ approximation which has given good agreement with theoretical and experimental evidence . in the general flight case they could be used in conjunction with an analogue computer or a step by step method of integration to predict the variations of heat flow and skin temperature with time . in the restricted case of flight at constant altitude and mach number, simple analytical methods and results are given which include the effects of radiation and can be applied to /thick/ as well as /thin/ skins where h is the aerodynamic heat transfer factor, and g, d and k are the heat capacity, thickness and thermal conductivity of the skin . if 0.1 the skin is approximately /thin/, i.e. temperature gradients across its thickness m...</code> | <code>1.0</code> |
  | <code>theoretical studies of creep buckling .</code>                                                                                                         | <code>note on creep buckling of columns . this paper is concerned with the solution of the creep buckling of columns .  instantaneous elastic and plastic deformations, as well as the transient and secondary creep, are considered .  formulae for the critical time at which a column fails are presented for integral values of the exponents appearing in the creep law .</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: None
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `enable_jit_checkpoint`: False
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `use_cpu`: False
- `seed`: 42
- `data_seed`: None
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: -1
- `ddp_backend`: None
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `auto_find_batch_size`: False
- `full_determinism`: False
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `use_cache`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.5198 | 500  | 0.3823        |


### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.2.3
- Transformers: 5.0.0
- PyTorch: 2.10.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.8.3
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->