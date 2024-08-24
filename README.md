# Werewolf Among Us: Multimodal Resources for Modeling Persuasion Behaviors in Social Deduction Games

This repo contains codes for the following paper

*Bolin Lai, Hongxin Zhang, Miao Liu, Aryan Pariani, Fiona Ryan, Wenqi Jia, Shirley Anugrah Hayati, James M. Rehg, Diyi Yang*: Werewolf Among Us: Multimodal Resources for Modeling Persuasion Behaviors in Social Deduction Games

```
@article{lai2022werewolf,
  title={Werewolf Among Us: A Multimodal Dataset for Modeling Persuasion Behaviors in Social Deduction Games},
  author={Lai, Bolin and Zhang, Hongxin and Liu, Miao and Pariani, Aryan and Ryan, Fiona and Jia, Wenqi and Hayati, Shirley Anugrah and Rehg, James M and Yang, Diyi},
  journal={arXiv preprint arXiv:2212.08279},
  year={2022}
}
```

If you would like to refer to it, please cite the paper mentioned above. 


## Repository Structure
- **baselines**: Code for baselines
- **data**: Prepare the data in this folder

## Usage
### Install dependency
```
conda env create -f env.yaml
conda activate PersuasionGames
```


### Run

#### Run multiple Experiments
We provide script `exp.sh` to run hyperparameter search for bert model.

Then you can use `utils.py` to gather the results and have the best performing hyper-parameters according to their dev results.

Then you can run `exp_context.sh` with the best hyperparameter to experiment on different context sizes.

#### Single Run
`CUDA_VISIBLE_DEVICES=0 python3 baselines/main.py --output_dir out`

Optional parameters:
- model_type (only _bert_ and _roberta_ available now, please be careful if you are adding other models)
- model_name
- batch_size
- learning_rate
- num_train_epochs
- seed
- avalon
  - controlling whether to evaluate on avalon set
- video
  - controlling whether to use video features

### Result
Results will be shown in the folder you assigned as output_dir (_out_ by default)

I've uploaded some results along with the training curves which can be visualized with 
`tensorboard --logdir out`
