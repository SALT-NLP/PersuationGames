# Werewolf Among Us: Multimodal Resources for Modeling Persuasion Behaviors in Social Deduction Games

### Findings of ACL 2023

**The old website for our dataet may be down. You can directly download the dataset via this [link](https://drive.google.com/drive/folders/1N4PymMbKXFzqy3fq4ZGjdrc0oiScV914). Please read *README* in this link for dowload guidance.** 

This repo contains codes for the following paper:

*[Bolin Lai](https://bolinlai.github.io/), Hongxin Zhang, Miao Liu, Aryan Pariani, Fiona Ryan, Wenqi Jia, Shirley Anugrah Hayati, James M. Rehg, Diyi Yang*: Werewolf Among Us: Multimodal Resources for Modeling Persuasion Behaviors in Social Deduction Games

If you find our work is helpful to your research, please use the bibtex below to cite the paper.


```
@inproceedings{lai2023werewolf,
  title={Werewolf Among Us: Multimodal Resources for Modeling Persuasion Behaviors in Social Deduction Games},
  author={Lai, Bolin and Zhang, Hongxin and Liu, Miao and Pariani, Aryan and Ryan, Fiona and Jia, Wenqi and Hayati, Shirley Anugrah and Rehg, James and Yang, Diyi},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={6570--6588},
  year={2023}
}
```


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
