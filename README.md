# Werewolf Among Us: Multimodal Resources for Modeling Persuasion Behaviors in Social Deduction Games

### Findings of ACL 2023

### [Project Page](https://bolinlai.github.io/projects/Werewolf-Among-Us/) | [Paper](https://aclanthology.org/2023.findings-acl.411.pdf) | [Dataset](https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us)

#### [Bolin Lai*](https://bolinlai.github.io/), [Hongxin Zhang*](https://icefoxzhx.github.io/), [Miao Liu*](https://aptx4869lm.github.io/), [Aryan Pariani*](https://scholar.google.com/citations?hl=en&user=EnC_6s0AAAAJ), [Fiona Ryan](https://fkryan.github.io/), [Wenqi Jia](https://vjwq.github.io/), [Shirley Anugrah Hayati](https://www.shirley.id/), [James M. Rehg](https://rehg.org/), [Diyi Yang](https://cs.stanford.edu/~diyiy/)

**The old website for our dataset is down. Please find more details on our new [project page](https://bolinlai.github.io/projects/Werewolf-Among-Us/) and [HuggingFace](https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us).**

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
