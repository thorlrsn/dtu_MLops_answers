# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import torch

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    # TODO 
    # Include test data as well??

    datas = []
    for i in range(5):
        datas.append(np.load((input_filepath + "train_" + str(i) + ".npz"), allow_pickle=True))
    imgs = torch.tensor(np.concatenate([c['images'] for c in datas])).reshape(-1, 1, 28, 28)
    labels = torch.tensor(np.concatenate([c['labels'] for c in datas]))

    # normalise data
    imgs = norm(imgs)
    torch.save(imgs,(output_filepath+"images_tensor.pt"))

def norm(tensor):
     # Step 2: creating a torch tensor
    t = tensor
    # Step 3: Computing the mean, std and variance
    mean, std, var = torch.mean(t), torch.std(t), torch.var(t)
    # Step 4: Normalizing the tensor
    t  = (t-mean)/std
    # Step 5: Again compute the mean, std and variance
    # after Normalize
    mean, std, var = torch.mean(t), torch.std(t), torch.var(t)
    return t

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # train_path = r"C:\Users\Thor\Documents\dtu_MLops_answers\S1\final_exercise\data\data_corrupt\corruptmnist/"
    # out_path = r"C:\Users\Thor\Documents\dtu_MLops_answers\S1\final_exercise\data\data_corrupt\corruptmnist/"
    main()
