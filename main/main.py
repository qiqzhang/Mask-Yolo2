import sys
sys.path.append("..")
from models.network import *
from utils.utils import get_args
from utils.config import process_config
from utils.dirs import create_dirs
from data_loader.data_generator import *
from trainer.trainer import Trainer
from utils.logger import Logger

def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir,config.checkpoint_dir])



    dataset = DataGenerator(config)

    sess = tf.Session()



    logger = Logger(sess, config)
    net = YOLO2(config)
    trainer = Trainer(sess,net,dataset,config,logger)

    trainer.train()

if __name__ == "__main__":
    main()
