import argparse
import yaml
from trainer import trainer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='./option/SVHF.yaml')

    args = parser.parse_args()

    with open(args.config_path, 'r') as config:
        config = yaml.load(config.read())
    Trainer = trainer(config)
    Trainer.train_()

if __name__ == '__main__':
    main()
