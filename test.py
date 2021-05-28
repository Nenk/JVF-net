import argparse
import yaml
# from trainer import Solver
from pytorch_metric_learning import losses, miners, distances, reducers, samplers, trainers, testers

class validate_for_VF_triplet():
    def __init__(self, aud_stream, vis_stream, accuracy_calculator, batch_size):
        self.aud_stream_model = aud_stream
        self.vis_stream_model = vis_stream
        self.accuracy_calculator = accuracy_calculator
        self.batch_size = batch_size

    def get_accuracy(self,  face_set, voice_set):
        train_embeddings, train_labels = self.get_all_embeddings(face_set, self.vis_stream_model)
        test_embeddings, test_labels = self.get_all_embeddings(voice_set, self.aud_stream_model)
        accuracies = self.accuracy_calculator.get_accuracy(test_embeddings,
                                                      train_embeddings,
                                                      test_labels,
                                                      train_labels,
                                                      False)
        return accuracies


    def get_all_embeddings(self, dataset, model):
        tester = testers.BaseTester(batch_size=self.batch_size,
                                    dataloader_num_workers=8,)
        return tester.get_all_embeddings(dataset, model)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='./option/baseline.yaml')

    args = parser.parse_args()

    with open(args.config_path, 'r') as config:
        config = yaml.load(config.read())
    solver = Solver(config)
    solver.test()

if __name__ == '__main__':
    main()
