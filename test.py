import argparse
import yaml
import torch
import torchvision
# from trainer import trainer
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning import losses, miners, distances, reducers, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions as c_f


class validate_for_VF_triplet():
    def __init__(self, aud_stream, vis_stream, accuracy_calculator, batch_size):
        self.aud_stream_model = aud_stream
        self.vis_stream_model = vis_stream
        self.accuracy_calculator = accuracy_calculator
        self.batch_size = batch_size

    def get_accuracy(self, face_set, voice_set):
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



mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

def imshow_face_figure(img, figsize=(8, 4)):
    img = inv_normalize(img)
    npimg = img.numpy()

    # fig, axes = plt.subplots(4, 3, figsize=(8, 4), tight_layout=True)
    # for row in axes:
    #     for col in row:
    #         col.plot(x, y)

    plt.figure(figsize = figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.savefig('./utils/Face_emotions.png', dpi = 600)
    plt.show()


class inference():
    def __init__(self, aud_stream, vis_stream, distance):
        # Create the dataset and load the trained model
        self.aud_stream_model = aud_stream
        self.vis_stream_model = vis_stream

        # Create the InferenceModel wrapper
        match_finder = MatchFinder(distance=distance, threshold=0.7)

        self.vis_inference_model = InferenceModel(self.vis_stream_model, match_finder=match_finder)
        self.aud_inference_model = InferenceModel(self.aud_stream_model, match_finder=match_finder)
        self.cross_inference_model = InferenceModel(self.aud_stream_model, self.vis_stream_model, match_finder=match_finder)
        i=1

    def get_nearerst_images(self, face_set, voice_set):
        # Get label from the datasets
        face_set_labels_to_indices = c_f.get_labels_to_indices(face_set.label[0:1000])
        voice_set_labels_to_indices = c_f.get_labels_to_indices(voice_set.label[0:1000])

        # Get nearest neighbors of a query
        # create faiss index
        train_face_vectors = torch.stack([face_set[i][0] for i in range(1000)], dim=0)
        self.vis_inference_model.train_indexer(train_face_vectors.to('cuda'))

        train_voice_vectors = torch.stack([torch.from_numpy(voice_set[i][0] ) for i in range(1000)], dim=0)  # 转为tensor tensor = torch.from_numpy(ndarray)
        self.aud_inference_model.train_indexer(train_voice_vectors.to('cuda'))

        # self.cross_inference_model.train_indexer(train_voice_vectors.to('cuda'))

        # emotions: (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        for index in sorted(face_set_labels_to_indices):
            emotion_class = face_set_labels_to_indices[index]
            # get 10 nearest neighbors for a neutral image
            for img_type in [emotion_class]:
                img = face_set[img_type[0]][0].unsqueeze(0)
                print("query image")
                imshow_face_figure(torchvision.utils.make_grid(img, nrow=5 ))
                indices, distances = self.vis_inference_model.get_nearest_neighbors(img.to('cuda'), k=10)
                nearest_imgs = [face_set[i][0] for i in indices[0]]
                print("nearest images")
                imshow_face_figure(torchvision.utils.make_grid(nearest_imgs, nrow=5))
                torchvision.utils.save_image(nearest_imgs,'./utils/Face_emotions_{}.png'.format(index),
                                             nrow=5, normalize=True, range=(-1,1), scale_each=False)

        # get 10 nearest neighbors for a neutral voice

    def get_pairs_images_from_voice(self, face_set, voice_set):
        # Get label from the datasets
        face_set_labels_to_indices = c_f.get_labels_to_indices(face_set.label[0:1000])
        voice_set_labels_to_indices = c_f.get_labels_to_indices(voice_set.label[0:1000])

        # Get nearest neighbors of a query
        # create faiss index
        train_face_vectors = torch.stack([face_set[i][0] for i in range(1000)], dim=0)
        self.vis_inference_model.train_indexer(train_face_vectors.to('cuda'))

        train_voice_vectors = torch.stack([torch.from_numpy(voice_set[i][0]) for i in range(1000)], dim=0)  # 转为tensor tensor = torch.from_numpy(ndarray)
        self.aud_inference_model.train_indexer(train_voice_vectors.to('cuda'))

        # emotions: (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        for index in sorted(voice_set_labels_to_indices):
            emotion_class = voice_set_labels_to_indices[index]
            # get 10 nearest neighbors for a neutral image
            for voice_type in [emotion_class]:
                voice = torch.from_numpy(voice_set[voice_type[0]][0]).unsqueeze(0)
                print("query voice {}".format(voice_set[voice_type[0]][1]))
                voice_query_emb = self.aud_inference_model.get_embeddings(voice.to('cuda'))
                indices, distances = self.vis_inference_model.indexer.search_nn(voice_query_emb.cpu().numpy(), k=10)
                nearest_imgs = [face_set[i][0] for i in indices[0]]
                nearest_imgs_labels = [face_set[i][1] for i in indices[0]]
                print("nearest images {}".format(nearest_imgs_labels))
                imshow_face_figure(torchvision.utils.make_grid(nearest_imgs, nrow=5))
                torchvision.utils.save_image(nearest_imgs, './utils/Face_voice_emotions_{}.png'.format(index),
                                             nrow=5, normalize=True, range=(-1, 1), scale_each=False)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config_path', type=str, default='./option/baseline.yaml')
#     args = parser.parse_args()
#
#     with open(args.config_path, 'r') as config:
#         config = yaml.load(config.read())
#     solver = trainer.trainer(config)
#     solver.test()
#
# if __name__ == '__main__':
#     main()
