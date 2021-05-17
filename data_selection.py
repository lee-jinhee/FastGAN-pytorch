import numpy as np
from tqdm import tqdm 
from copy import deepcopy
import os
import time
import random

import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances

from inception import InceptionV3

def get_activations(images, model, batch_size=50, dims=2048, device='cpu', verbose=True):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : List of image files
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    num_batches = len(images) // batch_size

    pred_arr = np.empty((len(images), dims))

    start_idx = 0

    for i in range(num_batches):
        start_time = time.time()

        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

        if verbose:
            print("\rINFO: Propagated batch %d/%d (%.4f sec/batch)" \
                % (i+1, num_batches, time.time()-start_time), end="", flush=True)

    return pred_arr

class DataSelector:
    def __init__(self, dataloader, num_data, full_size, name, device) -> None:
        self.pdist = torch.nn.PairwiseDistance(p=2)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx]).to(device)
        self.dataloader = dataloader
        self.features = None
        self.num_data = num_data
        self.full_size = full_size
        self.min_distances = None
        self.metric = 'euclidean'
        self.device = device
        self.dataset_name = False
        if name is not None:
            self.dataset_name = True
    
    def _register_train_dataset_feats(self, path='train_feats_itp_95_seed2.pkl'):
        # if not os.path.exists('stack_train_feats_itp_95_seed2.pkl'):
        #     if not os.path.exists(path):
        print('register_train_dataset_feats')
        data_iter = iter(self.dataloader)
        feats = None
        for data in tqdm(data_iter):
            if self.dataset_name:
                data = data[0]
            data_feats = get_activations(data, model=self.inception, device=self.device, 
                    verbose = False)
            if feats is None:
                feats = data_feats
            else:
                feats = np.concatenate([feats, data_feats], axis=0)
            #     data_feats = np.stack()
            # for i in range(len(idx)):
            #     feats[int(idx[i].item())] = data_feats[i]

        # pickle.dump(feats, open(path, "wb"))
            # else:
            #     train_feats = pickle.load(open(path, 'rb'))

        # train_feats = feats 
        # sorted_train_feats_keys = sorted(train_feats.keys())
        # for k in sorted_train_feats_keys:
        #     add_train_feats = torch.unsqueeze(train_feats[k], 0)
        #     if self.stack_train_feats is None:
        #         self.stack_train_feats = add_train_feats
        #     else:
        #         self.stack_train_feats = torch.cat((self.stack_train_feats, add_train_feats), 0)
        # pickle.dump(self.stack_train_feats, open('stack_train_feats_itp_95_seed2.pkl', 'wb'))
        # else:
        #     self.stack_train_feats = pickle.load(open('stack_train_feats_itp_95_seed2.pkl', 'rb'))
        self.features = feats
        print(feats)
        print('register_train_dataset_feats done')

    # def _get_next_index(self, selected):
    #     stack_train_feats = self.stack_train_feats.to(self.device)
    #     target = stack_train_feats[selected,:]
    #     # Use cdist
    #     diff = torch.cdist(target, stack_train_feats).cpu()
    #     return int(torch.argmax(torch.min(diff)))
    
    # def k_center_greedy(self):
    #     selected = [random.randrange(self.num_data)]
    #     while len(selected) < self.num_data:
    #         self._get_nex_index(selected)

    def update_distances(self, cluster_centers, reset_dist=False):
        """Update min distances given cluster centers.

        Args:
        cluster_centers: indices of cluster centers
        only_new: only calculate distance for newly selected points and update
            min_distances.
        rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

        if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1,1)
        else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def k_center_greedy_select(self, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.

        Args:
        model: model with scikit-like API with decision_function implemented
        already_selected: index of datapoints already selected

        Returns:
        indices of points selected to minimize distance to cluster centers
        """
        self._register_train_dataset_feats()
        print('Getting transformed features...')
        new_batch = []

        for _ in range(self.num_data):
            if len(new_batch) == 0:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.full_size))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.

            self.update_distances([ind], reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
                % max(self.min_distances))

        return new_batch
    
    def random_select(self):
        return list(random.sample(range(self.full_size),self.num_data))



# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""






