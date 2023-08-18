import os
import os.path
import numpy as np
import random
import pickle
import json
import math

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt

import h5py
import utils


class FewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5, # number of novel categories.
                 nKbase=-1, # number of base categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=15*5, # number of test examples for all the novel categories.
                 nTestBase=15*5, # number of test examples for all the base categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000, # number of batches per epoch.
                 phase = 'train',
                 ):

        self.dataset = dataset
        self.nKnovel = nKnovel
        self.nKbase = nKbase
        self.phase = phase

        self.category = dataset.keys()
        self.ncategory = len(self.category)

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).
        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.
        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.category)
        assert(len(self.dataset[cat_id][0]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        idx = random.sample(list(range(len(self.dataset[cat_id][0]))), sample_size)
        data = []
        for i in idx:
            data += [(self.dataset[cat_id][0][i], self.dataset[cat_id][1][i], self.dataset[cat_id][2][i])]
        return data#random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.
        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.
        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set=='base':
            labelIds = self.category
        elif cat_set=='novel':
            labelIds = self.category
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.
        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories
        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert(nKnovel <= self.ncategory)
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        """
        Sample `nTestBase` number of images from the `Kbase` categories.
        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nTestBase: the total number of images that will be sampled.
        Returns:
            Tbase: a list of length `nTestBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nTestBase`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                #Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]
                Tbase += [img_id for img_id in imd_ids]

        assert(len(Tbase) == nTestBase)

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):
        """Samples train and test examples of the novel categories.
        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestNovel: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.
        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))
            # print(222,nEvalExamplesPerClass, nExemplars) 6, 15
            # print(333,len(imd_ids), np.shape(imd_ids[0][0])) 21, (seq_len, 76)
            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]
            Tnovel += [imds_tnovel]
            Exemplars += [imds_ememplars]

            # Tnovel += [(img_id, nKbase+Knovel_idx) for img_id in imds_tnovel]
            # Exemplars += [(img_id, nKbase+Knovel_idx) for img_id in imds_ememplars]
            # Tnovel += [[img_id] for img_id in imds_tnovel]
            # Exemplars += [[img_id] for img_id in imds_ememplars]
        # assert(len(Tnovel) == nTestNovel)
        # assert(len(Exemplars) == len(Knovel) * nExemplars)
        # print(444, len(Exemplars)) 5
        # print(np.shape(Exemplars)) (5, 15, 2)

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.
        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).
        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = []
        labels = []
        label_true = []
        for cat in examples:
            images.append([img for img, _, _ in cat])
            labels.append([lab for _, lab, _ in cat])
            label_true.append([lab2 for _, _, lab2 in cat])
        # print(666,len(images), np.shape(images[0])) 30 (2627, 76)
        #images = torch.stack(torch.from_numpy(images))
        #labels = torch.LongTensor(examples[:][:][1])
        return images, labels, label_true

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        def load_function(iter_idx):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            # print(444,np.shape(Test)) (5, 6, 2)
            Xt, Yt, Ytt = self.createExamplesTensorData(Test)
            Kall = torch.LongTensor(Kall)
            # print(555,np.shape(Exemplars)) (5, 15, 2)
            if len(Exemplars) > 0:
                Xe, Ye, Yet = self.createExamplesTensorData(Exemplars)
                # print(111,np.shape(Xe)) #(5, 15, x, 76)
                # print(222,np.shape(Ye))
                # print(333,np.shape(Xt))
                # print(444,np.shape(Yt))
                # print(555,np.shape(Kall))
                # print(666,np.shape(nKbase))
                return Xe, Ye, Yet, Xt, Yt, Ytt, Kall, nKbase
            else:
                return Xt, Yt, Ytt, Kall, nKbase

        # tnt_dataset = tnt.dataset.ListDataset(
        #     elem_list=range(self.epoch_size), load=load_function)
        # # data_loader = torch.utils.data.DataLoader(tnt_dataset, num_workers=(0 if self.is_eval_mode else self.num_workers))
        # data_loader = tnt_dataset.parallel(
        #     #batch_size=self.batch_size,
        #     num_workers=(0 if self.is_eval_mode else self.num_workers),
        #     shuffle=(False if self.is_eval_mode else True))
        # return data_loader
        data = load_function(epoch)
        # print(data)
        return data

    def __call__(self, epoch=0):
        data = self.get_iterator(epoch)

        return data

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
