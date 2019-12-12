 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from experiments.experiment import Experiment
from error_handling.console_logger import ConsoleLogger
from evaluation.alignment_stats import AlignmentStats
from evaluation.embedding_space_stats import EmbeddingSpaceStats
from evaluation.gradient_stats import GradientStats
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve

import json
import yaml
import torch
import numpy as np
import random
import pickle
import os
import librosa
from speech_utils.mu_law import MuLaw

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Experiments(object):

    def __init__(self, experiments):
        self._experiments = experiments

    @property
    def experiments(self):
        return self._experiments

    def train(self):
        for experiment in self._experiments:
            Experiments.set_deterministic_on(experiment.seed)
            experiment.train()
            torch.cuda.empty_cache()

    def save_embedding(self):
        for experiment in self._experiments:
            Experiments.set_deterministic_on(experiment.seed)
            embedding_list = experiment.save_embeddings()
            path = '../data/ibm/features/'
            with open('{0}/embedding.txt'.format(path), 'w') as f:
                f.write(str(embedding_list))
            # with open('{0}/embedding.pickle'.format(path), 'wb') as handle:
            #     pickle.dump(embedding_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            plt.figure(figsize=(15,10))
            plot = convolve(np.squeeze(embedding_list), Gaussian2DKernel(stddev=1))
            # plot = np.squeeze(embedding_list)
            plt.imshow(np.squeeze(plot.T), cmap=plt.cm.RdBu, interpolation='nearest')
            plt.savefig('{}/heatmap.png'.format(path), bbox_inches='tight')
            torch.cuda.empty_cache()

    def evaluate_once(self, evaluation_options, eval_folder, configuration, heatmap, produce_metrics):
        for experiment in self._experiments:
            Experiments.set_deterministic_on(experiment.seed)
            evaluate_dict = experiment.evaluate_once(eval_folder, configuration)
            # path = '/home/taresh/VQ-VAE-Speech/data/ibm/features/{}'.format(eval_folder)
            
            # # with open('{0}/{1}.txt'.format(path, eval_folder), 'w') as f:
            # #     f.write(str(evaluate_dict))
            # # print (evaluate_dict['EH_2401_fuel-tax_con_opening_JLchunk1.wav']['valid_reconstructions'].shape)
            # print (evaluate_dict.keys())
            # a = np.squeeze(evaluate_dict['EH_2401_fuel-tax_con_opening_JLchunk3.wav']['preprocessed_audio'].detach().cpu().numpy())
            # a2 = np.squeeze(evaluate_dict['EH_2401_fuel-tax_con_opening_JLchunk1.wav']['preprocessed_audio'].detach().cpu().numpy())
            # a3 = np.squeeze(evaluate_dict['EH_2401_fuel-tax_con_opening_JLchunk2.wav']['preprocessed_audio'].detach().cpu().numpy())

            # b = np.squeeze(evaluate_dict['EH_2401_fuel-tax_con_opening_JLchunk3.wav']['valid_reconstructions'].detach().cpu().numpy())
            # b2 = np.squeeze(evaluate_dict['EH_2401_fuel-tax_con_opening_JLchunk1.wav']['valid_reconstructions'].detach().cpu().numpy())
            # b3 = np.squeeze(evaluate_dict['EH_2401_fuel-tax_con_opening_JLchunk2.wav']['valid_reconstructions'].detach().cpu().numpy())

            # b = MuLaw.decode(np.argmax(b, axis=0))
            # b2 = MuLaw.decode(np.argmax(b2, axis=0))
            # b3 = MuLaw.decode(np.argmax(b3, axis=0))

            # # import sys
            # # sys.exit(0)

            # librosa.output.write_wav('val_targeta.wav', a, configuration['sampling_rate'])
            # librosa.output.write_wav('val_targeta2.wav', a2, configuration['sampling_rate'])
            # librosa.output.write_wav('val_targeta3.wav', a3, configuration['sampling_rate'])

            # librosa.output.write_wav('valb.wav', b, configuration['sampling_rate'])
            # librosa.output.write_wav('valb2.wav', b2, configuration['sampling_rate'])
            # librosa.output.write_wav('valb3.wav', b3, configuration['sampling_rate'])
            path = '../data/ibm/features/{}'.format(eval_folder)
            print (path)
            print(evaluate_dict.keys())

            with open('{}.pickle'.format("../data/ibm/train4"), 'wb') as handle:
                pickle.dump(evaluate_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # torch.cuda.empty_cache()
            
            # print("concatenate quantized {}".format(evaluate_dict['OAF_youth_happy.wav']['concatenated_quantized'].shape))
            # print("quantized {}".format(evaluate_dict['OAF_youth_happy.wav']['quantized'].shape))
            # print("encoding_indices {}".format(evaluate_dict['OAF_youth_happy.wav']['encoding_indices'].shape))
            # print("encodings {}".format(evaluate_dict['OAF_youth_happy.wav']['encodings'].shape))
            # print("distances {}".format(evaluate_dict['OAF_youth_happy.wav']['distances'].shape))
            with open('{0}/{1}.txt'.format(path, eval_folder + '1'), 'w') as f:
                f.write(str(evaluate_dict))
            if heatmap:
                keys = list(evaluate_dict.keys())
                first_plot = evaluate_dict[keys[0]]['quantized']
                second_plot = evaluate_dict[keys[1]]['quantized']
                

                gs = gridspec.GridSpec(1, 3)
                # gs.update(wspace=0.000005, hspace=0.00001)
                fig = plt.figure(figsize=(15,10))
                ax = fig.add_subplot(gs[0, 0]) # row 0, col 0
                first_plot = convolve(np.squeeze(first_plot.cpu().detach().numpy()), Gaussian2DKernel(stddev=1))
                ax.imshow(np.squeeze(first_plot), cmap=plt.cm.RdBu, interpolation='nearest')

                ax = fig.add_subplot(gs[0, 1]) # row 0, col 0
                second_plot = convolve(np.squeeze(second_plot.cpu().detach().numpy()), Gaussian2DKernel(stddev=1))
                ax.imshow(np.squeeze(second_plot), cmap=plt.cm.RdBu, interpolation='nearest')

                ax = fig.add_subplot(gs[0, 2])
                # convolved_map = convolve(np.squeeze(hm.cpu().detach().numpy()), Gaussian2DKernel(stddev=1))
                convolved_map = first_plot - second_plot
                im = ax.imshow(convolved_map, cmap=plt.cm.RdBu, interpolation='nearest')
            
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                # plt.subplots_adjust(wspace=0, hspace=0)
                fig.savefig('{}/heatmap.png'.format(path), bbox_inches='tight')
                with open('{0}/heatmap.pickle'.format(path, eval_folder), 'wb') as handle:
                    pickle.dump(convolved_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if produce_metrics:
                print ('doing nothing as of now')

            # if produce_metrics:
            #     keys = list(evaluate_dict.keys())
            #     print ('here')

            #     import sys
            #     sys.exit(0)

            #     wav_names = object_dict.keys()
            #     X = np.zeros((2800, 1536))
            #     # Y = np.empty([2800, 1], dtype=object)
            #     Y = np.zeros((2800, 1))
            #     names = {'happy.wav':1, 'sad.wav':2, 'angry.wav':3, 'disgust.wav':4, 'ps.wav':5, 'fear.wav':6, 'neutral.wav':7}

            #     # names = ['happy.wav', 'sad.wav', 'angry.wav', 'disgust.wav', 'ps.wav', 'fear.wav', 'neutral.wav']

            #     for i, wav in enumerate(wav_names):
            #         # print(object_dict[wav]['quantized'].squeeze().reshape(1, -1).shape)
            #         for name in names: 
            #             if name in wav.split('_'):
            #                 Y[i] = names[name]
            #                 # print (object_dict[wav]['quantized'].squeeze().flatten().shape)
            #                 X[i] = object_dict[wav]['quantized'].squeeze().flatten().cpu().detach().numpy()
            #                 break



            torch.cuda.empty_cache()

    def evaluate(self, evaluation_options):
        # TODO: put all types of evaluation in evaluation_options, and skip this loop if none of them are set to true
        for experiment in self._experiments:
            Experiments.set_deterministic_on(experiment.seed)
            experiment.evaluate(evaluation_options)
            torch.cuda.empty_cache()

        if type(self._experiments[0].seed) == list:
            Experiments.set_deterministic_on(self._experiments[0].seed[0]) # For now use only the first seed there
        else:
            Experiments.set_deterministic_on(self._experiments[0].seed)

        if evaluation_options['compute_quantized_embedding_spaces_animation']:
            EmbeddingSpaceStats.compute_quantized_embedding_spaces_animation(
                all_experiments_paths=[experiment.experiment_path for experiment in self._experiments],
                all_experiments_names=[experiment.name for experiment in self._experiments],
                all_results_paths=[experiment.results_path for experiment in self._experiments]
            )

        if evaluation_options['plot_clustering_metrics_evolution']:
            AlignmentStats.compute_clustering_metrics_evolution(
                all_experiments_names=[experiment.name for experiment in self._experiments],
                result_path=self._experiments[0].results_path
            )

        if evaluation_options['check_clustering_metrics_stability_over_seeds']:
            AlignmentStats.check_clustering_metrics_stability_over_seeds(
                all_experiments_names=[experiment.name for experiment in self._experiments],
                result_path=self._experiments[0].results_path
            )

        if evaluation_options['plot_gradient_stats']:
            all_experiments_paths = [experiment.experiment_path for experiment in self._experiments]
            all_experiments_names = [experiment.name for experiment in self._experiments]
            all_results_paths = [experiment.results_path for experiment in self._experiments]
            gradient_stats_entries = list()
            for i in range(len(all_experiments_paths)):
                experiment_path = all_experiments_paths[i]
                experiment_name = all_experiments_names[i]
                experiment_results_path = all_results_paths[i]
                # List all file names related to the gradient stats for the current observed experiment
                file_names = [file_name for file_name in os.listdir(experiment_path) if 'gradient-stats' in file_name and experiment_name in file_name]

                # Sort file names by epoch number and iteration number as well
                file_names = sorted(file_names, key=lambda x: 
                    (int(x.replace(experiment_name + '_', '').replace('_gradient-stats.pickle', '').split('_')[0]),
                    int(x.replace(experiment_name + '_', '').replace('_gradient-stats.pickle', '').split('_')[1]))
                )

                with tqdm(file_names) as bar:
                    bar.set_description('Processing')
                    for file_name in bar:
                        with open(experiment_path + os.sep + file_name, 'rb') as file:
                            split_file_name = file_name.replace(experiment_name + '_', '').replace('_gradients-stats.pickle', '').split('_')
                            gradient_stats_entries.append((int(split_file_name[0]), int(split_file_name[1]), pickle.load(file)))

                GradientStats.plot_gradient_flow_over_epochs(
                    gradient_stats_entries,
                    output_file_name=experiment_results_path + os.sep + experiment_name + '_gradient_flow.png'
                )

    @staticmethod
    def set_deterministic_on(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def load(experiments_path):
        experiments = list()
        with open(experiments_path, 'r') as experiments_file:
            experiment_configurations = json.load(experiments_file)

            configuration = None
            with open(experiment_configurations['configuration_path'], 'r') as configuration_file:
                configuration = yaml.load(configuration_file, Loader=yaml.FullLoader)

            if type(experiment_configurations['seed']) == list:
                for seed in experiment_configurations['seed']:
                    for experiment_configuration_key in experiment_configurations['experiments'].keys():
                        experiment = Experiment(
                            name=experiment_configuration_key + '-seed' + str(seed),
                            experiments_path=experiment_configurations['experiments_path'],
                            results_path=experiment_configurations['results_path'],
                            global_configuration=configuration,
                            experiment_configuration=experiment_configurations['experiments'][experiment_configuration_key],
                            seed=seed
                        )
                        experiments.append(experiment)
            else:
                for experiment_configuration_key in experiment_configurations['experiments'].keys():
                    experiment = Experiment(
                        name=experiment_configuration_key,
                        experiments_path=experiment_configurations['experiments_path'],
                        results_path=experiment_configurations['results_path'],
                        global_configuration=configuration,
                        experiment_configuration=experiment_configurations['experiments'][experiment_configuration_key],
                        seed=experiment_configurations['seed']
                    )
                    experiments.append(experiment)

        return Experiments(experiments)
