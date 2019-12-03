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

from experiments.base_trainer import BaseTrainer

import torch
from torch import nn
import torch.optim as optim
import os

class WavenetTrainer(BaseTrainer):

    def __init__(self, device, data_stream, configuration, experiments_path, experiment_name, **kwargs):
        super().__init__(device, data_stream, configuration, experiments_path, experiment_name)

        self._model = kwargs.get('model', None)


        # checkpoint = torch.load('/home/taresh/wavenet_pretrained.pth')
        # state_dict = self._model._decoder._wavenet.state_dict()
        # for (k,v) in state_dict.items():
        #     a, b = k.split('.')[:2]
        #     if 'conv_layers' in a and 'last_conv' not in a and int (b) <= 17:
        #         print (k)
        #         state_dict[k] = checkpoint['state_dict'][k]
        # self._model._decoder._wavenet.load_state_dict(state_dict)
        # for i in range (18):
        #     for p in self._model._decoder._wavenet.conv_layers[i].parameters():
        #         p.requires_grad=False
        # # TODO: make more modular
        # print ('frozen first 17 layers of conv_layers using /home/taresh/wavenet_pretrained.pth')
        self._criterion = kwargs.get('criterion', nn.CrossEntropyLoss(reduction='none'))
        self._optimizer = kwargs.get('optimizer',
            optim.Adam(self._model.parameters(), lr=configuration['learning_rate'], amsgrad=True))
        self._configuration = configuration

    def iterate(self, data, epoch, iteration, iterations, train_bar):

        one_hot = data['one_hot'].to(self._device)
        shape = list(one_hot.shape)
        new_shape = tuple(shape[:3] + [8000] + [shape[4]])
        one_hot_padded = torch.zeros(new_shape)
        one_hot_padded[:,:,:,:self._configuration['length'],:] = one_hot
        one_hot_padded = one_hot_padded.to(self._device)
        source = data['input_features'].to(self._device)
        speaker_id = data['speaker_id'].to(self._device)
        if self._configuration['output_type'] == 'log_filterbank':
            target = data['output_features'].to(self._device).permute(0, 2, 1).contiguous().float()
        elif self._configuration['output_type'] == 'audio':
            target = data['one_hot'].to(self._device).squeeze(1).contiguous().long()
        else:
            raise ('not implemented')

        self._optimizer.zero_grad()

        reconstructed_x, x_dec, vq_loss, losses, perplexity, encoding_indices, concatenated_quantized = \
            self._model(source, one_hot_padded, None)

        n, c, t, e = target.shape
        target_new = torch.argmax(target.squeeze(-1).permute(0, 2, 1)[:,1:,:].reshape(n*(t-1), c), dim=1)
        reconstructed_x_new = reconstructed_x.squeeze(-1).permute(0, 2, 1)[:,:t-1,:].reshape(n*(t-1), c)
        
        reconstruction_loss = self._criterion(reconstructed_x_new, target_new)
        reconstruction_loss = torch.mean(torch.sum(reconstruction_loss.reshape(n, t-1), dim=1))

        loss = vq_loss + reconstruction_loss
        losses['reconstruction_loss'] = reconstruction_loss.item()
        losses['loss'] = loss.item()

        self._record_codebook_stats(iteration, iterations, self._model.vq,
            concatenated_quantized, encoding_indices, data['speaker_id'], epoch)

        loss.backward()

        self._record_gradient_stats({'model': self._model, 'encoder': self._model.encoder,
            'vq': self._model.vq, 'decoder': self._model.decoder}, iteration, iterations, epoch)

        self._optimizer.step()

        perplexity_value = perplexity.item()
        train_bar.set_description('Epoch {}: loss {:.4f} perplexity {:.3f}'.format(
            epoch + 1, losses['loss'], perplexity_value))

        return losses, perplexity_value

    def save(self, epoch, **kwargs):
        torch.save({
            'experiment_name': self._experiment_name,
            'epoch': epoch + 1,
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'train_res_recon_error': kwargs.get('train_res_recon_error', -1),
            'train_res_perplexity': kwargs.get('train_res_perplexity', -1)},
            os.path.join(self._experiments_path, '{}_{}_checkpoint.pth'.format(
                self._experiment_name, epoch + 1))
        )

