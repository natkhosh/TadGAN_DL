#!/usr/bin/env python
# coding: utf-8

import os
import logging
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import anomaly_detection


class Encoder(nn.Module):
    """
    Генератор E служит кодировщиком, который отображает последовательности временных рядов в скрытое пространство
    signal_shape = длине последовательности временного ряда
    """

    def __init__(self, encoder_path, signal_shape=100):
        super(Encoder, self).__init__()
        self.signal_shape = signal_shape
        self.lstm = nn.LSTM(input_size=self.signal_shape, hidden_size=20, num_layers=1, bidirectional=True)
        self.dense = nn.Linear(in_features=40, out_features=20)
        self.encoder_path = encoder_path

    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return x


class Decoder(nn.Module):
    """
    Генератор G служит декодером, преобразующим скрытое пространство в восстановленные временные ряды
    """

    def __init__(self, decoder_path, signal_shape=100):
        super(Decoder, self).__init__()
        self.signal_shape = signal_shape
        self.lstm = nn.LSTM(input_size=20, hidden_size=64, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(in_features=128, out_features=self.signal_shape)
        self.decoder_path = decoder_path

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return x


class CriticX(nn.Module):
    """
    Дискриминатор Critic Cx должен различать последовательности реальных временных рядов из X
    и сгенерированные последовательности временных рядов из G(z)
    """
    def __init__(self, critic_x_path, signal_shape=100):
        super(CriticX, self).__init__()

        self.signal_shape = signal_shape
        self.dense1 = nn.Linear(in_features=self.signal_shape, out_features=20)
        self.dense2 = nn.Linear(in_features=20, out_features=1)
        self.critic_x_path = critic_x_path

    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class CriticZ(nn.Module):
    """
    Дискриминатор Critic Cz измеряет качество отображения в скрытое пространство.
    """
    def __init__(self, critic_z_path):
        super(CriticZ, self).__init__()

        self.dense1 = nn.Linear(in_features=20, out_features=1)
        self.critic_z_path = critic_z_path

    def forward(self, x):
        x = self.dense1(x)
        return x

    def unroll_signal(self, x):
        x = np.array(x).reshape(100)
        return np.median(x)


def test(self):
    """
    Возвращает датафрейм с исходными значениями, восстановленными значениемя, ошибкой реконструкции, Critic оценкой.
    """
    df = self.test_dataset.copy()
    X_ = list()

    RE = list()  # Reconstruction error
    CS = list()  # Critic score

    for i in range(0, df.shape[0]):
        x = df.rolled_signal[i]
        x = np.reshape(x, (1, 100, 1))
        z = Encoder(x)                  # E(x)
        z = np.expand_dims(z, axis=2)
        x_ = Decoder(z)                 # G(E(x))

        re = anomaly_detection.dtw_reconstruction_error(np.squeeze(x_).numpy(), np.squeeze(x).numpy()) # reconstruction error
        cs = CriticX(x)
        cs = np.squeeze(cs).numpy()
        RE.append(re)
        CS.append(cs)

        x_ = anomaly_detection.unroll_signal(x_)

        X_.append(x_)

    df['generated_signals'] = X_

    return df
