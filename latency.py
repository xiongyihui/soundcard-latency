#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import threading

import sounddevice as sd
import numpy as np


rate = 16000
frame_size = rate // 100

sd.default.samplerate = rate
sd.default.channels = (1, 1)
sd.default.dtype = ("float32", "float32")
sd.default.blocksize = frame_size
sd.default.latency = ("low", "low")
sd.default.prime_output_buffers_using_stream_callback = True

np.random.seed()
noise = np.random.normal(0, 1, rate // 10) * np.hanning(rate // 10)
print(f"noise dtype {noise.dtype}")
# noise = (noise.astype('float64') * (2**24)).astype('int32')
zeros = np.zeros(rate // 10)
source = np.concatenate((zeros, noise, zeros)).astype("float32")

output_index = 0
output_time = 0
input_time = 0
adc_time = 0
dac_time = 0
input_blocks = []


def callback(data, frames, t, status):
    global input_time
    global adc_time

    if not input_blocks:
        input_time = time.time() - frames / rate
        adc_time = t.inputBufferAdcTime
        dt = t.currentTime - t.inputBufferAdcTime
        print(f"input latency {dt * 1000} ms")
        print(data.shape)

    if status:
        print(status)

    input_blocks.append(data.copy())


def out_callback(outdata, frames, t, status):
    global output_index
    global output_time
    global dac_time

    if not output_index:
        output_time = time.time()
        dac_time = t.outputBufferDacTime
        dt = t.outputBufferDacTime - t.currentTime
        print(f"output latency {dt * 1000} ms")

    if status:
        print(status)

    data = source[output_index : output_index + frames]
    size = len(data)
    output_index += size
    if size < frames:
        outdata[:size, 0] = data
        outdata[size:, 0].fill(0)
        raise sd.CallbackStop
    else:
        outdata[:, 0] = data


event = threading.Event()

with sd.InputStream(callback=callback):
    with sd.OutputStream(
        callback=out_callback,
        finished_callback=event.set,
    ):
        event.wait()

    time.sleep(0.1)


recording = np.concatenate(input_blocks)
print(recording.dtype)
print(recording.shape)


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    """
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    """

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / (np.abs(R) + np.finfo(float).eps), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau, cc


sig = recording[:, 0]
ref = source
offset, cc = gcc_phat(sig, ref, fs=1)


dt = offset * 1000 / rate
delay = (output_time - input_time) * 1000
delay2 = (dac_time - adc_time) * 1000
print(f"dt = {dt} ms")
print(f"delay = {delay} ms")
print(f"delay 2 = {delay} ms")
print(f"system latency = {dt - delay} ms")
print(f"system latency 2 = {dt - delay2} ms")

offset = int(offset)
centre = (len(sig) + len(ref)) // 2 + offset
margin = 100
t = np.linspace(0, len(ref), len(ref))

try:
    import matplotlib.pyplot as plt

    plt.subplot(311)
    plt.plot(ref)
    # plt.plot(t, ref, '-', t, sig[dt:dt+len(ref)] * 32)
    plt.subplot(312)
    plt.plot(sig[offset : offset + len(ref)])
    # plt.plot(sig)
    plt.subplot(313)
    plt.plot(cc[centre - margin : centre + margin])
    plt.show()
except ImportError:
    pass
