from functools import partial

import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from PIL import Image
from pathlib import Path

from toolbox.image import bicubic_rescale
from toolbox.image import modcrop
from toolbox.dataset import DATASET


def load_set(name, mode='train', lr_sub_size=11, lr_sub_stride=5, scale=3, random=0):
    hr_sub_size = lr_sub_size * scale
    hr_sub_stride = lr_sub_stride * scale
    if random:
        x_list = np.random.random(random)
        y_list = np.random.random(random)
    else:
        x_list, y_list = None, None
    lr_gen_sub = partial(generate_sub_images, size=lr_sub_size,
                         stride=lr_sub_stride, random=(x_list, y_list))
    hr_gen_sub = partial(generate_sub_images, size=hr_sub_size,
                         stride=hr_sub_stride, random=(x_list, y_list))

    lr_sub_arrays = []
    hr_sub_arrays = []
    for path in DATASET[name.upper()].__getattr__(mode):
        lr_image, hr_image = load_image_pair(path, scale=scale, mode='YCbCr')
        lr_sub_arrays += [img_to_array(img) for img in lr_gen_sub(lr_image)]
        hr_sub_arrays += [img_to_array(img) for img in hr_gen_sub(hr_image)]
    x = np.stack(lr_sub_arrays).astype('uint8')
    y = np.stack(hr_sub_arrays).astype('uint8')
    return x, y


def load_image_pair(path, scale=3, mode='RGB'):
    image = load_img(path)
    image = image.convert(mode)
    hr_image = modcrop(image, scale)
    lr_image = bicubic_rescale(hr_image, 1 / scale, mode)
    return lr_image, hr_image


def generate_sub_images(image, size, stride, random=(None, None)):
    x_list, y_list = random
    if x_list is not None and y_list is not None:
        for x in x_list:
            for y in y_list:
                i = int(image.size[0] * x)
                j = int(image.size[1] * y)
                yield image.crop([i, j, i + size, j + size])
    else:
        for i in range(0, image.size[0] - size + 1, stride):
            for j in range(0, image.size[1] - size + 1, stride):
                yield image.crop([i, j, i + size, j + size])


def get_frame_pitch(mode, size):
    """Get bytes length of one frame.
    For the detail of mode fourcc, please see https://www.fourcc.org/

    :param mode: Must be any of {YV12, YV21, NV12, NV21, RGB, BGR}
    :param size: frame size, must be a tuple or list
    :return: frame bytes, [channel0 bytes, channel1 bytes, channel2 bytes, ...]

    RGB, BGR, and UV channel of NV12, NV21 is packed, while YV12 and YV21 is planar, hence we have:
      - **channel0** of YV12, YV21, NV12, NV21 if Y
      - **channel1** of YV12 is U, of YV21 is V, of NV12 is UV, of NV21 is VU
      - **channel2** of YV12 is V, of YV21 is U
    """

    if mode in ('YV12', 'YV21'):
        return size[0] * size[1] * 3 // 2, [size[0] * size[1], size[0] * size[1] // 4, size[0] * size[1] // 4]
    if mode in ('NV12', 'NV21'):
        return size[0] * size[1] * 3 // 2, [size[0] * size[1], size[0] * size[1] // 2]
    if mode in ('RGB', 'BGR'):
        return size[0] * size[1] * 3, [size[0] * size[1] * 3]


def get_frame_channel_shape(mode, size):
    """Get each channel's shape according to mode and frame size.
    For the detail of mode fourcc, please see https://www.fourcc.org/

    :param mode: Must be any of {YV12, YV21, NV12, NV21, RGB, BGR}
    :param size: frame size, must be a tuple or list
    :return: 3-D tensor of [C, W, H] for YV12, YV21, NV12, NV21; [W, H, C] for RGB and BGR
    """
    if mode in ('YV12', 'YV21'):
        return np.array([1, size[0], size[1]]), np.array([1, size[0] // 2, size[1] // 2]), np.array(
            [1, size[0] // 2, size[1] // 2])
    if mode in ('NV12', 'NV21'):
        return np.array([1, size[0], size[1]]), np.array([2, size[0] // 2, size[1] // 2])
    if mode in ('RGB', 'BGR'):
        return np.array([size[0], size[1], 3])


def turn_frame_to_yuv(frame, mode, turn_array=False):
    """Change frame color space from any mode to YUV

    :param frame: 3-D tensor in either [H, W, C] or [C, H, W]
    :param mode: Must be any of {YV12, YV21, NV12, NV21, RGB, BGR}
    :param turn_array: turn PIL image to numpy array
    :return: 3-D tensor of YUV in [H, W, C]
    """

    if not isinstance(frame, list):
        raise TypeError("frame must be a list of numpy array")
    if not mode in ('YV12', 'YV21', 'NV12', 'NV21', 'RGB', 'BGR'):
        raise ValueError("invalid mode: " + mode)
    if mode in ('YV12', 'YV21', 'NV12', 'NV21'):
        if mode in ('YV12', 'YV21'):
            y, u, v = frame
        elif mode in ('NV12', 'NV21'):
            y, uv = frame
            u = uv.flatten()[0::2].reshape([1, uv.shape[1] // 2, uv.shape[2]])
            v = uv.flatten()[1::2].reshape([1, uv.shape[1] // 2, uv.shape[2]])
        else:
            y = u = v = None
        y = np.transpose(y)
        u = np.transpose(u)
        v = np.transpose(v)
        if '21' in mode:
            u, v = v, u
        up_u = np.zeros(shape=[u.shape[0] * 2, u.shape[1] * 2, u.shape[2]])
        up_v = np.zeros(shape=[v.shape[0] * 2, v.shape[1] * 2, v.shape[2]])
        up_u[0::2, 0::2, :] = up_u[0::2, 1::2, :] = u
        up_u[1::2, ...] = up_u[0::2, ...]
        up_v[0::2, 0::2, :] = up_v[0::2, 1::2, :] = v
        up_v[1::2, ...] = up_v[0::2, ...]
        yuv = np.concatenate([y, up_u, up_v], axis=-1)
        yuv = np.transpose(yuv, [1, 0, 2])  # PIL needs [W, H, C]
        image = Image.fromarray(yuv.astype('uint8'), mode='YCbCr')
    elif mode in ('RGB', 'BGR'):
        assert len(frame) is 1
        rgb = np.asarray(frame[0])
        if mode == 'BGR':
            rgb = rgb[..., ::-1]
        rgb = np.transpose(rgb, [1, 0, 2])
        image = Image.fromarray(rgb, mode='RGB').convert('YCbCr')
    else:
        raise RuntimeError("unreachable!")
    return img_to_array(image) if turn_array else image


def load_video_set_raw(name, scale=3, size=(0, 0), mode='YV12', method='train', seq_depth=1,
                       patch_size=48, stride=1, max_patch=None):
    """Load raw video data into numpy arrays in low, high-resolution pairs

    :param name: name defined in dataset, see Dataset
    :param scale: low-resolution down-scale factor
    :param size: a 2-int tuple of fixed width and height of the input video
    :param mode: format of input video. Must be any of {YV12, YV21, NV12, NV21, RGB, BGR}
    :param method: Must be one of {'train', 'eval', 'test'}
    :param seq_depth: video sequence length in network
    :param patch_size: the size of training patch in mini-batch
    :param stride: the crop stride to get patches
    :param max_patch: the max number of patches

    :return: numpy array of [N, S, H, W, C], where N is patch amounts, S is sequence length
    """

    mode = mode.upper()
    method = method.lower()
    if not isinstance(size, (tuple, list)):
        raise TypeError("size must be a 2-tuple or list")
    if not mode in ('YV12', 'YV21', 'NV12', 'NV21', 'RGB', 'BGR'):
        raise ValueError("invalid input format " + mode + ". Must be one of {YV12, YV21, NV12, NV21, RGB, BGR}")
    if not method in ('train', 'eval', 'test'):
        raise ValueError("invalid mode value: " + method)
    if size[0] == 0 or size[1] == 0:
        raise ValueError("size can't be zero")
    if patch_size < 16 or seq_depth < 1 or stride < 1:
        raise ValueError("invalid parameters")
    if patch_size % scale != 0:
        patch_size = (patch_size // scale + 1) * scale
    if stride % scale != 0:
        stride = (stride // scale + 1) * scale

    lr_gen_sub = partial(generate_sub_images, size=patch_size // scale, stride=stride // scale)
    hr_gen_sub = partial(generate_sub_images, size=patch_size, stride=stride)
    total_files = len(DATASET[name.upper()].__getattr__(method))
    patch_each_file = max_patch // total_files if max_patch else 0
    lr_batch, hr_batch = [], []

    for path in DATASET[name.upper()].__getattr__(method):
        fd = Path(path).open('rb')
        data = np.fromfile(fd, dtype='uint8', sep='')
        pitch, channel_pitch = get_frame_pitch(mode, size)
        all_frames = np.split(data, range(pitch, data.size, pitch), axis=0)
        lr_seq, hr_seq = [], []
        for frame in all_frames:
            channel_split = [sum(channel_pitch[:i]) for i in range(1, len(channel_pitch))]
            channel = np.split(frame, channel_split)
            channel = [np.reshape(channel[i], get_frame_channel_shape(mode, size)[i]) for i in range(len(channel))]
            yuv = turn_frame_to_yuv(channel, mode)
            yuv_hr = modcrop(yuv, scale)
            yuv_lr = bicubic_rescale(yuv_hr, 1 / scale)
            x = np.stack([img_to_array(img) for img in lr_gen_sub(yuv_lr)])
            y = np.stack([img_to_array(img) for img in hr_gen_sub(yuv_hr)])
            lr_seq.append(x)
            hr_seq.append(y)
        x = np.stack(lr_seq).transpose([1, 0, 2, 3, 4])
        y = np.stack(hr_seq).transpose([1, 0, 2, 3, 4])
        lr_batch += np.split(x, range(seq_depth, x.shape[1], seq_depth), axis=1)[:-1]
        hr_batch += np.split(y, range(seq_depth, y.shape[1], seq_depth), axis=1)[:-1]
    batch = [(lr, hr) for lr, hr in zip(lr_batch, hr_batch)]
    np.random.shuffle(batch)
    feature = np.concatenate([lr for lr, _ in batch]).astype('uint8')
    label = np.concatenate([hr for _, hr in batch]).astype('uint8')
    return feature, label
