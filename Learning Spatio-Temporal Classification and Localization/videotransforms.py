import numpy as np
import numbers
import random
import torch

class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i+h, j:j+w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def HorizontalFlip(imgs, boxes):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    """
    Args:
        img (seq Images): seq Images to be flipped.
    Returns:
        seq Images: Randomly flipped seq images.
    """
    boxes_flipped = np.zeros_like(boxes)
    if len(boxes.shape) == 3:
        for jx,j in enumerate(boxes):
             for ix,i in enumerate(j):
                if i.sum() == 0: continue
                x,y,w,h = boxes[jx][ix]
                boxes_flipped[jx][ix] = np.array([(112-x-w),y,w,h]) 
        imgs_flipped = torch.flip(imgs, [4])
    else:
        for ix,i in enumerate(boxes):
            if i.sum() == 0: continue
            x,y,w,h = boxes[ix]
            boxes_flipped[ix] = np.array([(112-x-w),y,w,h])
        imgs_flipped = torch.flip(imgs, [3])

    return (imgs_flipped, boxes_flipped)


def HorizontalFlip_0(imgs, boxes):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    """
    Args:
        img (seq Images): seq Images to be flipped.
    Returns:
        seq Images: Randomly flipped seq images.
    """
    boxes_flipped = np.zeros_like(boxes)
    if len(boxes.shape) == 3:
        for jx,j in enumerate(boxes):
             for ix,i in enumerate(j):
                x,y,w,h = boxes[jx][ix]
                boxes_flipped[jx][ix] = np.array([(112-x-w),y,w,h]) 
    else:
        for ix,i in enumerate(boxes):
            x,y,w,h = boxes[ix]
            boxes_flipped[ix] = np.array([(112-x-w),y,w,h])
             
    return (np.array(torch.flip(torch.tensor(imgs), [2])).copy(),boxes_flipped)

def convert_img(img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format
    """
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img



class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(
                ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image\
                but got list of {0}'.format(type(clip[0])))
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = tensor_clip.div(255)
            return tensor_clip

class ClipToTensor_seg(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=1, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(
                ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image\
                but got list of {0}'.format(type(clip[0])))
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = tensor_clip.div(255)
            return tensor_clip
