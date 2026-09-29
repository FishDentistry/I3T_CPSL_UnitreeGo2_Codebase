# Copyright 2026 I3T
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grounding DINO model loading and inference isolated from ROS interfaces."""

import os

import numpy as np


class GroundingDinoBackend:
    """Load Grounding DINO and return pixel-space detections."""

    def __init__(self, config_path, checkpoint_path, device):
        if not os.path.isfile(config_path):
            raise RuntimeError(
                'Grounding DINO config does not exist: {}'.format(
                    config_path
                )
            )
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                'Grounding DINO checkpoint does not exist: {}'.format(
                    checkpoint_path
                )
            )

        try:
            import torch
            from groundingdino.datasets import transforms as dino_transforms
            from groundingdino.models import build_model
            from groundingdino.util.misc import clean_state_dict
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import get_phrases_from_posmap
        except ImportError as error:
            raise RuntimeError(
                'Grounding DINO dependencies are unavailable: {}'.format(
                    error
                )
            )

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA was requested but torch.cuda.is_available() is false'
            )

        arguments = SLConfig.fromfile(config_path)
        arguments.device = device
        model = build_model(arguments)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(clean_state_dict(state_dict), strict=False)
        model.eval()
        model.to(device)

        self._torch = torch
        self._model = model
        self._device = device
        self._get_phrases = get_phrases_from_posmap
        self._transform = dino_transforms.Compose([
            dino_transforms.RandomResize([800], max_size=1333),
            dino_transforms.ToTensor(),
            dino_transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    @property
    def device(self):
        """Return the selected torch device."""
        return self._device

    def detect(self, image_rgb, targets, box_threshold, text_threshold):
        """Detect target phrases in an RGB uint8 image."""
        from PIL import Image

        if not targets:
            return []
        caption = ' . '.join(targets).strip()
        if not caption.endswith('.'):
            caption += ' .'

        image_pillow = Image.fromarray(image_rgb.astype(np.uint8), 'RGB')
        image_tensor, _ = self._transform(image_pillow, None)
        image_tensor = image_tensor.to(self._device)

        with self._torch.no_grad():
            outputs = self._model(
                image_tensor[None], captions=[caption]
            )
        logits = outputs['pred_logits'].cpu().sigmoid()[0]
        boxes = outputs['pred_boxes'].cpu()[0]
        keep = logits.max(dim=1)[0] > box_threshold
        logits = logits[keep]
        boxes = boxes[keep]

        tokenized = self._model.tokenizer(caption)
        image_height, image_width = image_rgb.shape[:2]
        detections = []
        for box, logit in zip(boxes, logits):
            phrase = self._get_phrases(
                logit > text_threshold,
                tokenized,
                self._model.tokenizer,
            ).replace('.', '').strip()
            if not phrase:
                phrase = 'unknown'
            center_x, center_y, width, height = box.tolist()
            x_min = (center_x - width / 2.0) * image_width
            y_min = (center_y - height / 2.0) * image_height
            x_max = (center_x + width / 2.0) * image_width
            y_max = (center_y + height / 2.0) * image_height
            detections.append({
                'label': phrase,
                'score': float(logit.max().item()),
                'bounding_box': (x_min, y_min, x_max, y_max),
            })
        return detections
