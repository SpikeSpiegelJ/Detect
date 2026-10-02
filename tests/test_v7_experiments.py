"""Offline checks for V7 experiment selection and train-only hard example mining."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
import val

spec = importlib.util.spec_from_file_location("miner", ROOT / "dataset/build_focus_hardneg_manifest.py")
miner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(miner)


class ExperimentsTest(unittest.TestCase):
    def test_only_data_and_name_differ(self):
        a, b = train.PRESETS['v7_control'], train.PRESETS['v7_hardneg']
        self.assertEqual({k for k in a if a[k] != b[k]}, {'data', 'name'})
        self.assertEqual(a['lr0'], 0.00005)
        self.assertEqual(a['mosaic'], 0)
        self.assertTrue(a['model'].is_file())

    def test_train_arguments(self):
        for preset in ('v7_control', 'v7_hardneg'):
            detector = Mock()
            with patch.object(sys, 'argv', ['train.py', '--preset', preset]), patch.object(train, 'build_model', return_value=detector):
                train.main()
            kwargs = detector.train.call_args.kwargs
            self.assertEqual(kwargs['epochs'], 40)
            self.assertFalse(kwargs['resume'])
            self.assertEqual(kwargs['seed'], 0)
            self.assertEqual(kwargs['data'], train.PRESETS[preset]['data'])

    def test_eval_selection(self):
        for preset in ('yolo26s', 'v7_control', 'v7_hardneg'):
            with patch.object(sys, 'argv', ['val.py', '--preset', preset]):
                args = val.parse_args()
            self.assertEqual(args.model.parent.parent.name, train.PRESETS[preset]['name'])
            self.assertEqual(args.split, 'val')

    def test_mining_low_confidence_missed_and_false_positive(self):
        image = ROOT / 'fake_train.jpg'
        labels = {image: (np.array([3, 6]), np.array([[.2, .2, .2, .2], [.8, .8, .2, .2]]))}
        boxes = SimpleNamespace(cls=np.array([3, 6]), conf=np.array([.1, .9]),
                                xyxy=np.array([[10, 10, 30, 30], [40, 40, 50, 50]]))
        result = SimpleNamespace(path=str(image), orig_shape=(100, 100), boxes=Mock())
        result.boxes.cpu.return_value.numpy.return_value = boxes
        detector = Mock(names={3:'Feeder_RRU',6:'Feeder_antenna'})
        detector.predict.return_value = iter([result])
        with patch('ultralytics.YOLO', return_value=detector), patch('ultralytics.data.loaders.LoadImagesAndVideos'):
            negative, positive = miner.mine_difficult_samples([image], labels, Path('unused.pt'),1280,2,'cpu',0,.25,.1,.001,.5)
        self.assertEqual(negative[6][image], .9)
        self.assertEqual([x['kind'] for x in positive[image]], ['low_confidence','unmatched_or_localization'])
        self.assertEqual(detector.predict.call_args.kwargs['conf'], .001)


if __name__ == '__main__':
    unittest.main()
