from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RELLISDataset_Group5(CustomDataset):
    """RELLIS dataset.

    """



    CLASSES = ("background/obstacle", "stable", "granular", "high resistance", "none")

    PALETTE = [[ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],
            [ 0, 153, 153 ],[ 0, 128, 255 ]]

    def __init__(self, **kwargs):
        super(RELLISDataset_Group5, self).__init__(
            img_suffix='_orig.jpg',
            seg_map_suffix='_group5.png',
            **kwargs)
        self.CLASSES = ("background/obstacle", "stable", "granular", "high resistance", "none")
        self.PALETTE = [[ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],
            [ 0, 153, 153 ],[ 0, 128, 255 ]]
