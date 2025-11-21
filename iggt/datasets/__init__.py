from .pointodyssey import PointOdysseyDUSt3R # noqa
from .sintel import SintelDUSt3R # noqa
from .tartanair import TarTanAirDUSt3R # noqa
from .blendedmvs import BlendedMVS # noqa
from .dl3dv import Dl3dv # noqa
from .replica import Replica
from .re10k import Re10K
from .carla import Carla # noqa
from .arkitscenes import ARKitScenes # noqa
from .arkitscenes_high import ARKitScenesHigh # noqa
from .wildrgb import Wildrgb
from .megadepth import MegaDepth # noqa
from .bedlam import Bedlam # noqa
from .mapfree import MapFree
from .co3d import Co3d # noqa
from .cop3d import Cop3d # noqa
from .dynamic_replica import Dynamic_Replica # noqa
from .scannet import Scannet # noqa
from .mp3d import Mp3d
from .hypersim import Hypersim # noqa
from .mvs_synth import Mvs_Synth # noqa
from .kubric import Kubric # noqa
from .spring import Spring # noqa
from .uasol import Uasol # noqa
from .unreal4k import Unreal4k # noqa
from .vkitti import Vkitti # noqa
from .waymo import Waymo # noqa
from .scannetpp import Scannetpp # noqa
from .infinigen import Infinigen # noqa
from .utils.transforms import ColorJitter, ImgNorm # noqa


def get_data_loader(dataset, seq_min_len, seq_max_len,
                    batch_size, num_workers=8,
                    shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from iggt.datasets.utils.misc import get_world_size, get_rank
    
    world_size = get_world_size()
    rank = get_rank()
    if isinstance(dataset, str):
        dataset = eval(dataset)
    
    try:
        sampler = dataset.make_sampler(batch_size, seq_min_len, seq_max_len, 
                                       shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
            
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1, #Do not modify this
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        )   
    
    return data_loader
