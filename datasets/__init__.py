import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from .dataset import CLIPDataset
from .mvtec import load_mvtec, mvtec_classes
from .visa import load_visa, visa_classes
from .spectrum import load_spectrum, spectrum_classes
from .sample import load_sample, sample_classes
from .deceptive_signal import load_deceptive_signal, deceptive_signal_classes
from .burst_signal import load_burst_signal, burst_signal_classes
from .dsss_signal import load_dsss_signal, dsss_classes
from .chirp_signal import load_chirp_signal, chirp_signal_classes
from .pulse_signal import load_pulse_signal, pulse_signal_classes
from .wideband_pulse import load_wideband_pulse, wideband_pulse_classes
from .wideband_pulse_png import load_wideband_pulse_png, wideband_pulse_png_classes
from .rf_spe_png import load_rf_spe_png, rf_spe_png_classes


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec': load_mvtec,
    'visa': load_visa,
    'spectrum': load_spectrum,
    'sample': load_sample,
    'deceptive_signal': load_deceptive_signal,
    'burst_signal': load_burst_signal,
    'dsss_signal': load_dsss_signal,
    'chirp_signal': load_chirp_signal,
    'pulse_signal': load_pulse_signal,
    'wideband_pulse': load_wideband_pulse,
    'wideband_pulse_png': load_wideband_pulse_png,
    'rf_spe_png': load_rf_spe_png,
}

dataset_classes = {
    'mvtec': mvtec_classes,
    'visa': visa_classes,
    'spectrum': spectrum_classes,
    'sample': sample_classes,
    'deceptive_signal': deceptive_signal_classes,
    'burst_signal': burst_signal_classes,
    'dsss_signal': dsss_classes,
    'chirp_signal': chirp_signal_classes,
    'pulse_signal': pulse_signal_classes,
    'wideband_pulse': wideband_pulse_classes,
    'wideband_pulse_png': wideband_pulse_png_classes,
    'rf_spe_png': rf_spe_png_classes,
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase, **kwargs):

    # 提取 burst_signal 额外的参数
    extra_kwargs = {}
    if kwargs.get('dataset') == 'burst_signal':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm10db')
    elif kwargs.get('dataset') == 'dsss_signal':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm10db')
    elif kwargs.get('dataset') == 'chirp_signal':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm10db')
    elif kwargs.get('dataset') == 'pulse_signal':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm20db')
    elif kwargs.get('dataset') == 'wideband_pulse':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm20db')
    elif kwargs.get('dataset') == 'wideband_pulse_png':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm20db')
    elif kwargs.get('dataset') == 'rf_spe_png':
        extra_kwargs['noise_level'] = kwargs.get('noise_level', 'm10db')
    elif kwargs.get('dataset') == 'deceptive_signal':
        extra_kwargs['freq'] = kwargs.get('freq', None)

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        **extra_kwargs
    )

    # RF datasets 的 __getitem__ 是纯 cv2 IO + resize, 无随机/全局状态, 线程安全.
    # morph_fusion 预处理在 model.transform 里, 在主进程里跑——但即便如此,
    # 多 worker 拿原始 cv2 array 也能把数据流水线从 0 worker 的同步阻塞中解放出来.
    rf_datasets = ('rf_spe_png',)
    is_rf = kwargs.get('dataset') in rf_datasets
    rf_workers = 8

    if phase == 'train':
        if is_rf:
            data_loader = DataLoader(
                dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                num_workers=rf_workers, pin_memory=True, persistent_workers=True,
            )
        else:
            data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                      num_workers=0)
    else:
        if is_rf:
            data_loader = DataLoader(
                dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                num_workers=rf_workers, pin_memory=True, persistent_workers=True,
            )
        else:
            data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                     num_workers=4, pin_memory=True)


    # debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    # # logger.info(debug_str)

    return data_loader, dataset_inst
