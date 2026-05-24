

# 部分数据集类名在生成 prompt 前需要统一成标准写法。
class_mapping = {
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut",
}

rf_signal_dataset_mapping = {
    "burst_signal": "burst",
    "chirp_signal": "chirp",
    "dsss_signal": "dsss",
    "deceptive_signal": "deceptive",
}

rf_state_anomaly = [
    "abnormal {}",
    "{} with anomalous signal energy",
    "{} with unexpected interference",
    "{} with injected radio-frequency interference",
    "{} with abnormal time-frequency structure",
    "{} with signal energy inconsistent with the normal background",
]

generic_state_anomaly = [
    "abnormal {}",
    "damaged {}",
    "flawed {}",
    "defective {}",
    "{} with anomaly",
    "{} with defect",
]

rf_domain_state_anomaly = [
    "abnormal {}",
    "anomalous {}",
    "{} with anomaly",
    "{} with abnormal pattern",
    "{} with unusual visual pattern",
    "{} with unexpected structure",
]

rf_signal_structured_classname = {
    "burst": "burst signal radio frequency spectrogram",
    "chirp": "chirp signal radio frequency spectrogram",
    "dsss": "DSSS spread-spectrum radio frequency spectrogram",
    "deceptive": "deceptive signal radio frequency spectrogram",
}

rf_signal_structured_state_anomaly = {
    "burst": [
        "{} with abnormal short-duration burst energy",
        "{} with unexpected transient narrowband pulse",
        "{} with hidden low-power burst trace",
        "{} with vertically truncated burst structure",
        "{} with burst energy inconsistent with normal background",
        "{} with anomalous time-localized RF emission",
    ],
    "chirp": [
        "{} with broken diagonal chirp trace",
        "{} with distorted frequency-sweeping slope",
        "{} with interrupted slanted time-frequency streak",
        "{} with abnormal chirp slope angle",
        "{} with faint low-power chirp interference",
        "{} with spurious diagonal RF artifact",
    ],
    "dsss": [
        "{} with abnormal diffuse wideband texture",
        "{} with uneven spread-spectrum power distribution",
        "{} with disrupted noise-like spectral pattern",
        "{} with missing segment in the wideband spread spectrum",
        "{} with anomalous broadband spectral flatness",
        "{} with weak DSSS-like interference embedded in background",
    ],
    "deceptive": [
        "{} with spoofed signal structure",
        "{} with signal pattern in an unauthorized frequency band",
        "{} with deceptive RF emission mimicking normal activity",
        "{} with counterfeit spectrum occupancy",
    ],
}

rf_scene_background_mapping = {
    "WeaponMuseum_spectrum": "an indoor spectrum scene with relatively stable background activity",
    "Playground_spectrum": "an open outdoor spectrum scene with sparse background activity",
    "TimeSquare_spectrum": "a crowded urban spectrum scene with complex background occupancy",
    "Gymnasium_spectrum": "a semi-enclosed spectrum scene with moderate background activity",
}


def get_rf_signal_key(classname, dataset_name=None):
    if dataset_name in rf_signal_dataset_mapping:
        return rf_signal_dataset_mapping[dataset_name]

    name = classname.lower()
    for signal_key in ("burst", "chirp", "dsss", "deceptive"):
        if signal_key in name:
            return signal_key

    return None


def is_rf_prompt_class(classname, dataset_name=None):
    return get_rf_signal_key(classname, dataset_name) is not None


def get_rf_scene_background(classname=None):
    return rf_scene_background_mapping.get(classname)


def get_prompt_classname(classname, dataset_name=None, prompt_mode="rf"):
    if prompt_mode == "legacy":
        return class_mapping.get(classname, classname)

    signal_key = get_rf_signal_key(classname, dataset_name)
    if signal_key is not None:
        if prompt_mode == "generic":
            return "image"
        prompt_classname = "radio frequency spectrogram"
        if prompt_mode == "rf_signal_structured":
            prompt_classname = rf_signal_structured_classname.get(signal_key, prompt_classname)
        if prompt_mode == "rf_scene_conditioned":
            scene_background = get_rf_scene_background(classname=classname)
            if scene_background is not None:
                prompt_classname = f"{prompt_classname} in {scene_background}"
        return prompt_classname

    return class_mapping.get(classname, classname)


def get_abnormal_prompt_states(classname, dataset_name=None, prompt_mode="rf"):
    if prompt_mode == "legacy":
        return state_anomaly + class_state_abnormal.get(classname, [])

    signal_key = get_rf_signal_key(classname, dataset_name)
    if is_rf_prompt_class(classname, dataset_name):
        if prompt_mode == "generic":
            return generic_state_anomaly
        if prompt_mode == "rf_domain":
            return rf_domain_state_anomaly
        states = []
        if prompt_mode == "rf_signal_structured" and signal_key is not None:
            for state in rf_signal_structured_state_anomaly.get(signal_key, []):
                if state not in states:
                    states.append(state)
            return states
        for state in rf_state_anomaly:
            if state not in states:
                states.append(state)
        if prompt_mode not in {"rf_object_agnostic"} and signal_key is not None:
            for state in class_state_abnormal.get(signal_key, []):
                if state not in states:
                    states.append(state)
        if prompt_mode not in {"rf_object_agnostic"}:
            for state in class_state_abnormal.get(classname, []):
                if state not in states:
                    states.append(state)
        return states

    return state_anomaly + class_state_abnormal.get(classname, [])


# 所有类别都会共用的通用异常描述模板。
state_anomaly = ["damaged {}",
                 "flawed {}",
                 "abnormal {}",
                 "imperfect {}",
                 "blemished {}",
                 "{} with flaw",
                 "{} with defect",
                 "{} with damage"]

# 旧实验里保留的备用异常模板集合。
abnormal_state0 = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

#
# 每个类别自己的专属异常模板，会和上面的通用模板一起用于构造异常 prompt。
class_state_abnormal = {
    'bottle': ['{} with large breakage', '{} with small breakage', '{} with contamination'],
    'toothbrush': ['{} with defect', '{} with anomaly'],
    'carpet': ['{} with hole', '{} with color stain', '{} with metal contamination', '{} with thread residue', '{} with thread', '{} with cut'],
    'hazelnut': ['{} with crack', '{} with cut', '{} with hole', '{} with print'],
    'leather': ['{} with color stain', '{} with cut', '{} with fold', '{} with glue', '{} with poke'],
    'cable': ['{} with bent wire', '{} with missing part', '{} with missing wire', '{} with cut', '{} with poke'],
    'capsule': ['{} with crack', '{} with faulty imprint', '{} with poke', '{} with scratch', '{} squeezed with compression'],
    'grid': ['{} with breakage',  '{} with thread residue', '{} with thread', '{} with metal contamination', '{} with glue', '{} with a bent shape'],
    'pill': ['{} with color stain', '{} with contamination', '{} with crack', '{} with faulty imprint', '{} with scratch', '{} with abnormal type'],
    'transistor': ['{} with bent lead', '{} with cut lead', '{} with damage', '{} with misplaced transistor'],
    'metal_nut': ['{} with a bent shape ', '{} with color stain', '{} with a flipped orientation', '{} with scratch'],
    'screw': ['{} with manipulated front',  '{} with scratch neck', '{} with scratch head'],
    'zipper': ['{} with broken teeth', '{} with fabric border', '{} with defect fabric', '{} with broken fabric', '{} with split teeth', '{} with squeezed teeth'],
    'tile': ['{} with crack', '{} with glue strip', '{} with gray stroke', '{} with oil', '{} with rough surface'],
    'wood': ['{} with color stain', '{} with hole', '{} with scratch', '{} with liquid'],

    'candle': ['{} with melded wax', '{} with foreign particals', '{} with extra wax', '{} with chunk of wax missing', '{} with weird candle wick', '{} with damaged corner of packaging', '{} with different colour spot'],
    'capsules': ['{} with scratch', '{} with discolor', '{} with misshape', '{} with leak', '{} with bubble'],
    # 'capsules': [],
    'cashew': ['{} with breakage', '{} with small scratches', '{} with burnt', '{} with stuck together', '{} with spot'],
    'chewinggum': ['{} with corner missing', '{} with scratches', '{} with chunk of gum missing', '{} with colour spot', '{} with cracks'],
    'fryum': ['{} with breakage', '{} with scratches', '{} with burnt', '{} with colour spot', '{} with fryum stuck together', '{} with colour spot'],
    'macaroni1': ['{} with color spot', '{} with small chip around edge', '{} with small scratches', '{} with breakage', '{} with cracks'],
    'macaroni2': ['{} with color spot', '{} with small chip around edge', '{} with small scratches', '{} with breakage', '{} with cracks'],
    'pcb1': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb2': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb3': ['{} with bent', '{} with scratch', '{} with missing', '{} with melt'],
    'pcb4': ['{} with scratch', '{} with extra', '{} with missing', '{} with wrong place', '{} with damage', '{} with burnt', '{} with dirt'],
    'pipe_fryum': ['{} with breakage', '{} with small scratches', '{} with burnt', '{} with stuck together', '{} with colour spot', '{} with cracks'],

    # 频谱类别使用更贴近”干扰 / 人为干扰 / 注入干扰 / 异常信号”的描述。
    '16QAM': ['{} with interference', '{} with intentional interference', '{} with injected interference', '{} with anomalous signal'],
    'CHIRP': ['{} with interference', '{} with intentional interference', '{} with injected interference', '{} with anomalous signal'],

    # Chirp 信号检测 - 强调斜线特征（时频图中频率随时间扫描呈斜线）
    'chirp': [
        # 斜线异常 - 断裂/缺失
        '{} with broken diagonal line',
        '{} with interrupted slanted streak',
        '{} with diagonal line missing segment',
        '{} with discontinuous slope trace',
        # 斜线异常 - 形态畸变
        '{} with distorted diagonal pattern',
        '{} with bent slanted line',
        '{} with curved instead of diagonal trace',
        '{} with irregular slope angle',
        # 斜线异常 - 多余/干扰
        '{} with extra diagonal artifact',
        '{} with anomalous cross streak',
        '{} with stray slanted line',
        '{} with spurious diagonal interference',
    ],
    'GMSK': ['{} with interference', '{} with intentional interference', '{} with injected interference', '{} with anomalous signal'],
    'QPSK': ['{} with interference', '{} with intentional interference', '{} with injected interference', '{} with anomalous signal'],
    'bearing': ['{} with interference', '{} with intentional interference', '{} with injected interference', '{} with anomalous signal'],
    'burst': ['{} with interference', '{} with intentional interference', '{} with injected interference', '{} with anomalous signal'],
    'stealthy': ['{} with interference', '{} with intentional interference', '{} with injected interference', '{} with anomalous signal'],

    # 欺骗信号检测 - 强调"与正常信号极度相似但出现在未授权频段"
    'deceptive': [
        # 强调视觉相似但频段错误
        '{} with identical appearance in unauthorized frequency',
        '{} visually identical but in wrong frequency band',
        '{} same signal pattern in forbidden frequency',
        '{} indistinguishable from normal but in unlicensed band',
        # 强调欺骗性/伪装性
        '{} spoofed signal in unauthorized band',
        '{} deceptive signal mimicking normal in wrong frequency',
        '{} camouflage signal in unexpected frequency',
        '{} counterfeit signal in out-of-band location',
        # 强调频段违规
        '{} in prohibited frequency with identical spectrum',
        '{} in restricted band with same visual pattern',
    ],

    # DSSS 信号检测 - 强调宽带扩频谱的断裂、畸变、功率分布异常
    'dsss': [
        # 宽带谱断裂/缺失
        '{} with broken wideband spectrum',
        '{} with discontinuous spread spectrum band',
        '{} with missing frequency segment in wideband',
        # 宽带谱畸变/失真
        '{} with distorted wideband pattern',
        '{} with deformed spread spectrum shape',
        '{} with warped wideband structure',
        # 功率分布异常（扩频应该功率均匀散布）
        '{} with uneven power distribution across band',
        '{} with anomalous power concentration',
        '{} with irregular spectral density pattern',
        # 异常谱线/伪影
        '{} with spurious spectral artifact',
        '{} with unexpected spectral spike in band',
        # 扩频码异常（视觉层面描述）
        '{} with abnormal spread spectrum texture',
        '{} with disrupted wideband noise-like pattern',
    ],

    # Burst 信号检测 - 强调短时、窄带、隐蔽、纵轴不完整等视觉特征
    'burst': [
        # 短时特征（时间维度窄条）
        '{} with short-duration narrow signal',
        '{} with transient narrow pulse',
        # 窄带特征（频率维度集中）
        '{} with narrowband burst signal',
        '{} with concentrated narrow frequency burst',
        # 隐蔽性（低强度、微弱、难以察觉）
        '{} with stealthy short narrow signal',
        '{} with hidden low-power burst',
        # 纵轴不完整（频谱截断、频率维度被切割）
        '{} with vertically truncated narrow signal',
        '{} with incomplete frequency range burst',
    ],
    }

