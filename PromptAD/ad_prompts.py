

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
    "metal_nut": "metal nut"
}


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
