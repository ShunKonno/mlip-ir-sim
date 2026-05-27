from __future__ import annotations


class IRSpectrumSimulator:
    """
    将来的に、平衡化→プロダクション→双極子トラッキング→スペクトル推定までを担う。
    現時点ではREADMEのAPI形状を固定するための骨格のみ。
    """

    def __init__(self, system):
        self.system = system

    def run(
        self,
        *,
        temperature: float,
        pressure: float,
        eq_time_ps: float,
        prod_time_ps: float,
        timestep_fs: float = 0.5,
    ):
        raise NotImplementedError("IR simulation engine is not implemented yet.")

