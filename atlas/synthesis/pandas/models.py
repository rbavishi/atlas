from typing import Optional

from atlas.models.imitation import DefaultIndependentOpModel


class PandasModelBasic(DefaultIndependentOpModel):
    def train_op(self, data_dir: str,
                 op_name: str, oid: Optional[str], sid: str, gen_name: str, gen_group: Optional[str]):
        pass
