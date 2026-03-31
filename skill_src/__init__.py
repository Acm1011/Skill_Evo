# se_code_auto package
# This file makes se_code_auto a Python package

__version__ = "1.0.0"

# Import commonly used modules for easier access
from . import reward_manager
from . import reward
from . import utils
from . import Synthesizer_dataset
from . import Synthesizer_ray_trainer
from . import Solver_dapo_ray_trainer
# 不在此预加载 solver_offline_driver：否则 `python -m skill_src.solver_offline_driver`
# 会先执行本包 __init__，runpy 会报 RuntimeWarning（模块已在 sys.modules）。
# 需要时请 from skill_src.solver_offline_driver import ... 或 -m 直接运行。
