import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ 单调的簿记 """
    work_dir = config.system.work_dir
    # 如果工作目录不存在，则创建它
    os.makedirs(work_dir, exist_ok=True)
    # 记录参数（如果有的话）
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # 记录配置本身
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ 一个轻量级的配置类，灵感来自yacs """
    # TODO: 转换为像yacs中那样从字典继承？
    # TODO: 实现冻结以防止自伤
    # TODO: 在读取/写入参数时进行额外的存在/覆盖检查？

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ 需要一个辅助函数来支持嵌套缩进以进行漂亮的打印 """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ 返回配置的字典表示 """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        从预期来自命令行的字符串列表更新配置，即sys.argv[1:]。

        参数预期为`--arg=value`的形式，并且arg可以使用.来表示嵌套的子属性。例如：

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "期望每个覆盖参数的形式为--arg=value，得到%s" % arg
            key, val = keyval # 解包

            # 首先将val转换为python对象
            try:
                val = literal_eval(val)
                """
                这里需要一些解释。
                - 如果val只是一个字符串，literal_eval将抛出ValueError
                - 如果val表示一个事物（如3, 3.14, [1,2,3], False, None等），它将被创建
                """
            except ValueError:
                pass

            # 找到适当的对象以插入属性
            assert key[:2] == '--'
            key = key[2:] # 去掉'--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # 确保该属性存在
            assert hasattr(obj, leaf_key), f"{key}不是配置中存在的属性"

            # 覆盖属性
            print("命令行覆盖配置属性%s为%s" % (key, val))
            setattr(obj, leaf_key, val)