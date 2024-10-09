import torch
import cornucopia as cc
import numpy as np
import monai as mn
from copy import deepcopy
from nitorch.tools import qmri
from monai.transforms.transform import MapTransform, Transform
import warnings
warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

param_keys = [
    "te",
    "tr",
    "ti1",
    "ti2",
    "fa1",
    "fa2",
]

seq_keys = ["mprage", "mp2rage", "gre", "fse", "flair", "spgr"]


def reformat_params(sequence, params):
    out_dict = {"field": float(params["receive"].item()) / 10}
    for seq in seq_keys:
        if seq in sequence:
            out_dict[seq] = 1.0
        else:
            out_dict[seq] = 0.0
    for p in param_keys:
        if p in params.keys():
            out_dict[p] = float(params[p])
        else:
            out_dict[p] = 0.0
    if "ti" in params.keys():
        out_dict["ti1"] = float(params["ti"])
    if "fa" in params.keys():
        if isinstance(params["fa"], (tuple, list)):
            out_dict["fa1"] = float(params["fa"][0])
            out_dict["fa2"] = float(params["fa"][1])
        else:
            out_dict["fa1"] = float(params["fa"])
    # add rescaling to [0,1] to make hypernetworks work better
    out_dict["te"] = out_dict["te"] / 1000
    out_dict["tr"] = out_dict["tr"] / 1000
    out_dict["ti1"] = out_dict["ti1"] / 1000
    out_dict["ti2"] = out_dict["ti2"] / 1000
    out_dict["fa1"] = out_dict["fa1"] / 180
    out_dict["fa2"] = out_dict["fa2"] / 180
    return out_dict


dict_keys = [
    "field",
    "mprage",
    "mp2rage",
    "gre",
    "fse",
    "flair",
    "spgr",
    "te",
    "tr",
    "ti1",
    "ti2",
    "fa1",
    "fa2",
]


def forward_model(mpm, params, num_ch=1):
    params = torch.chunk(
        params[0].detach().cpu(), num_ch, dim=-1
    )  # assume batch size 1
    outputs = []
    for p in params:
        if p.sum() > 0:
            out_dict = {k: v for k, v in zip(dict_keys, p.tolist())}
            # print(f"out_dict: {out_dict}")
            out_dict["field"] = out_dict["field"] * 10
            out_dict["te"] = out_dict["te"] * 1000
            out_dict["tr"] = out_dict["tr"] * 1000
            out_dict["ti1"] = out_dict["ti1"] * 1000
            out_dict["ti2"] = out_dict["ti2"] * 1000
            out_dict["fa1"] = out_dict["fa1"] * 180
            out_dict["fa2"] = out_dict["fa2"] * 180
            # print(f"out_dict: {out_dict}")
            if out_dict["mprage"] == 1:
                out = qmri.generators.mprage(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2s=mpm[0, 2],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    ti=out_dict["ti1"],
                    te=out_dict["te"],
                    fa=out_dict["fa1"],
                    device=mpm.device,
                )
            elif out_dict["mp2rage"] == 1:
                out = qmri.generators.mp2rage(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2s=mpm[0, 2],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    ti1=out_dict["ti1"],
                    ti2=out_dict["ti2"],
                    te=out_dict["te"],
                    fa=(out_dict["fa1"], out_dict["fa2"]),
                    device=mpm.device,
                )
            elif out_dict["gre"] == 1:
                out = qmri.gre(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2s=mpm[0, 2],
                    mt=mpm[0, 3],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    te=out_dict["te"],
                    fa=out_dict["fa1"],
                    device=mpm.device,
                ).volume
            elif out_dict["fse"] == 1:
                out = qmri.generators.fse(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2=mpm[0, 2],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    te=out_dict["te"],
                    device=mpm.device,
                )
            elif out_dict["flair"] == 1:
                out = qmri.generators.flair(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2=mpm[0, 2],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    ti=out_dict["ti1"],
                    te=out_dict["te"],
                    device=mpm.device,
                )
            elif out_dict["spgr"] == 1:
                out = qmri.generators.spgr(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2s=mpm[0, 2],
                    mt=mpm[0, 3],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    te=out_dict["te"],
                    fa=out_dict["fa1"],
                    device=mpm.device,
                )
            if len(out.shape) == len(mpm.shape) + 1:
                out = out[0]
            outputs.append(out)
        else:
            outputs.append(torch.zeros_like(mpm[0, 0]))
    return torch.stack(outputs, dim=0)[None]


def ensure_list(x):
    if type(x) != list:
        if type(x) == tuple:
            return list(x)
        else:
            return [x]
    else:
        return x


class log10norm:
    def __init__(self, mu, std=1):
        self.mu = np.log10(mu)
        self.std = std

    def __call__(self):
        return 10 ** np.random.normal(self.mu, self.std)


class uniform:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self):
        return np.random.uniform(self.low, self.high)


class log10uniform:
    def __init__(self, low, high):
        self.low = np.log10(low)
        self.high = np.log10(high)

    def __call__(self):
        return 10 ** np.random.uniform(self.low, self.high)


class BlochTransform(cc.Transform):
    def __init__(
        self,
        num_ch=1,
        flair_params={
            "te": log10norm(20e-3, 0.1),
            "tr": log10uniform(1e-3, 5000e-3),
            "ti": log10uniform(1e-3, 3000e-3),
        },
        fse_params={
            "te": log10uniform(1e-3, 3000e-3),
            "tr": log10uniform(1e-3, 3000e-3),
        },
        mp2rage_params={
            "tr": 2300e-3,
            "ti1": uniform(600e-3, 900e-3),
            "ti2": 2200e-3,
            "tx": log10norm(5.8e-3, 0.5),
            "te": log10norm(2.9e-3, 0.5),
            "fa": (uniform(3, 6), uniform(3, 6)),
            "n": 160,
            "eff": 0.96,
        },
        mprage_params={
            "tr": 2300e-3,
            "ti": uniform(600e-3, 900e-3),
            "tx": uniform(4e-3, 8e-3),
            "te": uniform(2e-3, 4e-3),
            "fa": uniform(5, 12),
            "n": 160,
            "eff": 0.96,
        },
        mprage_t1_params={
            "tr": uniform(1900e-3, 2500e-3),
            "ti": uniform(600e-3, 1200e-3),
            "te": uniform(2e-3, 4e-3),
            "fa": uniform(5, 12),
            "n": uniform(100, 200),
            "eff": uniform(0.8, 1.0),
        },
        spgr_params={
            "te": log10uniform(2e-3, 80e-3),
            "tr": log10uniform(5e-3, 800e-3),
            "fa": uniform(5, 50),
        },
        gre_params={
            "te": log10uniform(2e-3, 80e-3),
            "tr": log10uniform(5e-3, 5000e-3),
            "fa": uniform(5, 50),
        },
        sequence=["mprage", "mp2rage", "gre", "fse", "flair", "spgr"],
        field_strength=(0.3, 7),
    ):
        super().__init__(shared=True)
        self.params = {
            "mprage": mprage_params,
            "mp2rage": mp2rage_params,
            "mprage-t1": mprage_t1_params,
            "gre": gre_params,
            "fse": fse_params,
            "flair": flair_params,
            "spgr": spgr_params,
        }
        self.funcs = {
            "mprage": qmri.generators.mprage,
            "mp2rage": qmri.generators.mp2rage,
            "mprage-t1": qmri.generators.mprage,
            "gre": qmri.gre,
            "fse": qmri.generators.fse,
            "flair": qmri.generators.flair,
            "spgr": qmri.generators.spgr,
        }
        self.sequence = ensure_list(sequence)
        self.field_strength = field_strength

    def sample(self, param, key, sequence):
        if isinstance(param, (float, int)):
            param = np.random.randn() * (param * 0.1) + param
            if param < 0:
                param = -param
            if key == "n":
                param = int(param)
            if key == "eff" and param > 1:
                param = 2 - param
        elif isinstance(param, (log10norm, log10uniform, uniform)):
            param = param()
        elif param == "loguni":
            param = 10 ** (np.random.uniform(-3, 3))
        elif param == "uni":
            param = np.random.uniform(0, 90 if sequence not in ["mprage"] else 10)
        return param

    def get_parameters(self):
        parameters = dict()
        parameters["sequence"] = self.sequence[np.random.randint(len(self.sequence))]
        params = deepcopy(self.params[parameters["sequence"]])
        for key in params.keys():
            if isinstance(params[key], (list, tuple)):
                params[key] = [
                    self.sample(val, key, parameters["sequence"]) for val in params[key]
                ]
            else:
                params[key] = self.sample(params[key], key, parameters["sequence"])
        params["receive"] = torch.Tensor([np.random.uniform(*self.field_strength)])[
            None
        ][None]
        parameters["params"] = params
        parameters["func"] = self.funcs[parameters["sequence"]]
        self.parameters = parameters

        return parameters

    def forward(self, pd, r1, r2s, mt):
        theta_list = [self.get_parameters() for i in range(self.num_ch)]
        img_list = [self.apply_transform(pd, r1, r2s, mt, theta_list[i]) for i in range(self.num_ch)]
        return torch.cat(img_list, dim=0)

    def apply_transform(self, pd, r1, r2s, mt, parameters):
        in_ = (
            [pd, r1, r2s]
            if parameters["sequence"] in ["fse", "flair", "mprage", "mp2rage"]
            else [pd, r1, r2s, mt]
        )
        out_ = parameters["func"](*in_, **parameters["params"])
        return out_.volume[0] if parameters["sequence"] == "gre" else out_


class MONAIBlochTransform(Transform):
    def __init__(self, num_ch=1, **kwargs):
        super().__init__()
        self.bloch = BlochTransform(num_ch=num_ch, **kwargs)

    def __call__(self, pd, r1, r2s, mt):
        return self.bloch(pd, r1, r2s, mt)
    

class MONAIBlochTransformD(MapTransform):
    def __init__(self, keys, num_ch=1, **kwargs):
        super().__init__(keys)
        self.bloch = MONAIBlochTransform(num_ch=num_ch, **kwargs)

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.bloch(data[key])
        return data