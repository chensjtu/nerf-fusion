#### for use encoder

from .img_encoder import SpatialEncoder, ImageEncoder
# from .resnetfc import ResnetFC
import warnings


# def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
#     mlp_type = conf.get_string("type", "mlp")  # mlp | resnet
#     if mlp_type == "mlp":
#         # net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)
#         warnings.warn("not implement")
#     elif mlp_type == "resnet":
#         net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
#     elif mlp_type == "empty" and allow_empty:
#         net = None
#     else:
#         raise NotImplementedError("Unsupported MLP type")
#     return net


def make_encoder(**kwargs):
    # enc_type = conf.get_string("type", "spatial")  # spatial | global
    enc_type = "spatial"
    if enc_type == "spatial":
        net = SpatialEncoder()
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net
