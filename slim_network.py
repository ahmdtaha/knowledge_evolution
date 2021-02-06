from layers import bn_type

def extract_slim(split_model,model):
    for (dst_n, dst_m), (src_n, src_m) in zip(split_model.named_modules(), model.named_modules()):
        if hasattr(src_m, "weight") and src_m.weight is not None:
            if hasattr(src_m, "mask"):
                src_m.extract_slim(dst_m,src_n,dst_n)
                # if src_m.__class__ == conv_type.SplitConv:
                # elif src_m.__class__ == linear_type.SplitLinear:
            elif src_m.__class__ == bn_type.SplitBatchNorm: ## BatchNorm has bn_maks not mask
                src_m.extract_slim(dst_m)

