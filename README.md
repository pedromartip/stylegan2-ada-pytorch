# Stylegan2-ada-pytorch
A modified stylegan2 to be trained for shoes

## CHANGES
**1. For transfer learning between 2 conditional models with different number of classes.**
In this case we will not copy the parameters of the embedding layer "mapping.embed" in G and D (its shape depends on the number of classes taken as an input). Modification of "copy_params_and_buffers" in "torch_utils/misc.py" in such a way that it does not copy all the parameters of the pretrained model.

```
if name in src_tensors and "embed" not in name:   
        tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
```

Credits to https://github.com/NVlabs/stylegan2-ada-pytorch.git
