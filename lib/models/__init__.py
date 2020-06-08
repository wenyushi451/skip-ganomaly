##
import importlib
import io

from onnx import optimizer
import onnx
import torch
from torch.onnx import OperatorExportTypes

##
def load_model(opt, dataloader):
    """ Load model based on the model name.

    Arguments:
        opt {[argparse.Namespace]} -- options
        dataloader {[dict]} -- dataloader class

    Returns:
        [model] -- Returned model
    """
    model_name = opt.model
    model_path = f"lib.models.{model_name}"
    model_lib  = importlib.import_module(model_path)
    model = getattr(model_lib, model_name.title())
    return model(opt, dataloader)


def load_then_export_model(opt, dataloader):
    """ Load model based on the model name.

    Arguments:
        opt {[argparse.Namespace]} -- options

    Returns:
        [model] -- Returned model
    """
    model_name = opt.model
    model_path = f"lib.models.{model_name}"
    model_lib  = importlib.import_module(model_path)
    model = getattr(model_lib, model_name.title())
    model_inst = model(opt, dataloader)
    model_inst.load_weights(opt.epoch)
    
    input_data = torch.empty((1, opt.nc, opt.isize, opt.isize),
                             dtype=next(model_inst.netg.parameters()).dtype,
                             device=next(model_inst.netg.parameters()).device)
    
    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model_inst.netd,
                (input_data, ),
                f,
                operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                verbose=True,  # NOTE: uncomment this for debugging
                # export_params=True,
            )
            onnx_model_d = onnx.load_from_string(f.getvalue())
            
            torch.onnx.export(
                model_inst.netg,
                (input_data, ),
                f,
                operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                verbose=True,  # NOTE: uncomment this for debugging
                # export_params=True,
            )
            onnx_model_g = onnx.load_from_string(f.getvalue())
            
    model_name = model_name.replace('-', '')
    
    path_g = f"./output/{model_name}/{opt.dataset}/train/weights/exportG.onnx"
    print(f'saving model in {path_g}')
    onnx.save(onnx_model_g, path_g)
    
    path_d = f"./output/{model_name}/{opt.dataset}/train/weights/exportD.onnx"
    print(f'saving model in {path_d}')
    onnx.save(onnx_model_d, path_d)
    
    return 