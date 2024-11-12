import torch
from torch2trt import torch2trt
from backbone.model_irse import IR_50
from demo_face.nets_retinaface.retinaface import RetinaFace
from util.config import cfg_re50

retinaface_model_path = '/home/admin2/PycharmProjects/pythonProject/demo_face/model/Retinaface_resnet50.pth'
ir_50_model_path = '/home/admin2/PycharmProjects/pythonProject/demo_face/model/backbone_ir50_asia.pth'

retinaface_model = RetinaFace(cfg=cfg_re50, phase='eval', pre_train=False).eval()
ir_50_model = IR_50([112, 112]).eval()

retinaface_model.load_state_dict(torch.load(retinaface_model_path))
ir_50_model.load_state_dict(torch.load(ir_50_model_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
retinaface_model.to(device)
ir_50_model.to(device)

dummy_input_retinaface = torch.ones(1, 3, 640, 640).to(device)
dummy_input_ir_50 = torch.ones(1, 3, 112, 112).to(device)

retinaface_model_trt = torch2trt(retinaface_model, [dummy_input_retinaface], fp16_mode=True, max_workspace_size=1 << 25)
ir_50_model_trt = torch2trt(ir_50_model, [dummy_input_ir_50], fp16_mode=True, max_workspace_size=1 << 25)

torch.save(retinaface_model_trt.state_dict(), 'model/retinaface_model_trt.pth')
torch.save(ir_50_model_trt.state_dict(), 'model/ir_50_model_trt.pth')
