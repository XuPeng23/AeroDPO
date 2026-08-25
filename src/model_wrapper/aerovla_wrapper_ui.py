import time
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor 
from scipy.spatial.transform import Rotation as R
import re
from peft import PeftModel
import tkinter as tk
from PIL import ImageTk
from src.model_wrapper.base_model import BaseModelWrapper 
from qwen_vl_utils import process_vision_info 


# RES_TEST = 224
RES_TEST = 384

# SHOW_RES = True
SHOW_RES = False


class AeroVLAWrapper(BaseModelWrapper):

    def __init__(self, model_args, data_args):

        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adapter_path = model_args.model_path 
        base_model_path = model_args.base_model_path 
        print(f"[AerialVLA-Qwen] Loading BASE from: {base_model_path}")
        print(f"[AerialVLA-Qwen] Loading ADAPTER from: {adapter_path}")
        
        self.NUM_BINS = 99
        self.norm_stats = {
            'forward': {'min': 0.0, 'max': 5.0},
            'down':    {'min': -5.0, 'max': 5.0},
            'yaw':     {'min': -1.1, 'max': 1.1}
        }

        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        self.processor.image_processor.max_pixels = RES_TEST * RES_TEST * 2
        
        self.processor.tokenizer.padding_side = 'left' 
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.merge_and_unload()
        
        # self.model = base_model
        
        self.model.to(self.device)
        self.model.eval()

    def get_semantic_direction(self, curr_state, target_pos):
        pos = np.array(curr_state['position'])
        q_raw = curr_state['orientation']
        if isinstance(q_raw, dict):
            quat = [q_raw.get('x',0), q_raw.get('y',0), q_raw.get('z',0), q_raw.get('w',1)]
        else:
            quat = q_raw

        vec_world = np.array(target_pos) - pos
        dist_xy = np.linalg.norm(vec_world[:2])
        
        r = R.from_quat(quat)
        vec_body = r.inv().apply(vec_world)
        
        x, y = vec_body[0], vec_body[1]
        angle_deg = np.degrees(np.arctan2(y, x))

        if dist_xy < 0.01:
            return ""
        else:
            if -15 <= angle_deg <= 15: return "straight ahead "
            if 15 < angle_deg <= 60: return "forward-right "
            if 60 < angle_deg <= 120: return "to your right "
            if 120 < angle_deg <= 180: return "to your right rear "
            if -60 <= angle_deg < -15: return "forward-left "
            if -120 <= angle_deg < -60: return "to your left "
            if -180 <= angle_deg < -120: return "to your left rear "

        return ""

    def prepare_inputs(self, episodes, target_positions, instructions=None):

        messages_batch = []
        batch_prompts = []  # For AerialVLA Prompt
        
        for i, ep in enumerate(episodes):
            curr_state = ep[-1]['sensors']['state'] 
            rgb_list = ep[-1]['rgb'] 
            frame_front_rgb = cv2.cvtColor(rgb_list[0], cv2.COLOR_BGR2RGB)
            frame_down_rgb = cv2.cvtColor(rgb_list[4], cv2.COLOR_BGR2RGB)
            img_front = Image.fromarray(frame_front_rgb)
            img_down = Image.fromarray(frame_down_rgb)

            if SHOW_RES:
                if i == 0:
                    print(f"\n[Resolution Debug] 1. AirSim Original Size (Front WxH): {img_front.size}")
            
            target_pos = target_positions[i]
            dir_text = self.get_semantic_direction(curr_state, target_pos)
            obj_desc = instructions[i].split("degrees from you.")[1].split(" Please control")[0].strip()

            # ================== UI START ====================================================
            # d_front = cv2.copyMakeBorder(cv2.resize(cv2.cvtColor(np.array(img_front), cv2.COLOR_RGB2BGR), (224, 224)), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            # d_down = cv2.copyMakeBorder(cv2.resize(cv2.cvtColor(np.array(img_down), cv2.COLOR_RGB2BGR), (224, 224)), 0, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            # view_panel = np.vstack((d_front, d_down))

            # txt_panel = cv2.copyMakeBorder(np.zeros((120, 420, 3), dtype=np.uint8), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            # words = f"Prompt: Fly {dir_text.strip() if dir_text.strip() else 'nearby'} and find the target. {obj_desc}".split()
            # x, y = 15, 28
            # for w in words:
            #     c = (0, 0, 255) if w in (dir_text.strip() if dir_text.strip() else "nearby").split() else (255, 255, 255)
            #     tw = cv2.getTextSize(w, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            #     if x + tw > 405: x, y = 15, y + 22
            #     cv2.putText(txt_panel, w, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            #     x += tw + 6

            # act_panel = cv2.copyMakeBorder(np.zeros((35, 420, 3), dtype=np.uint8), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            # cv2.putText(act_panel, "Action: Inferring...", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

            # if not hasattr(self, 'tk_roots'): self.tk_roots, self.tk_lbls = {}, {}
            # if i not in self.tk_roots:
            #     r = tk.Tk()
            #     r.withdraw()
            #     self.tk_roots[i], self.tk_lbls[i] = r, []
            #     # for geo in ["+1745+1085", "+1170+655", "+1170+785"]:
            #     for geo in ["+2880+1210", "+2230+970", "+2230+1100"]:
            #         top = tk.Toplevel(r)
            #         top.overrideredirect(True)
            #         top.attributes('-topmost', True)
            #         top.geometry(geo)
            #         lbl = tk.Label(top, borderwidth=0, highlightthickness=0)
            #         lbl.pack()
            #         self.tk_lbls[i].append(lbl)

            # for idx, panel in enumerate([view_panel, txt_panel, act_panel]):
            #     img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)))
            #     self.tk_lbls[i][idx].configure(image=img)
            #     self.tk_lbls[i][idx].image = img
            # self.tk_roots[i].update()
            # ================== UI END ======================================================
            
            target_size = (RES_TEST, RES_TEST)

            img_front = img_front.resize(target_size, resample=Image.Resampling.BICUBIC)
            img_down = img_down.resize(target_size, resample=Image.Resampling.BICUBIC)
            w, h = target_size
            mosaic = Image.new('RGB', (w, h * 2), (0, 0, 0))
            mosaic.paste(img_front, (0, 0))
            mosaic.paste(img_down, (0, h))

            if SHOW_RES:
                if i == 0:
                    print(f"[Resolution Debug] 2. After PIL Resize & Mosaic (WxH): {mosaic.size}")
            
            prompt_text = f"Fly {dir_text}and find the target. {obj_desc}\nAction: "
            batch_prompts.append(prompt_text)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": mosaic},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            messages_batch.append(messages)

        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        image_inputs, video_inputs = process_vision_info(messages_batch)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding="longest",
            return_tensors="pt"
        )

        if SHOW_RES:
            if "pixel_values" in inputs:
                print(f"[Resolution Debug] 3. Processor Final Tensor Shape: {inputs['pixel_values'].shape}")
            if "image_grid_thw" in inputs:
                thw = inputs['image_grid_thw'][0].tolist()
                print(f"[Resolution Debug] 4. Qwen-VL Image Patch Grid (T, H, W): {thw}")
                print(f"[Resolution Debug] 5. Total Visual Tokens generated: {thw[1] * thw[2]}")
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs, None, batch_prompts


    def run(self, inputs, episodes, rot_to_targets):
    
        t0 = time.time()

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,
                use_cache=True,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            
        time_elapse = time.time() - t0
            
        batch_actions = []
        batch_should_stop = []

        input_len = inputs['input_ids'].shape[1]

        for i, output_ids in enumerate(generated_ids):
            new_ids = output_ids[input_len:]
            text = self.processor.decode(new_ids, skip_special_tokens=True).strip()
            
            pred_fwd, pred_down, pred_yaw = self._parse_action_from_text(text)
            
            #  ================== UI START ====================================================
            # if hasattr(self, 'tk_lbls') and i in self.tk_lbls:
            #     act_panel = cv2.copyMakeBorder(np.zeros((35, 420, 3), dtype=np.uint8), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            #     cv2.putText(act_panel, f"Action: {text}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            #     img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(act_panel, cv2.COLOR_BGR2RGB)))
            #     self.tk_lbls[i][2].configure(image=img)
            #     self.tk_lbls[i][2].image = img
            #     self.tk_roots[i].update()
            #  ================== UI END ======================================================
            
            # ================== Debug Info ==================
            print(f"\n[Step Debug] Env {i} {'-'*60}")
            print(f" [Inference Time]: {time_elapse:.3f}s")
            # print(f" [Gen Action] >>> {text}") 
            print(f" [Parsed Val] >>> Fwd={pred_fwd:.2f}, Down={pred_down:.2f}, Yaw={pred_yaw:.2f}")
            # print("-" * 76)
            # ====================================================
            
            has_land_str = "LAND" in text or "<LAND>" in text
            has_land_num = (pred_fwd < 0.01) and (abs(pred_down) < 0.01) and (abs(pred_yaw) < 0.01)
            has_land = has_land_str or has_land_num
            batch_should_stop.append(has_land)
            
            batch_actions.append({
                'fwd': pred_fwd,
                'down': pred_down,
                'yaw': pred_yaw
            })
            
        return batch_actions, batch_should_stop
    

    def _parse_action_from_text(self, text):
        matches = re.findall(r"\d+", text)
        fwd, down, yaw = 0.0, 0.0, 0.0
        
        if len(matches) >= 3:
            try:
                bin_fwd, bin_down, bin_yaw = map(int, matches[:3])
                def dequantize(bin_val, axis):
                    vmin, vmax = self.norm_stats[axis]['min'], self.norm_stats[axis]['max']
                    return (max(0, min(self.NUM_BINS - 1, bin_val)) / (self.NUM_BINS - 1)) * (vmax - vmin) + vmin
                fwd, down, yaw = dequantize(bin_fwd, 'forward'), dequantize(bin_down, 'down'), dequantize(bin_yaw, 'yaw')
            except:
                pass
        return fwd, down, yaw
        