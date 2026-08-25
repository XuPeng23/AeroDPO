import os
from pathlib import Path
import sys
import time
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from utils.logger import logger
from utils.utils import *
from src.model_wrapper.aerovla_wrapper_ui import AeroVLAWrapper
# from src.model_wrapper.aerovla_wrapper_HIL import AeroVLAWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from src.vlnce_src.assist import Assist
from src.vlnce_src.closeloop_util import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized

import sys
import logging
import warnings
from transformers import logging as hf_logging


hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*meshgrid.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


def eval(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir):

    model_wrapper.eval()
    
    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            
            for i, batch_item in enumerate(env_batchs):
                print(f" [New Mission] ---------------------------------------------")
                print(f" Instruction: \"{batch_item['instruction']}\"")
                print(f" Target Object: {batch_item['object']['asset_name']}")
                
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)

            pbar.update(n=eval_env.batch_size)
            
            inputs, rot_to_targets, _ = model_wrapper.prepare_inputs(
                batch_state.episodes, 
                batch_state.target_positions, 
                instructions=batch_state.instructions
            )

            for t in range(int(args.maxWaypoints) + 1):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))

                is_terminate = batch_state.check_batch_termination(t)
                
                if is_terminate:
                    break
                
                t_start = time.time()
                
                # 1. Model Inference
                batch_actions, model_stops = model_wrapper.run(
                    inputs=inputs,
                    episodes=batch_state.episodes,
                    rot_to_targets=rot_to_targets
                )
                t_run = time.time()

                # 2. AirSim 
                eval_env.makeActions(batch_actions)
                t_fly = time.time()

                # 3. Get Images
                time.sleep(0.01)
                outputs = eval_env.get_obs()
                t_obs = time.time()

                # 4. State Update
                batch_state.update_from_env_output(outputs)
                batch_state.predict_dones = model_stops 
                batch_state.update_metric()
                t_metric = time.time()
                
                # 5. Qwen
                inputs, _, _ = model_wrapper.prepare_inputs(
                    episodes=batch_state.episodes, 
                    target_positions=batch_state.target_positions, 
                    instructions=batch_state.instructions
                )
                t_prep = time.time()

                print(f"\n[Profiler] Run(Model): {t_run - t_start:.3f}s | Fly(AirSim): {t_fly - t_run:.3f}s | Obs: {t_obs - t_fly:.3f}s | Metric: {t_metric - t_obs:.3f}s | Prep(Qwen): {t_prep - t_metric:.3f}s\n")

        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path
    
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    
    setup()

    assert CheckPort(), 'error port'

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False
    
    model_wrapper = AeroVLAWrapper(model_args=model_args, data_args=data_args)
    
    # [Important Note for AerialVLA]
    # The 'Assist' module here functions STRICTLY as a backend environment monitor 
    # for evaluation purposes (e.g., depth-based collision detection, stuck monitoring). 
    # It DOES NOT provide any oracle path guidance, ground-truth waypoints, 
    # or dense hints to the AerialVLA Model during the inference loop.
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)
    
    eval(model_wrapper=model_wrapper,
         assist=assist,
         eval_env=eval_env,
         eval_save_dir=eval_save_path)
    
    eval_env.delete_VectorEnvUtil()
