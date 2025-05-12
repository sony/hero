

import hydra
import time

from accelerate import Accelerator
from accelerate.logging import get_logger
from config.hydra_configs.configs.configs import TrainerConfig, instantiate_with_cfg

from accelerate.utils import set_seed, ProjectConfiguration

import os
import numpy as np
import torch
import wandb
import PIL
import random
import datetime
import tempfile

from ddpo_trainer import DDPOTrainer
from rl4dgm.user_feedback_interface.preference_functions import PickScore
from rl4dgm.user_feedback_interface.user_feedback_interface import HumanFeedbackInterface, AIFeedbackInterface
from rl4dgm.utils.query_generator import EvaluationQueryGenerator
from rl4dgm.models.dataset import TripletDataset, HumanDatasetSimilarity
from rl4dgm.encoder_trainers import RepresentationTrainer


logger = get_logger(__name__)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


###### Main training loop ######
@hydra.main(version_base=None, config_path="config/hydra_configs/conf", config_name="config")
def main(cfg: TrainerConfig) -> None:

    print("-"*50)
    print("Config", cfg)
    # print("\n\n", cfg.dataset.dataset_name)
    print("-"*50)
    # breakpoint()

    # create directories to save sampled images
    img_save_dir = os.path.join("sampled_images", cfg.ddpo_conf.run_name, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=False)
    
    
    n_human_data_for_training = cfg.human_encoder_conf.n_data_needed_for_training # minimum number of human data to accumulate before training the human encoder
    
    # set seed
    np.random.seed(cfg.ddpo_conf.seed)
    torch.manual_seed(cfg.ddpo_conf.seed)
    random.seed(cfg.ddpo_conf.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.ddpo_conf.seed)

    ############################################
    # Initialize accelerator
    ############################################
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(cfg.ddpo_conf.logdir, cfg.ddpo_conf.run_name),
        automatic_checkpoint_naming=True,
        total_limit=cfg.ddpo_conf.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=cfg.ddpo_conf.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train_gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=cfg.ddpo_conf.train_gradient_accumulation_steps*cfg.ddpo_conf.train_num_update)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="active-diffusion",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.ddpo_conf.run_name}},
        )


    ############################################
    # Initialize trainers
    ############################################
    torch.manual_seed(cfg.ddpo_conf.seed) # DONT REMOVE THIS
    
    print("\nInitializing DDPO trainer...")
    ddpo_trainer = DDPOTrainer(
        config=cfg.ddpo_conf, 
        logger=logger, 
        accelerator=accelerator
    )
    print("...done\n")

    print("Initializing human encoder trainer...")
    human_encoder_trainer = RepresentationTrainer(
        config_dict=dict(cfg.human_encoder_conf),
        seed=cfg.ddpo_conf.seed,
        trainset=None,
        testset=None,
        accelerator=accelerator,
        save_dir=os.path.join(cfg.ddpo_conf.logdir, cfg.ddpo_conf.run_name, "human_encoder"),
    )
    print("...done\n")


    ############################################
    # Initialize other components
    ############################################
    query_generator = EvaluationQueryGenerator(seed=cfg.ddpo_conf.seed)

    ############################################
    # Initialize feedback interfaces
    ############################################
    latent_save_dir = os.path.join(cfg.ddpo_conf.logdir, cfg.ddpo_conf.run_name, "latents")
    os.makedirs(latent_save_dir, exist_ok=True)

    if cfg.query_conf.feedback_agent == "human":
        feedback_interface = HumanFeedbackInterface(
            feedback_type=cfg.query_conf.feedback_type,
            run_name=cfg.ddpo_conf.run_name,
        )
    elif cfg.query_conf.feedback_agent == "ai":
        feedback_interface = AIFeedbackInterface(
            feedback_type=cfg.query_conf.feedback_type,
            preference_function=PickScore,
        )
    # setup query config
    # random query is the only supported query method now
    if cfg.query_conf.query_type == "random":
        query_kwargs = {"n_queries" : cfg.query_conf.n_feedback_per_query}
    elif cfg.query_conf.query_type == "all":
        query_kwargs = {}
    
    ############################################################
    # Initialize human dataset to accumulate
    ############################################################
    human_dataset = HumanDatasetSimilarity(
        device=accelerator.device,
    )
    
    #########################################################
    # MAIN TRAINING LOOP
    #########################################################
    n_total_human_feedback = 0
    best_sample_latent = None
    best_sample_latent_prev = None
    best_sample_encoding = None
    best_sample_encoding_from_prev_encoder = None
    best_to_prev_best_cossim = None
    best_to_best_cossim_cur_and_prev_encoder = None
    predicted_cossim = None
    best_noise_latent = None
    positive_noise_latents = None

    n_ddpo_train_calls = 0 # number of times ddpo training loop was called
    loop = 0 # number of times the main training loop has run
    
    while n_ddpo_train_calls < cfg.ddpo_conf.n_outer_loops:

        ############################################################
        # Sample from SD model
        ############################################################
        if cfg.ddpo_conf.sample_from_all_good_latents:
            print("sampling from best and good image latent")
            samples, all_latents = ddpo_trainer.sample(
                logger=logger,
                epoch=loop,
                save_images=True,
                img_save_dir=img_save_dir,
                high_reward_latents=positive_noise_latents,
            )
        else:
            print("sampling from random latent")
            samples, all_latents = ddpo_trainer.sample(
                logger=logger,
                epoch=loop,
                save_images=True,
                img_save_dir=img_save_dir,
                high_reward_latents=None,
            )

        sd_features = all_latents[:,-1,:,:,:].float()
        sd_noises = all_latents[:, 0,:,:,:].float()
        print("Sampled from SD model with shape ", sd_features.shape)

        ############################################################
        # Compute cosine similarity before training
        ############################################################
        if best_sample_latent is not None:
            with torch.no_grad():
                human_encodings = human_encoder_trainer.model(sd_features)
                best_sample_encoding = human_encoder_trainer.model(best_sample_latent).squeeze()
                predicted_cossim = torch.nn.functional.cosine_similarity(human_encodings, best_sample_encoding.expand(human_encodings.shape))

        ############################################################
        # Active query --> get human rewards
        ############################################################
        print("Begin active query")
        if loop == 0 and cfg.query_conf.query_everything_fisrt_iter:
            # if first iteration, query everything
            query_indices = np.arange(sd_features.shape[0])
        
        else:
            query_indices = query_generator.get_query_indices(
                indices=np.arange(samples.shape[0]), 
                query_type=cfg.query_conf.query_type,
                # n_queries=cfg.query_conf.n_feedbacks_per_query,
                **query_kwargs,
            )
            
            # if still in warmup phase and the number of queries is too little, add random indices to query
            if query_indices.shape[0] < cfg.query_conf.min_n_queries:
                is_warmup = human_encoder_trainer.n_calls_to_train >= cfg.human_encoder_conf.n_warmup_epochs
                if is_warmup or not cfg.query_conf.only_enforce_min_queries_during_warmup:
                    n_random_queries = cfg.query_conf.min_n_queries - query_indices.shape[0]
                    indices = np.setdiff1d(np.arange(samples.shape[0]), query_indices)
                    additional_random_query_indices = query_generator.get_query_indices(
                        indices=indices,
                        query_type="random",
                        n_queries=n_random_queries,
                    )
                    query_indices = np.concatenate([query_indices, additional_random_query_indices])
                    print(f"not enough queries from active query. randomly sampled {n_random_queries}: {query_indices}")

        new_best_sample_index, positive_indices, negative_indices = feedback_interface.query_batch(
            prompts=["amusement street"]*query_indices.shape[0],
            image_batch=samples[query_indices],
            query_indices=np.arange(query_indices.shape[0]),
        )
        if new_best_sample_index is not None:
            is_best_image_updated = True
            positive_indices = np.setdiff1d(positive_indices, new_best_sample_index)
            # if best sample was updated, update best_sample_latent
            best_sample_latent_prev = best_sample_latent
            best_sample_latent = sd_features[query_indices[new_best_sample_index]]
            best_noise_latent = sd_noises[query_indices[new_best_sample_index]].unsqueeze(0)
        else:
            is_best_image_updated = False

        positive_noise_latents = torch.cat([
            sd_noises[query_indices[positive_indices]],
            best_noise_latent,
        ], dim=0)
    
        # update number of human feedback collected
        n_total_human_feedback += query_indices.shape[0]    
        n_positive_samples = positive_indices.shape[0]
        n_negative_samples = negative_indices.shape[0]
        print("n data", query_indices.shape[0])
        print(f"positive indices ({n_positive_samples}) : {positive_indices}")
        print(f"negative indices ({n_negative_samples}) : {negative_indices}")
        print(f"n positive samples", n_positive_samples)

        ############################################################
        # Train human encoder
        ############################################################
        # make a dataset
        print("\nPreparing dataset for human encoder training")
        human_encoder_trainset = TripletDataset(
            features=sd_features[query_indices],
            best_sample_feature=best_sample_latent,
            device=accelerator.device,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            sampling_method="default", # TODO add to config
        )
        human_encoder_trainer.initialize_dataloaders(trainset=human_encoder_trainset)
       
        human_encodings = human_encoder_trainer.model(sd_features)
        best_sample_encoding = human_encoder_trainer.model(best_sample_latent).squeeze()
  
        initial_rewards = torch.nn.functional.cosine_similarity(human_encodings, best_sample_encoding.expand(human_encodings.shape))
        initial_rewards = (initial_rewards+1)/2
        print(f'Number of positive indices: {len(positive_indices)}')
        _, indices = torch.topk(initial_rewards, len(positive_indices))
        
        common_elements = torch.isin(indices.cpu(), torch.from_numpy(positive_indices))
        print(f'Number of positive indices in top k: {common_elements.sum().item()}')
        
        human_encoder_trainer.train_model()

        ############################################################
        # Get final rewards for DDPO training
        ############################################################
        # get human encodings for samples in this batch
        with torch.no_grad():
            human_encodings = human_encoder_trainer.model(sd_features)
            best_sample_encoding_from_prev_encoder = best_sample_encoding
            best_sample_encoding = human_encoder_trainer.model(best_sample_latent).squeeze()
            if best_sample_encoding_from_prev_encoder is not None and not is_best_image_updated:
                best_to_best_cossim_cur_and_prev_encoder = torch.nn.functional.cosine_similarity(best_sample_encoding, best_sample_encoding_from_prev_encoder, dim=0).item()
                best_to_best_cossim_cur_and_prev_encoder = (best_to_best_cossim_cur_and_prev_encoder + 1) / 2
            if best_sample_latent_prev is not None and is_best_image_updated:
                best_sample_encoding_prev = human_encoder_trainer.model(best_sample_latent_prev).flatten()
                best_to_prev_best_cossim = torch.nn.functional.cosine_similarity(best_sample_encoding, best_sample_encoding_prev, dim=0).item()
                best_to_prev_best_cossim = (best_to_prev_best_cossim + 1) / 2
                
        print("\nComputing final rewards")

        if cfg.ddpo_conf.reward_mode == "similarity-to-best-sample":
            ### Similarity with the best sample ### (for similarity-based reward)
            final_rewards = torch.nn.functional.cosine_similarity(human_encodings, best_sample_encoding.expand(human_encodings.shape))
            final_rewards = (final_rewards+1)/2
            print(f'Number of positive indices: {len(positive_indices)}')
            _, indices = torch.topk(final_rewards, len(positive_indices))
        
            common_elements = torch.isin(indices.cpu(), torch.from_numpy(positive_indices))
            print(f'Number of positive indices in top k: {common_elements.sum().item()}')
            # final_rewards = torch.softmax(final_rewards, dim=0)
        elif cfg.ddpo_conf.reward_mode == "similarity-to-all-positive":
            ### Similarity to all positive samples ###
            positive_sample_encodings = human_encodings[query_indices[positive_indices]]
            expand_dim = (positive_sample_encodings.shape[0], human_encodings.shape[0], human_encodings.shape[1])
            positive_sample_encodings = positive_sample_encodings.unsqueeze(1).expand(expand_dim)
            human_encodings = human_encodings.unsqueeze(0).expand(expand_dim)
            final_rewards = torch.nn.functional.cosine_similarity(human_encodings, positive_sample_encodings, dim=2).mean(0)
            final_rewards = (final_rewards+1)/2


        ### Ground truth human reward ### (for query everything)
        ai_indices = np.setdiff1d(np.arange(sd_features.shape[0]), np.array(query_indices)) # feature indices where AI reward is used

        # if using dummy human, get feedback for samples queried to the human for evaluation
        if (ai_indices.shape[0] > 0) and cfg.query_conf.feedback_agent == "ai":
            ############################################################
            # Get final rewards for DDPO training
            ############################################################
            # get ground truth human rewards
            gnd_truth_human_rewards = feedback_interface.query_batch(
                prompts=["amusement street"]*ai_indices.shape[0],
                image_batch=samples[ai_indices],
                query_indices=np.arange(ai_indices.shape[0]),
            )
            gnd_truth_human_rewards = torch.Tensor(gnd_truth_human_rewards)
        
        ############################################################
        # Train SD via DDPO
        ############################################################
        if human_encoder_trainer.n_calls_to_train >= cfg.human_encoder_conf.n_warmup_epochs:            
            print("Training DDPO...")
            ddpo_trainer.train_from_reward_labels(
                raw_rewards=final_rewards,
                logger=logger,
                epoch=loop,
            )
            n_ddpo_train_calls += 1
        else:
            print(f"Human encoder and error predictor warmup has not completed. Skipping DDPO training. Warmup step {human_encoder_trainer.n_calls_to_train}/{cfg.human_encoder_conf.n_warmup_epochs}")
        # end = time.time()
        # print(f"Training time = {end-start}")
        # input()

        # if model checkpoint should be saved in this iteration, save the best and positive latents
        if (loop % cfg.ddpo_conf.save_freq == 0) and accelerator.is_main_process:
            torch.save(
                best_noise_latent, 
                os.path.join(latent_save_dir, f"best_{loop}.pt")
            )
            torch.save(
                positive_noise_latents, 
                os.path.join(latent_save_dir, f"positives_{loop}.pt")
            )

        # increment loop
        loop += 1

        try:
            positive_to_best_cossim = final_rewards[query_indices[positive_indices]]
            negative_to_best_cossim = final_rewards[query_indices[negative_indices]]
            best_to_prev_best_cossim = best_to_prev_best_cossim if is_best_image_updated else None
            best_to_best_cossim_cur_and_prev_encoder if not is_best_image_updated else None
            eval_positive_to_best_cossim = predicted_cossim[query_indices[positive_indices]].mean().item() if predicted_cossim is not None else None
            eval_negative_to_best_cossim = predicted_cossim[query_indices[negative_indices]].mean().item() if predicted_cossim is not None else None

            accelerator.log({
                "human_feedback_total" : n_total_human_feedback,
                "n_human_encoder_training" : human_encoder_trainer.n_calls_to_train,
                "n_ddpo_training" : n_ddpo_train_calls,
                "n_positive_samples" : positive_indices.shape[0],
                "n_negative_samples" : negative_indices.shape[0],
                "train_positive_to_best_avg_cossim" : positive_to_best_cossim.mean(), # avg cossim between all positive images chosen by human this loop and the current best image
                "train_negative_to_best_avg_cossim" : negative_to_best_cossim.mean(), # avg cossim between all negative images chosen by human this loop and the current best image
                "eval_positive_to_best_avg_cossim" : eval_positive_to_best_cossim, # 
                "eval_negative_to_best_avg_cossim" : eval_negative_to_best_cossim, # 
                "is_best_image_updated" : int(is_best_image_updated), # whether the best image got updated this batch (1=yes, 0=no)
                "best_to_prev_best_avg_cossim" : best_to_prev_best_cossim, # cossim between new best image and previous best image using the most recent encoder (only logged if best image is updated this loop)
                "best_to_best_cossim" : best_to_best_cossim_cur_and_prev_encoder, # cossim between best image encoding by encoder this iteration and encoding by encoder in the previous iteration of the loop (only logged if best image is NOT updated this loop)
            })
        except Exception as error:
            print(error)
            print("Something went wrong with logging")
            breakpoint()

if __name__ == "__main__":
    main()